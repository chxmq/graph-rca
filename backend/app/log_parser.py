import ollama
import logging
import json
import asyncio
from typing import Callable, Awaitable
from app.models import LogEntry, LogChain
from app.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_TEMPERATURE
from app._ollama import response_text
from app.prompts import PARSER_SYSTEM_PROMPT, parser_batch_prompt

logger = logging.getLogger(__name__)


class LogParser:
    def __init__(self, model: str = OLLAMA_MODEL, timeout: float = OLLAMA_TIMEOUT, batch_size: int = 16):
        try:
            self.model = model
            self.timeout = timeout
            self.batch_size = batch_size
            self.ollama_async_client = ollama.AsyncClient(host=OLLAMA_HOST, timeout=timeout)
            self.ollama_options = ollama.Options(temperature=OLLAMA_TEMPERATURE)
            self.system_prompt = PARSER_SYSTEM_PROMPT
        except (ConnectionError, ConnectionRefusedError) as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {OLLAMA_HOST}. "
                f"Make sure the Ollama service is running and the model '{model}' is installed. "
                f"Error: {str(e)}"
            ) from e
        except Exception as e:
            # Pydantic config errors, missing env vars, etc. are NOT connection issues —
            # let them bubble as-is so callers don't get a misleading 503.
            raise RuntimeError(
                f"Failed to initialize LogParser. Error: {str(e)}"
            ) from e
        
    def parse_log_from_file(self, log_file: str) -> LogChain:
        """
        Parse log entries from a file for CLI/experiment callers.

        Production FastAPI routes use parse_log_async directly to avoid nesting
        asyncio.run inside the server event loop.
        """
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                log_data = f.read()

            if not log_data.strip():
                raise ValueError("Empty log file")

            return self.parse_log(log_data)

        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to parse log file: {str(e)}") from e

    def parse_log(self, log_data: str) -> LogChain:
        """Sync entry point — used by CLI/experiment scripts only.

        Raises RuntimeError if called from within a running event loop;
        production FastAPI code paths should use ``parse_log_async``
        directly to avoid nested asyncio.run.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise RuntimeError(
                "parse_log() cannot run inside an active event loop; "
                "use await parse_log_async(...) instead."
            )
        try:
            return asyncio.run(self.parse_log_async(log_data))
        except ValueError:
            raise
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to parse log data: {exc}") from exc

    async def parse_log_async(
        self,
        log_data: str,
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
        concurrency: int = 4,
        source_total_lines: int | None = None,
    ) -> LogChain:
        try:
            if not log_data:
                raise ValueError("Empty log data provided")

            log_data_split = log_data.split("\n")
            total_lines = source_total_lines if source_total_lines is not None else len(log_data_split)
            logger.info("[◆] Processing %d log lines", total_lines)

            parse_errors: list[str] = []
            non_empty_logs = []
            for idx, log in enumerate(log_data_split, 1):
                stripped = log.strip()
                if not stripped:
                    continue
                if len(stripped) > 4096:
                    parse_errors.append(f"line {idx}: line too long, truncated to 4096 chars")
                    stripped = stripped[:4096]
                non_empty_logs.append((idx, stripped))
            batches = [non_empty_logs[i : i + self.batch_size] for i in range(0, len(non_empty_logs), self.batch_size)]
            logger.info("[◆] Processing %d batches (batch_size=%d)", len(batches), self.batch_size)

            semaphore = asyncio.Semaphore(concurrency)
            parsed_batches: list[list[LogEntry]] = [list() for _ in batches]
            processed = 0

            async def _process_batch(batch_idx: int, batch: list[tuple[int, str]]) -> None:
                nonlocal processed
                async with semaphore:
                    try:
                        parsed_batches[batch_idx] = await self._extract_batch_by_llm_async(
                            batch,
                            parse_errors=parse_errors,
                        )
                    except Exception as e:
                        logger.error("  [✗] Failed batch %d parse: %s", batch_idx + 1, e)
                        parse_errors.extend([f"line {line_no}: {e}" for line_no, _ in batch])
                    finally:
                        processed += 1
                        if progress_callback:
                            await progress_callback(processed, len(batches))

            await asyncio.gather(*[_process_batch(i, batch) for i, batch in enumerate(batches)])
            log_entries = [entry for batch in parsed_batches for entry in batch]
            if not log_entries:
                raise ValueError("No valid log entries found after LLM processing")
            return LogChain(
                log_chain=log_entries,
                total_lines=total_lines,
                parsed_lines=len(log_entries),
                parse_errors=parse_errors,
            )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to parse log data: {str(e)}") from e

    async def _extract_batch_by_llm_async(
        self,
        indexed_batch: list[tuple[int, str]],
        parse_errors: list[str],
    ) -> list[LogEntry]:
        lines = [line for _, line in indexed_batch]
        prompt = parser_batch_prompt(lines)
        response = await self.ollama_async_client.generate(
            model=self.model,
            prompt=prompt,
            system=self.system_prompt,
            options=self.ollama_options,
            format="json",
        )

        llm_text = response_text(response)
        if not llm_text:
            raise ValueError("Empty response from language model for batch")

        parsed = json.loads(llm_text)
        if not isinstance(parsed, list):
            raise ValueError("Batch response is not a JSON array")

        valid_entries: list[LogEntry] = []
        for i, item in enumerate(parsed):
            try:
                entry = LogEntry.model_validate(item)
                valid_entries.append(entry)
                line_no = indexed_batch[i][0] if i < len(indexed_batch) else "?"
                logger.debug("  [✓] Parsed line %s: %s - %s...", line_no, entry.level, entry.message[:50])
            except Exception as exc:
                logger.error("  [✗] Invalid parsed item at batch index %d: %s", i, exc)
                line_no = indexed_batch[i][0] if i < len(indexed_batch) else "?"
                parse_errors.append(f"line {line_no}: invalid LLM item ({exc})")
        if len(parsed) > len(indexed_batch):
            for extra_idx in range(len(indexed_batch), len(parsed)):
                parse_errors.append(
                    f"line ?: extra LLM output item at batch index {extra_idx}"
                )
        if len(parsed) < len(indexed_batch):
            for missing_idx in range(len(parsed), len(indexed_batch)):
                line_no = indexed_batch[missing_idx][0]
                parse_errors.append(f"line {line_no}: LLM omitted output item")
        return valid_entries
