import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.log_parser import LogParser


def _llm_payload(line_no: int) -> dict:
    return {
        "timestamp": f"2023-01-01T00:00:{line_no:02d}",
        "message": f"line {line_no}",
        "level": "INFO",
        "pid": "1234",
        "component": "",
        "error_code": "",
        "username": "",
        "ip_address": "",
        "group": "",
        "trace_id": "",
        "request_id": "",
    }


@pytest.fixture
def mock_ollama():
    with patch("ollama.Client") as mock_client, patch("ollama.AsyncClient") as mock_async:
        mock_resp = Mock()
        mock_resp.response = json.dumps([_llm_payload(0)])
        mock_async.return_value.generate = AsyncMock(return_value=mock_resp)
        yield mock_client, mock_async


@pytest.mark.asyncio
async def test_parse_valid_log(mock_ollama):
    parser = LogParser()
    result = await parser.parse_log_async("2023-01-01 INFO Test message")
    assert len(result.log_chain) == 1
    assert result.log_chain[0].message == "line 0"


@pytest.mark.asyncio
async def test_empty_log_raises(mock_ollama):
    parser = LogParser()
    with pytest.raises(ValueError):
        await parser.parse_log_async("")


@pytest.mark.asyncio
async def test_multi_batch_parsing():
    """30 lines → at least 2 batches (batch_size=16)."""
    with patch("ollama.Client"), patch("ollama.AsyncClient") as mock_async:
        # Each batch returns matching number of items
        async def _fake_generate(model, prompt, system, options, format):
            # parser_batch_prompt embeds the input list as JSON; we don't
            # introspect the prompt but echo plausible items
            line_count = prompt.count('"line ')
            if line_count == 0:
                # Fallback: count newlines in the embedded array
                line_count = max(1, prompt.count("\n  \"") )
            payload = [_llm_payload(i) for i in range(line_count)]
            return Mock(response=json.dumps(payload))

        mock_async.return_value.generate = AsyncMock(side_effect=_fake_generate)
        parser = LogParser()
        log_data = "\n".join(f"line {i}" for i in range(30))
        result = await parser.parse_log_async(log_data)
        # Both batches should produce entries (non-empty)
        assert len(result.log_chain) >= 1
        assert len(result.parse_errors) == 0


@pytest.mark.asyncio
async def test_one_batch_failure_records_per_line_errors():
    """Failing batch produces parse_errors for every line in that batch."""
    with patch("ollama.Client"), patch("ollama.AsyncClient") as mock_async:
        call = {"n": 0}

        async def _fake_generate(**_):
            call["n"] += 1
            if call["n"] == 1:
                raise RuntimeError("boom on batch 1")
            return Mock(response=json.dumps([_llm_payload(0)]))

        mock_async.return_value.generate = AsyncMock(side_effect=_fake_generate)
        parser = LogParser()
        log_data = "\n".join(f"line {i}" for i in range(20))
        result = await parser.parse_log_async(log_data)
        assert len(result.parse_errors) > 0
        assert any("boom" in err for err in result.parse_errors)


def test_parse_log_inside_event_loop_raises(mock_ollama):
    """sync parse_log should refuse to run inside an active asyncio loop."""
    import asyncio

    async def _runner():
        parser = LogParser()
        return parser.parse_log("hello")

    with pytest.raises(RuntimeError, match="cannot run inside an active event loop"):
        asyncio.run(_runner())
