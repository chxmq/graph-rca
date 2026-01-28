# Fix for SSL_CERT_FILE environment variable issue on some servers
import os
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

import ollama
import logging
from datetime import datetime
from app.models.parsing_data_models import LogEntry, LogChain

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LLAMA = "llama3.2:3b"
QWEN = "qwen2.5-coder:3b"

class LogParser:
    def __init__(self,model:str="llama3.2:3b", timeout:float=30.0):
        try:
            self.model = model
            self.timeout = timeout
            self.ollama_client = ollama.Client(host='http://localhost:11435', timeout=timeout)
            self.ollama_options = ollama.Options(temperature=0.2)
            self.system_prompt = f"You are an expert in log parsing. You are given a log entry and a pydantic model. Extract and fill the fields of the model with the information from the log entry."
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at localhost:11435. "
                f"Make sure the Ollama service is running and the model '{model}' is installed. "
                f"Check with: docker exec -it graph-rca-ollama ollama list. "
                f"Error: {str(e)}"
            )
        
    def extract_log_info_by_llm(self, log_entry: str) -> LogEntry:
        """Extract log information using the specified language model"""
        try:
            user_prompt = f"""Parse this log entry into JSON format:
            {log_entry}
            
            Required fields:
            - timestamp (ISO 8601 format)
            - message (original log message)
            - level (log severity level)
            
            Optional fields (include ONLY if present):
            - pid (process ID as string)
            - component (source component/module)
            - error_code (error code as string)
            - username
            - ip_address
            - group
            - trace_id
            - request_id
            
            Return empty strings for missing fields. Maintain original case for field values.
            """
            
            response = self.ollama_client.generate(
                model=self.model,
                prompt=user_prompt,
                system=self.system_prompt,
                options=self.ollama_options,
                format="json"
            )

            # Handle empty responses
            if not response or not response.response.strip():
                raise ValueError("Empty response from language model")
                
            try:
                # Parse and validate the JSON response
                parsed = LogEntry.model_validate_json(response.response)
                
                # Validate mandatory fields
                if not all([parsed.timestamp, parsed.message, parsed.level]):
                    raise ValueError("Missing required fields in parsed entry")
                    
                return parsed
                
            except Exception as e:
                logger.error(f"JSON validation error: {e}")
                logger.error(f"Problematic JSON: {response.response}")
                raise

        except Exception as e:
            raise RuntimeError(f"Failed to extract log info: {str(e)}")
        
    def parse_log_from_file(self, log_file: str) -> LogChain:
        """Parse log entries from a log file."""
        try:
            # Temporary extension check for testing
            if not log_file.endswith((".txt", ".log", ".out", ".err")):
                raise ValueError("Unsupported file type")
            
            with open(log_file, "r") as f:
                log_data = f.read()
            
            if not log_data.strip():
                raise ValueError("Empty log file")
            
            return self.parse_log(log_data)
            
        except ValueError as e:
            # Re-raise validation errors as-is
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to parse log file: {str(e)}")
    
    def parse_log(self, log_data: str) -> LogChain:
        """Parse log entries from a string using LLM"""
        try:
            if not log_data:
                raise ValueError("Empty log data provided")
                
            log_data_split = log_data.split("\n")
            log_entries = []
            
            total_lines = len(log_data_split)
            logger.info(f"[◆] Processing {total_lines} log lines")

            for idx, log in enumerate(log_data_split, 1):
                if not log.strip():
                    continue
                    
                try:
                    logger.info(f"  [►] Line {idx}/{total_lines}: {log[:80]}...")
                    entry = self.extract_log_info_by_llm(log)
                    log_entries.append(entry)
                    logger.info(f"  [✓] Parsed: {entry.level} - {entry.message[:50]}...")
                    
                except Exception as e:
                    logger.error(f"  [✗] Failed line {idx}: {str(e)}")
                    continue
            
            if not log_entries:
                raise ValueError("No valid log entries found after LLM processing")
                
            return LogChain(log_chain=log_entries)
            
        except Exception as e:
            raise RuntimeError(f"Failed to parse log data: {str(e)}")
