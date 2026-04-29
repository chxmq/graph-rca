"""
Pydantic models for the Graph-RCA pipeline.

Contains all data models for:
- Log parsing (LogEntry, LogChain)
- DAG construction (DAGNode, DAG)
- Context extraction (Context)
- RAG responses (SummaryResponse, SolutionQuery)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime, timezone
from uuid import uuid4


# ---------------------------------------------------------------------------
# Log Parsing
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    """Complete log entry — the only model actively used by the pipeline."""
    timestamp: datetime = Field(description="Timestamp of the log entry")
    message: str = Field(description="Log message content")
    level: str = Field(description="Log level")
    pid: str = Field("", description="Process ID of the application")
    component: str = Field("", description="Component/module generating the log")
    error_code: str = Field("", description="Error code if applicable")
    username: str = Field("", description="Username of the user generating the log")
    ip_address: str = Field("", description="IP address of the user generating the log")
    group: str = Field("", description="Group of the log entry")
    trace_id: str = Field("", description="Distributed tracing ID")
    request_id: str = Field("", description="Request ID of the user generating the log")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime | str) -> datetime:
        """Parse the timestamp and force tz-awareness.

        Naive timestamps (without tzinfo) are assumed to be UTC.  Mixing
        naive and aware datetimes within a single LogChain would later
        explode in ``GraphGenerator._set_parent_child_relationships`` when
        sorting, so we normalise here once.
        """
        parsed: datetime | None = None
        if isinstance(v, datetime):
            parsed = v
        else:
            for fmt in (
                None,
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%b %d %H:%M:%S",
            ):
                try:
                    if fmt is None:
                        parsed = datetime.fromisoformat(v)
                    else:
                        parsed = datetime.strptime(v, fmt)
                        if fmt == "%b %d %H:%M:%S":
                            parsed = parsed.replace(year=datetime.now(timezone.utc).year)
                    break
                except (ValueError, TypeError):
                    continue
        if parsed is None:
            raise ValueError(f"timestamp '{v}' is not a valid ISO 8601 or common datetime format")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed


class LogChain(BaseModel):
    """Collection of log entries and parse metadata."""
    log_chain: list[LogEntry] = Field(description="List of log entries")
    total_lines: int = Field(0, description="Total lines received for parsing")
    parsed_lines: int = Field(0, description="Successfully parsed lines")
    parse_errors: list[str] = Field(default_factory=list, description="Per-line parse failures")


# ---------------------------------------------------------------------------
# DAG Construction
# ---------------------------------------------------------------------------

class DAGNode(BaseModel):
    """Node in a DAG-compatible correlation graph."""
    id: str = Field(description="Unique identifier of the node")
    parent_ids: list[str] = Field(default_factory=list, description="Unique identifiers of parent nodes")
    children: list[str] = Field(description="List of unique identifiers of the children nodes")
    log_entry: LogEntry = Field(description="Log entry information")


class DAG(BaseModel):
    """DAG-compatible correlation graph of log entries."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    nodes: list[DAGNode] = Field(description="List of nodes in the graph")
    root_id: str = Field(description="Unique identifier of the root node")
    root_cause: Optional[str] = Field(default=None, description="Root cause of the issue")
    leaf_ids: list[str] = Field(description="List of unique identifiers of the leaf nodes")


# ---------------------------------------------------------------------------
# Context Extraction
# ---------------------------------------------------------------------------

class Context(BaseModel):
    """Context information for the log chain"""
    dag_id: str = Field(description="Associated DAG identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Context creation timestamp")
    root_cause: str = Field(description="Root cause of the issue")
    causal_chain: list[str] = Field(description="Causal chain of the issue")


# ---------------------------------------------------------------------------
# RAG Responses
# ---------------------------------------------------------------------------

class SummaryResponse(BaseModel):
    summary: list[str] = Field(description="list of summary points extracted from logs")
    root_cause_expln: str = Field(description="Explanation of the identified root cause")
    severity: str = Field(description="Severity level of the issue")
    parse_failed: bool = Field(default=False, description="True if structured response parsing failed")


class SolutionQuery(BaseModel):
    context: str = Field(description="Context information for the query")
    query: str = Field(description="Generated or provided query text")
    response: str = Field(description="Solution response from the LLM")
    sources: list[str] = Field(default_factory=list, description="List of documentation sources used")
