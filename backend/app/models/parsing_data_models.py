from pydantic import BaseModel, Field
from typing import Optional

"""This module contains Pydantic models for parsing log entries"""


class LogEntry(BaseModel):
    """Complete log entry — the only model actively used by the pipeline."""
    timestamp: str = Field(description="Timestamp of the log entry")
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


class LogChain(BaseModel):
    """Collection of log entries"""
    log_chain: list[LogEntry] = Field(description="List of log entries")
