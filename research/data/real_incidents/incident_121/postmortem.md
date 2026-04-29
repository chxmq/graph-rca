# Google Incident

**Category:** Software
**Severity:** Medium
**Source:** https://status.cloud.google.com/incidents/mREMLwZFe3FuLLn3zfTw

## Description

On 13 October 2022 23:30 US/Pacific, there was an unexpected increase of incoming and logging traffic combined with a bug in Google’s internal streaming RPC library that triggered a deadlock and caused the Write API Streaming frontend to be overloaded. And BigQuery Storage WriteAPI observed elevated error rates in the US Multi-Region for a period of 5 hours.

## Root Cause

On 13 October 2022 23:30 US/Pacific, there was an unexpected increase of incoming and logging traffic combined with a bug in Google’s internal streaming RPC library that triggered a deadlock and cause...
