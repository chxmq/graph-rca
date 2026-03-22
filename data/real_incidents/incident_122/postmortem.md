# Google Incident

**Category:** Infrastructure
**Severity:** Critical
**Source:** https://status.cloud.google.com/incidents/ow5i3PPK96RduMcb1SsW

## Description

A policy change containing blank fields triggered a null pointer exception in Service Control, Google's API management and control plane system. The code path that failed was not feature flag protected and lacked proper error handling. When the policy data replicated globally, it caused Service Control binaries to crash loop across all regions. While a red-button fix was deployed within 40 minutes, larger regions like us-central-1 experienced extended recovery times (up to 2h 40m) due to a thundering herd problem when Service Control tasks restarted, overloading the underlying Spanner infrastructure. The incident affected Google and Google Cloud APIs globally, with recovery times varying by product architecture.

## Root Cause

A thundering herd problem when service control tasks restarted, overloading the underlying spanner infrastructure
