# CrowdStrike Incident

**Category:** Memory
**Severity:** Medium
**Source:** https://www.crowdstrike.com/falcon-content-update-remediation-and-guidance-hub/

## Description

A Content update containing undetected errors was deployed due to a bug in the Content Validator in the deployment stage. This problematic content caused an out-of-bounds memory read, resulting in a Windows operating system crash (BSOD) on 8.5 million Windows machines. The update was reverted within 78 minutes, but the incident highlighted the need for improved validation and testing processes.

## Root Cause

A bug in the content validator in the deployment stage
