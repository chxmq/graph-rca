# Google Incident

**Category:** Software
**Severity:** Medium
**Source:** https://status.cloud.google.com/incident/compute/17003#5660850647990272

## Description

Many changes to a rarely modified load balancer were applied through a very slow code path. This froze all public addressing changes for ~2 hours.

## Root Cause

Many changes to a rarely modified load balancer were applied through a very slow code path
