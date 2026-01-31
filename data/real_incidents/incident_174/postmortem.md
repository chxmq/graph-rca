# Stack Exchange Incident

**Category:** Infrastructure
**Severity:** Medium
**Source:** http://web.archive.org/web/20160720200842/https://stackstatus.net/post/147710624694/outage-postmortem-july-20-2016

## Description

Backtracking implementation in the underlying regex engine turned out to be very expensive for a particular post leading to health-check failures and eventual outage.

## Root Cause

Backtracking implementation in the underlying regex engine turned out to be very expensive for a particular post leading to health-check failures and eventual outage
