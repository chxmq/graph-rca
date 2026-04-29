# AppNexus Incident

**Category:** Database
**Severity:** Medium
**Source:** https://web.archive.org/web/20250505112812/https://medium.com/xandr-tech/2013-09-17-outage-postmortem-586b19ae4307

## Description

A double free revealed by a database update caused all "impression bus" servers to crash simultaneously. This wasn't caught in staging and made it into production because a time delay is required to trigger the bug, and the staging period didn't have a built-in delay.

## Root Cause

A double free revealed by a database update caused all "impression bus" servers to crash simultaneously
