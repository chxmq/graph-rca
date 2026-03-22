# Tarsnap Incident

**Category:** Cloud
**Severity:** Medium
**Source:** https://mail.tarsnap.com/tarsnap-announce/msg00035.html

## Description

A batch job which scans for unused blocks in Amazon S3 and marks them to be freed encountered a condition where all retries for freeing certain blocks would fail. The batch job logs its actions to local disk and this log grew without bound. When the filesystem filled, this caused other filesystem writes to fail, and the Tarsnap service stopped. Manually removing the log file restored service.

## Root Cause

A batch job which scans for unused blocks in Amazon S3 and marks them to be freed encountered a condition where all retries for freeing certain blocks would fail
