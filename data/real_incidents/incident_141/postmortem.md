# Kickstarter Incident

**Category:** Database
**Severity:** Medium
**Source:** https://web.archive.org/web/20170728131458/https://kickstarter.engineering/the-day-the-replication-died-e543ba45f262

## Description

Primary DB became inconsistent with all replicas, which wasn't detected until a query failed. This was caused by a MySQL bug which sometimes caused `order by` to be ignored.

## Root Cause

A mysql bug which sometimes caused `order by` to be ignored
