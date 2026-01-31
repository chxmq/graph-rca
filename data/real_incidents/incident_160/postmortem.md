# Reddit Incident

**Category:** Network
**Severity:** Critical
**Source:** https://web.archive.org/web/20230727225235/https://www.reddit.com/r/RedditEng/comments/11xx5o0/you_broke_reddit_the_piday_outage/

## Description

Outage for over 5 hours when a critical Kubernetes cluster upgrade failed. The failure was caused by node metadata that changed between versions which brought down workload networking.

## Root Cause

Node metadata that changed between versions which brought down workload networking
