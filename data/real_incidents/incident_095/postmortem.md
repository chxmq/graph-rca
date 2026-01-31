# CircleCI Incident

**Category:** Infrastructure
**Severity:** Medium
**Source:** https://circleci.statuspage.io/incidents/hr0mm9xmm3x6

## Description

A GitHub outage and recovery caused an unexpectedly large incoming load. For reasons that aren't specified, a large load causes CircleCI's queue system to slow down, in this case to handling one transaction per minute.

## Root Cause

A GitHub outage and recovery caused an unexpectedly large incoming load
