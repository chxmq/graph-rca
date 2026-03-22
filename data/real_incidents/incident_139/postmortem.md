# Joyent Incident

**Category:** Database
**Severity:** Critical
**Source:** https://web.archive.org/web/20220528044329/https://www.joyent.com/blog/manta-postmortem-7-27-2015

## Description

Operations on Manta were blocked because a lock couldn't be obtained on their PostgreSQL metadata servers. This was due to a combination of PostgreSQL's transaction wraparound maintenance taking a lock on something, and a Joyent query that unnecessarily tried to take a global lock.

## Root Cause

A combination of postgresql's transaction wraparound maintenance taking a lock on something, and a joyent query that unnecessarily tried to take a global lock
