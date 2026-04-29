# Discord Incident

**Category:** Memory
**Severity:** Medium
**Source:** https://status.discordapp.com/incidents/dj3l6lw926kl

## Description

A flapping service lead to a thundering herd reconnecting to it once it came up. This lead to a cascading error where frontend services ran out of memory due to internal queues filling up.

## Root Cause

Internal queues filling up
