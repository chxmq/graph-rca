# Yeller Incident

**Category:** Network
**Severity:** Medium
**Source:** https://web.archive.org/web/20201018145502/http://yellerapp.com/posts/2014-08-04-postmortem1.html

## Description

A network partition in a cluster caused some messages to get delayed, up to 6-7 hours. For reasons that aren't clear, a rolling restart of the cluster healed the partition. There's some suspicious that it was due to cached routes, but there wasn't enough logging information to tell for sure.

## Root Cause

Cached routes, but there wasn't enough logging information to tell for sure
