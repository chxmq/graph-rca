# NASA Incident

**Category:** Software
**Severity:** Medium
**Source:** https://web.archive.org/web/20161230103247/https://research.microsoft.com/en-us/um/people/mbj/Mars_Pathfinder/Authoritative_Account.html

## Description

NASA's Mars Pathfinder spacecraft experienced system resets a few days after landing on Mars (1997).  Debugging features were remotely enabled until the cause was found: a [priority inversion](https://en.wikipedia.org/wiki/Priority_inversion) problem in the VxWorks operating system.  The OS software was remotely patched (all the way to Mars) to fix the problem by adding priority inheritance to the task scheduler.

## Root Cause

NASA's Mars Pathfinder spacecraft experienced system resets a few days after landing on Mars (1997)
