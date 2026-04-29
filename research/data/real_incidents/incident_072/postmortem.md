# Allegro Incident

**Category:** Infrastructure
**Severity:** Medium
**Source:** https://allegro.tech/2015/01/allegro-cast-post-mortem.html

## Description

The [Allegro](https://web.archive.org/web/20211204232004/https://allegro.pl/) platform suffered a failure of a subsystem responsible for asynchronous distributed task processing. The problem affected many areas, e.g. features such as purchasing numerous offers via cart and bulk offer editing (including price list editing) did not work at all. Moreover, it partially failed to send daily newsletter with new offers. Also some parts of internal administration panel were affected.

## Root Cause

The [Allegro](https://web
