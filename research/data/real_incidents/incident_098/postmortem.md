# Cloudflare Incident

**Category:** Software
**Severity:** Critical
**Source:** https://web.archive.org/web/20211006055154/https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/

## Description

A CPU exhaustion was caused by a single WAF rule that contained a poorly written regular expression that ended up creating excessive backtracking. This rule was deployed quickly to production and a series of events lead to a global 27 minutes downtime of the Cloudflare services.

## Root Cause

A single waf rule that contained a poorly written regular expression that ended up creating excessive backtracking
