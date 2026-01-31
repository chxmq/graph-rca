# Stripe Incident

**Category:** Database
**Severity:** Medium
**Source:** https://support.stripe.com/questions/outage-postmortem-2015-10-08-utc

## Description

Manual operations are regularly executed on production databases. A manual operation was done incorrectly (missing dependency), causing the Stripe API to go down for 90 minutes.

## Root Cause

Manual operations are regularly executed on production databases
