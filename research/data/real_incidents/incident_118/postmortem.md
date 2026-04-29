# Google Incident

**Category:** Infrastructure
**Severity:** Medium
**Source:** https://gist.github.com/jomo/2bae3821acb433d0446d

## Description

A mail system emailed people more than 20 times. This happened because mail was sent with a batch cron job that sent mail to everyone who was marked as waiting for mail. This was a non-atomic operation and the batch job didn't mark people as not waiting until all messages were sent.

## Root Cause

A mail system emailed people more than 20 times
