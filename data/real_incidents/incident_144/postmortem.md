# Mailgun Incident

**Category:** Database
**Severity:** Medium
**Source:** https://status.mailgun.com/incidents/p9nxxql8g9rh

## Description

Secondary MongoDB servers became overloaded and while troubleshooting accidentally pushed a change that sent all secondary traffic to the primary MongoDB server, overloading it as well and exacerbating the problem.

## Root Cause

Secondary MongoDB servers became overloaded and while troubleshooting accidentally pushed a change that sent all secondary traffic to the primary MongoDB server, overloading it as well and exacerbatin...
