# Heroku Incident

**Category:** Software
**Severity:** Medium
**Source:** https://status.heroku.com/incidents/642?postmortem

## Description

Having a system that requires scheduled manual updates resulted in an error which caused US customers to be unable to scale, stop or restart dynos, or route HTTP traffic, and also prevented all customers from being able to deploy.

## Root Cause

Having a system that requires scheduled manual updates resulted in an error which caused US customers to be unable to scale, stop or restart dynos, or route HTTP traffic, and also prevented all custom...
