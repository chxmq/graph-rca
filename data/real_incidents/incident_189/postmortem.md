# Zerodha Incident

**Category:** Infrastructure
**Severity:** Medium
**Source:** https://zerodha.com/marketintel/bulletin/229363/post-mortem-of-technical-issue-august-29-2019

## Description

The Order Management System (OMS) provided to Zerodha, a stock broker, collapsed when an order for 1M units of a penny stock was divided into more than 0.1M individual trades against the typical few hundreds, triggering a collapse of the OMS, which was not encountered prior by its provider - Refinitiv (formerly Thomson Reuters), a subsidiary of the London Stock Exchange.

## Root Cause

The Order Management System (OMS) provided to Zerodha, a stock broker, collapsed when an order for 1M units of a penny stock was divided into more than 0
