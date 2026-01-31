# Stackdriver Incident

**Category:** Database
**Severity:** Medium
**Source:** https://www.stackdriver.com/post-mortem-october-23-stackdriver-outage/

## Description

In October 2013, [Stackdriver](https://www.stackdriver.com/), experienced an outage, when its Cassandra cluster crashed. Data published by various services into a message bus was being injested into the Cassandra cluster. When the cluster failed, the failure percolated to various producers, that ended up blocking on queue insert operations, eventually leading to the failure of the entire application.

## Root Cause

In October 2013, [Stackdriver](https://www
