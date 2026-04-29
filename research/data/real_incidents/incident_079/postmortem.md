# Amazon Incident

**Category:** Infrastructure
**Severity:** Critical
**Source:** https://aws.amazon.com/message/11201/

## Description

Scaling the front-end cache fleet for Kinesis caused all of the servers in the fleet to exceed the maximum number of threads allowed by an operating system configuration. Multiple critical downstream services affected, from Cognito to Lambda to CloudWatch.

## Root Cause

Scaling the front-end cache fleet for Kinesis caused all of the servers in the fleet to exceed the maximum number of threads allowed by an operating system configuration
