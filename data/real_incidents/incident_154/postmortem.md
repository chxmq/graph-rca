# OpenAI Incident

**Category:** Database
**Severity:** Medium
**Source:** https://web.archive.org/web/20240426015133/https://openai.com/blog/march-20-chatgpt-outage

## Description

Queues for requests and responses in a Redis cache became corrupted and out of sequence, leading to some requests revealing other people's user data to some users, including app activity data and some billing info.

## Root Cause

Queues for requests and responses in a Redis cache became corrupted and out of sequence, leading to some requests revealing other people's user data to some users, including app activity data and some...
