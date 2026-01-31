# trivago Incident

**Category:** Security
**Severity:** Medium
**Source:** https://tech.trivago.com/2021/10/05/postmortem-removing-all-users-from-github.com/trivago/

## Description

Due to a human error, all engineers lost access to the central source code management platform (GitHub organization). An Azure Active Directory Security group controls the access to the GitHub organization. This group was removed during the execution of a manual and repetitive task.

## Root Cause

A human error, all engineers lost access to the central source code management platform (github organization)
