# Heroku Incident

**Category:** Infrastructure
**Severity:** Medium
**Source:** https://engineering.heroku.com/blogs/2017-02-15-filesystem-corruption-on-heroku-dynos/

## Description

An upgrade silently disabled a check that was meant to prevent filesystem corruption in running containers. A subsequent deploy caused filesystem corruption in running containers.

## Root Cause

An upgrade silently disabled a check that was meant to prevent filesystem corruption in running containers
