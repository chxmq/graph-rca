# Discord Incident (2020-01-01)
Root Cause: Automated GCP migration of a Redis primary caused it to drop offline, triggering cascading failures in how Discord handles Redis failover.
Category: Infrastructure/Redis
Source: https://status.discordapp.com/incidents/qk9cdgnqnhcn
