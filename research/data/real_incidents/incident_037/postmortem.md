# Twilio Incident (2013-07-19)
Root Cause: Redis network partition caused resync storm, crashing the master and leaving billing in read-only mode, causing retry over-billing.
Category: Database/Redis
Source: https://www.twilio.com/blog/2013/07/billing-incident-post-mortem-breakdown-analysis-and-root-cause.html
