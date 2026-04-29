# Azure Leap Day Outage 2012
Root Cause: Certificate creation bug. Added +1 year to today's date (Feb 29) resulting in invalid Feb 29, 2013 expiration.
Impact: Global Azure services unavailable for ~24 hours as certificates were rejected.
Timeline:
- 00:00 UTC Feb 29: Outage begins as new certs are generated
- 01:00 UTC: Engineering alerted
- Restoration took most of the day across regions.
