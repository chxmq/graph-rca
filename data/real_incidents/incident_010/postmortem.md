# AWS S3 Outage Feb 28, 2017
Root Cause: Human error (Typo).
Command to remove servers was entered incorrectly, removing too many servers in US-EAST-1.
Impact: Massive cascading failure affecting S3, EC2, EBS, and hundreds of downstream customers.
