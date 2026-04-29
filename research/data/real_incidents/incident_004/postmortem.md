# TravisCI Worker Outage 2017
Root Cause: New worker bash execution mode change + missing Docker tag 'v2.5.0' on Docker Hub preventing successful rollback.
Timeline:
- Feb 2: v2.6.2 Rollout
- Feb 3: Identified false failures, start rollback
- Feb 4: Discovered rollback not working (instances running v2.6.2)
- Feb 5 00:31: Rollback completed after fixing Docker tag.
