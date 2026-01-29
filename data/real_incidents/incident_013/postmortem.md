# Gitlab Outage Jan 31, 2017
Root Cause: Database directory deletion on primary.
Context: High load caused replication lag. Admin tried to wipe staging but ran `rm -rf` on the wrong terminal (primary).
Outcome: 6 hours of data lost across issues, merge requests, users.
