# Google Outage Dec 15, 2022
Root Cause: Null pointer exception during policy replication.
Trigger: Blank fields in a policy change were not handled, crashing the Service Control binary globally.
