# GitHub Outage Nov 2021
Root Cause: Semaphore deadlock during table rename in schema migration.
Impact: Replicas crashed, creating a cascading failure on healthy nodes due to increased load.
