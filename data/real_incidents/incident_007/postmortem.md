# Joyent Manta Postmortem July 27, 2015
Root Cause: Blocked locks on metadata servers.
Conflict: PG autovacuum held a lock, and a separate query tried to take a global lock, blocking subsequent requests.
