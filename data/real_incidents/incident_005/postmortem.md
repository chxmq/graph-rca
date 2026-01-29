# Railway Outage Oct 28, 2025
Root Cause: Postgres table lock from non-concurrent index creation.
Impact: Table locked for 30 mins, exhausted 100% of connection pool slots including administrative slots (PgBouncer overflow).
Timeline:
- 18:34: Migration live
- 18:36: Monitoring alerts
- 19:00: Migration completed, lock released, recovery starts.
