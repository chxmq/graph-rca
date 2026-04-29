# Sentry Outage July 2015
Root Cause: TXID wraparound in Postgres.
Impact: Autovacuum failed to keep up with high write volume, database stopped accepting writes.
