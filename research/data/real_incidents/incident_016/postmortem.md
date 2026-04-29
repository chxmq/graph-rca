# Etsy Outage 2012
Root Cause: Max value of signed 32-bit INT reached for primary keys.
Outcome: Database stopped accepting new records, requiring migration to BigInt.
