# Postmortem — why Allegro went down
Direct cause: Special offer PLN 1 Honor 7C phones attracted 3x normal traffic.
Root Cause: Resource reservation deadlock. Services reserved more than needed, preventing autoscaling.
Timeline: 
- 11:55: Traffic spike, scaling required
- 11:58: Cluster resources exhausted due to reservations
- 12:05: Frontend service Opbox starts failing
- 12:20: Restoration after adding resources and shutting non-critical services.
