# Cloudflare Outage July 17, 2020
Root Cause: BGP Route Leak in Atlanta. 
Config Error: Removed prefix-list 6-SITE-LOCAL from the 'from' clause of a policy, causing it to match and export ALL routes.
Timeline:
- 20:25: Backbone link Newark-Chicago lost
- 21:12: Atlanta config change (Start of outage)
- 21:39: Atlanta router disabled, service restored
