# Cloudflare Outage July 2, 2019
Root Cause: CPU exhaustion caused by a single WAF rule regular expression that triggered catastrophic backtracking.
Regex: `(?:(?:"|'|\]|\}|\d).*|.*(?:"|'|\]|\}|\d))$` (Simplified example from report)
Timeline:
- 13:42: WAF rule deployed
- 13:43: Global CPU spike detected
- 14:02: Global WAF disable implemented
- 14:09: Restoration begins
