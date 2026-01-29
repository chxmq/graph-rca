from collect_incidents import IncidentDataCollector
import json
import os

def populate():
    collector = IncidentDataCollector()
    
    # 1. Allegro - Resource Reservation Deadlock (2018)
    print("Populating Allegro 2018...")
    collector.create_incident_directory("001")
    collector.save_metadata("001", {
        "company": "Allegro",
        "category": "Configuration/Scaling",
        "date": "2018-07-18",
        "severity": "Critical"
    })
    collector.save_ground_truth("001", {
        "incident_id": "001",
        "source": "https://blog.allegro.tech/2018/08/postmortem-why-allegro-went-down.html",
        "root_cause": "Misconfiguration in cluster resource management caused services to reserve excessive resources (CPU/RAM) they didn't use, preventing new instances from starting despite available physical capacity.",
        "root_cause_timestamp": "2018-07-18T11:55:00Z",
        "category": "Configuration",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("001", """# Postmortem — why Allegro went down
Direct cause: Special offer PLN 1 Honor 7C phones attracted 3x normal traffic.
Root Cause: Resource reservation deadlock. Services reserved more than needed, preventing autoscaling.
Timeline: 
- 11:55: Traffic spike, scaling required
- 11:58: Cluster resources exhausted due to reservations
- 12:05: Frontend service Opbox starts failing
- 12:20: Restoration after adding resources and shutting non-critical services.
""")

    # 2. Cloudflare - Edge Router Crash (2019)
    print("Populating Cloudflare 2019...")
    collector.create_incident_directory("002")
    collector.save_metadata("002", {
        "company": "Cloudflare",
        "category": "Network/Software",
        "date": "2019-07-02",
        "severity": "Critical"
    })
    collector.save_ground_truth("002", {
        "incident_id": "002",
        "source": "https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/",
        "root_cause": "A poorly written regular expression in a WAF rule caused CPU exhaustion (catastrophic backtracking) on edge routers.",
        "root_cause_timestamp": "2019-07-02T13:42:00Z",
        "category": "Network",
        "severity": "Critical",
        "logs_available": True,
        "root_cause_in_logs": True
    })
    collector.save_postmortem("002", """# Cloudflare Outage July 2, 2019
Root Cause: CPU exhaustion caused by a single WAF rule regular expression that triggered catastrophic backtracking.
Regex: `(?:(?:\"|'|\]|\}|\\d).*|.*(?:\"|'|\]|\}|\\d))$` (Simplified example from report)
Timeline:
- 13:42: WAF rule deployed
- 13:43: Global CPU spike detected
- 14:02: Global WAF disable implemented
- 14:09: Restoration begins
""")

    # 3. Cloudflare - Atlanta BGP Leak (2020)
    print("Populating Cloudflare 2020...")
    collector.create_incident_directory("003")
    collector.save_metadata("003", {
        "company": "Cloudflare",
        "category": "Network/BGP",
        "date": "2020-07-17",
        "severity": "Critical"
    })
    collector.save_ground_truth("003", {
        "incident_id": "003",
        "source": "https://blog.cloudflare.com/cloudflare-outage-on-july-17-2020/",
        "root_cause": "Configuration typo in Atlanta router: deactivating a prefix-list instead of the term leaked all BGP routes with high local-preference.",
        "root_cause_timestamp": "2020-07-17T21:12:00Z",
        "category": "Network",
        "severity": "Critical",
        "logs_available": True,
        "root_cause_in_logs": True
    })
    collector.save_postmortem("003", """# Cloudflare Outage July 17, 2020
Root Cause: BGP Route Leak in Atlanta. 
Config Error: Removed prefix-list 6-SITE-LOCAL from the 'from' clause of a policy, causing it to match and export ALL routes.
Timeline:
- 20:25: Backbone link Newark-Chicago lost
- 21:12: Atlanta config change (Start of outage)
- 21:39: Atlanta router disabled, service restored
""")

    # 4. TravisCI - Worker Rollback Failure (2017)
    print("Populating TravisCI 2017...")
    collector.create_incident_directory("004")
    collector.save_metadata("004", {
        "company": "TravisCI",
        "category": "CI-CD/Tools",
        "date": "2017-02-02",
        "severity": "High"
    })
    collector.save_ground_truth("004", {
        "incident_id": "004",
        "source": "https://www.traviscistatus.com/incidents/sxrh0l46czqn",
        "root_cause": "Worker v2.6.2 change in bash exit code handling (login shell) caused false failures; rollback to v2.5.0 failed due to missing Docker Hub tag.",
        "root_cause_timestamp": "2017-02-02T12:00:00Z",
        "category": "CI-CD",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("004", """# TravisCI Worker Outage 2017
Root Cause: New worker bash execution mode change + missing Docker tag 'v2.5.0' on Docker Hub preventing successful rollback.
Timeline:
- Feb 2: v2.6.2 Rollout
- Feb 3: Identified false failures, start rollback
- Feb 4: Discovered rollback not working (instances running v2.6.2)
- Feb 5 00:31: Rollback completed after fixing Docker tag.
""")

    # 5. Railway - DB Index Lock (2025)
    print("Populating Railway 2025...")
    collector.create_incident_directory("005")
    collector.save_metadata("005", {
        "company": "Railway",
        "category": "Database/Locking",
        "date": "2025-10-28",
        "severity": "Critical"
    })
    collector.save_ground_truth("005", {
        "incident_id": "005",
        "source": "https://blog.railway.com/p/incident-report-oct-28th-2025",
        "root_cause": "Database index creation on a 1B record table without CONCURRENTLY option locked the table, exhausting all connection slots.",
        "root_cause_timestamp": "2025-10-28T18:34:00Z",
        "category": "Database",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("005", """# Railway Outage Oct 28, 2025
Root Cause: Postgres table lock from non-concurrent index creation.
Impact: Table locked for 30 mins, exhausted 100% of connection pool slots including administrative slots (PgBouncer overflow).
Timeline:
- 18:34: Migration live
- 18:36: Monitoring alerts
- 19:00: Migration completed, lock released, recovery starts.
""")

    # 6. Azure - Leap Year Cert (2012)
    print("Populating Azure 2012...")
    collector.create_incident_directory("006")
    collector.save_metadata("006", {
        "company": "Azure",
        "category": "Software/Time",
        "date": "2012-02-29",
        "severity": "Critical"
    })
    collector.save_ground_truth("006", {
        "incident_id": "006",
        "source": "https://azure.microsoft.com/en-us/blog/summary-of-windows-azure-service-disruption-on-feb-29th-2012/",
        "root_cause": "Incorrect date arithmetic (adding 1 year to Feb 29) created certificates with an invalid expiration date (Feb 29, 2013).",
        "root_cause_timestamp": "2012-02-29T00:00:00Z",
        "category": "Software",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("006", """# Azure Leap Day Outage 2012
Root Cause: Certificate creation bug. Added +1 year to today's date (Feb 29) resulting in invalid Feb 29, 2013 expiration.
Impact: Global Azure services unavailable for ~24 hours as certificates were rejected.
Timeline:
- 00:00 UTC Feb 29: Outage begins as new certs are generated
- 01:00 UTC: Engineering alerted
- Restoration took most of the day across regions.
""")

if __name__ == "__main__":
    populate()
    print("\n✅ Successfully populated initial 6 incidents!")
