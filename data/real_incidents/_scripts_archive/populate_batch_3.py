from collect_incidents import IncidentDataCollector
import json
import os

def populate_batch_3():
    collector = IncidentDataCollector()
    
    # 13. Gitlab - Database Deletion (2017)
    print("Populating Gitlab 2017...")
    collector.create_incident_directory("013")
    collector.save_metadata("013", {
        "company": "Gitlab",
        "category": "Database/Human Error",
        "date": "2017-01-31",
        "severity": "Critical"
    })
    collector.save_ground_truth("013", {
        "incident_id": "013",
        "source": "https://about.gitlab.com/2017/02/10/postmortem-of-database-outage-of-january-31/",
        "root_cause": "System administrator inadvertently deleted the production database directory during a replication lag crisis.",
        "root_cause_timestamp": "2017-01-31T23:00:00Z",
        "category": "Database",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("013", """# Gitlab Outage Jan 31, 2017
Root Cause: Database directory deletion on primary.
Context: High load caused replication lag. Admin tried to wipe staging but ran `rm -rf` on the wrong terminal (primary).
Outcome: 6 hours of data lost across issues, merge requests, users.
""")

    # 14. CrowdStrike - Windows BSOD (2024)
    print("Populating CrowdStrike 2024...")
    collector.create_incident_directory("014")
    collector.save_metadata("014", {
        "company": "CrowdStrike",
        "category": "Software/Update",
        "date": "2024-07-19",
        "severity": "Critical"
    })
    collector.save_ground_truth("014", {
        "incident_id": "014",
        "source": "https://www.crowdstrike.com/falcon-content-update-remediation-and-guidance-hub/",
        "root_cause": "A buggy content update in Falcon sensor triggered an out-of-bounds memory read in the kernel driver, causing Windows BSOD.",
        "root_cause_timestamp": "2024-07-19T04:09:00Z",
        "category": "Software",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("014", """# CrowdStrike Global Outage July 19, 2024
Root Cause: Out-of-bounds memory read in C-00000291*.sys driver during a rapid response content update.
Impact: 8.5 million Windows machines crashed, global transport and banking affected.
""")

    # 15. GitHub - Schema Migration Deadlock (2021)
    print("Populating GitHub 2021...")
    collector.create_incident_directory("015")
    collector.save_metadata("015", {
        "company": "GitHub",
        "category": "Database/Migration",
        "date": "2021-11-20",
        "severity": "High"
    })
    collector.save_ground_truth("015", {
        "incident_id": "015",
        "source": "https://github.blog/2021-12-01-github-availability-report-november-2021/",
        "root_cause": "MySQL read replicas entered a semaphore deadlock during the final 'rename' step of a massive schema migration.",
        "root_cause_timestamp": "2021-11-20T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("015", """# GitHub Outage Nov 2021
Root Cause: Semaphore deadlock during table rename in schema migration.
Impact: Replicas crashed, creating a cascading failure on healthy nodes due to increased load.
""")

    # 16. Etsy - ID Overflow (2012)
    print("Populating Etsy 2012...")
    collector.create_incident_directory("016")
    collector.save_metadata("016", {
        "company": "Etsy",
        "category": "Database/Software",
        "date": "2012-01-01",
        "severity": "High"
    })
    collector.save_ground_truth("016", {
        "incident_id": "016",
        "source": "https://blog.etsy.com/news/2012/demystifying-site-outages/",
        "root_cause": "Signed 32-bit integer overflow in database IDs caused operations to fail.",
        "root_cause_timestamp": "2012-01-01T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("016", """# Etsy Outage 2012
Root Cause: Max value of signed 32-bit INT reached for primary keys.
Outcome: Database stopped accepting new records, requiring migration to BigInt.
""")

    # 17. Sentry - Transaction Wraparound (2015)
    print("Populating Sentry 2015...")
    collector.create_incident_directory("017")
    collector.save_metadata("017", {
        "company": "Sentry",
        "category": "Database/Postgres",
        "date": "2015-07-23",
        "severity": "High"
    })
    collector.save_ground_truth("017", {
        "incident_id": "017",
        "source": "https://blog.sentry.io/2015/07/23/transaction-id-wraparound-in-postgres",
        "root_cause": "PostgreSQL transaction ID wraparound forced the database into read-only mode to prevent data loss.",
        "root_cause_timestamp": "2015-07-23T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("017", """# Sentry Outage July 2015
Root Cause: TXID wraparound in Postgres.
Impact: Autovacuum failed to keep up with high write volume, database stopped accepting writes.
""")

    # 18. Salesforce - Policy Change (2023)
    print("Populating Salesforce 2023...")
    collector.create_incident_directory("018")
    collector.save_metadata("018", {
        "company": "Salesforce",
        "category": "Security/Policy",
        "date": "2023-09-20",
        "severity": "Critical"
    })
    collector.save_ground_truth("018", {
        "incident_id": "018",
        "source": "https://help.salesforce.com/s/articleView?id=000396429&type=1",
        "root_cause": "A security policy change inadvertently blocked access to resources beyond its intended scope, preventing logins.",
        "root_cause_timestamp": "2023-09-20T14:48:00Z",
        "category": "Security",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("018", """# Salesforce Outage Sept 20, 2023
Root Cause: Misconfigured security policy change.
Impact: Global login disruption for a subset of customers.
""")

    # 19. Slack - Network Saturation (2021)
    print("Populating Slack 2021...")
    collector.create_incident_directory("019")
    collector.save_metadata("019", {
        "company": "Slack",
        "category": "Network/AWS",
        "date": "2021-01-04",
        "severity": "High"
    })
    collector.save_ground_truth("019", {
        "incident_id": "019",
        "source": "https://slack.engineering/slacks-outage-on-january-4th-2021/",
        "root_cause": "Network saturation in AWS transit gateways caused massive packet loss and connection drops.",
        "root_cause_timestamp": "2021-01-04T15:00:00Z",
        "category": "Network",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("019", """# Slack Outage Jan 4, 2021
Root Cause: AWS Transit Gateway saturation causing packet loss.
Impact: 5% of users globally disconnected, unable to reconnect.
""")

    # 20. Google Service Control - Null Pointer (2022)
    print("Populating Google 2022...")
    collector.create_incident_directory("020")
    collector.save_metadata("020", {
        "company": "Google",
        "category": "Software/Null Pointer",
        "date": "2022-12-15",
        "severity": "Critical"
    })
    collector.save_ground_truth("020", {
        "incident_id": "020",
        "source": "https://status.cloud.google.com/incidents/ow5i3PPK96RduMcb1SsW",
        "root_cause": "A policy change with blank fields triggered a null pointer exception in Service Control, causing global crash loops.",
        "root_cause_timestamp": "2022-12-15T00:00:00Z",
        "category": "Software",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("020", """# Google Outage Dec 15, 2022
Root Cause: Null pointer exception during policy replication.
Trigger: Blank fields in a policy change were not handled, crashing the Service Control binary globally.
""")

if __name__ == "__main__":
    populate_batch_3()
    print("\n✅ Successfully populated Batch 3!")
