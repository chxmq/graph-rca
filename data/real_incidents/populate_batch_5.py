from collect_incidents import IncidentDataCollector
import json
import os

def populate_batch_5():
    collector = IncidentDataCollector()
    
    # 31. PagerDuty - Quorum Loss (2013)
    print("Populating PagerDuty 2013...")
    collector.create_incident_directory("031")
    collector.save_metadata("031", {
        "company": "PagerDuty",
        "category": "Network/Quorum",
        "date": "2013-04-13",
        "severity": "Critical"
    })
    collector.save_ground_truth("031", {
        "incident_id": "031",
        "source": "https://www.pagerduty.com/blog/outage-post-mortem-april-13-2013/",
        "root_cause": "Common peering point failure across two 'independent' cloud deployments caused high latency and prevented the application from establishing quorum.",
        "root_cause_timestamp": "2013-04-13T00:00:00Z",
        "category": "Network",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 32. Roblox - Consul/BoltDB (2021)
    print("Populating Roblox 2021...")
    collector.create_incident_directory("032")
    collector.save_metadata("032", {
        "company": "Roblox",
        "category": "Infrastructure/Consul",
        "date": "2021-10-28",
        "severity": "Critical"
    })
    collector.save_ground_truth("032", {
        "incident_id": "032",
        "source": "https://blog.roblox.com/2022/01/roblox-return-to-service-10-28-10-31-2021/",
        "root_cause": "Consul streaming feature enabled higher-than-expected load on BoltDB, leading to 73-hour global outage.",
        "root_cause_timestamp": "2021-10-28T00:00:00Z",
        "category": "Infrastructure",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 33. Spotify - Cascading failure (2013)
    print("Populating Spotify 2013...")
    collector.create_incident_directory("033")
    collector.save_metadata("033", {
        "company": "Spotify",
        "category": "Software/Cascading",
        "date": "2013-06-04",
        "severity": "High"
    })
    collector.save_ground_truth("033", {
        "incident_id": "033",
        "source": "https://labs.spotify.com/2013/06/04/incident-management-at-spotify/",
        "root_cause": "Lack of exponential backoff in internal microservices caused retry storms and cascading service degradation.",
        "root_cause_timestamp": "2013-06-04T00:00:00Z",
        "category": "Software",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 34. Slack - DB Capacity (2018)
    print("Populating Slack 2018...")
    collector.create_incident_directory("034")
    collector.save_metadata("034", {
        "company": "Slack",
        "category": "Database/Scaling",
        "date": "2018-06-27",
        "severity": "High"
    })
    collector.save_ground_truth("034", {
        "incident_id": "034",
        "source": "https://slackhq.com/this-was-not-normal-really",
        "root_cause": "Massive reconnection spike following a network glitch overwhelmed database capacity, leading to cascading connection failures.",
        "root_cause_timestamp": "2018-06-27T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 35. Stack Exchange - SQL Server Bugcheck (2017)
    print("Populating Stack Exchange 2017...")
    collector.create_incident_directory("035")
    collector.save_metadata("035", {
        "company": "Stack Exchange",
        "category": "Database/SQL Server",
        "date": "2017-01-24",
        "severity": "High"
    })
    collector.save_ground_truth("035", {
        "incident_id": "035",
        "source": "https://stackstatus.net/post/156407746074/outage-postmortem-january-24-2017",
        "root_cause": "Primary SQL Server triggered a kernel bugcheck/crash, forcing sites into read-only mode and eventual outage.",
        "root_cause_timestamp": "2017-01-24T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 36. AWS Kinesis - Thread Limit (2020)
    print("Populating AWS Kinesis 2020...")
    collector.create_incident_directory("036")
    collector.save_metadata("036", {
        "company": "Amazon",
        "category": "Infrastructure/OS Config",
        "date": "2020-11-25",
        "severity": "Critical"
    })
    collector.save_ground_truth("036", {
        "incident_id": "036",
        "source": "https://aws.amazon.com/message/11201/",
        "root_cause": "Scaling the Kinesis front-end fleet exceeded OS thread limits due to memory-intensive configuration, breaking Cognito/Lambda.",
        "root_cause_timestamp": "2020-11-25T05:15:00Z",
        "category": "Infrastructure",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 37. Twilio - Redis Billing (2013)
    print("Populating Twilio 2013...")
    collector.create_incident_directory("037")
    collector.save_metadata("037", {
        "company": "Twilio",
        "category": "Database/Redis",
        "date": "2013-07-19",
        "severity": "High"
    })
    collector.save_ground_truth("037", {
        "incident_id": "037",
        "source": "https://www.twilio.com/blog/2013/07/billing-incident-post-mortem-breakdown-analysis-and-root-cause.html",
        "root_cause": "Redis network partition caused resync storm, crashing the master and leaving billing in read-only mode, causing retry over-billing.",
        "root_cause_timestamp": "2013-07-19T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 38. Zerodha - Penny Stock Scaling (2019)
    print("Populating Zerodha 2019...")
    collector.create_incident_directory("038")
    collector.save_metadata("038", {
        "company": "Zerodha",
        "category": "Software/Scaling",
        "date": "2019-08-29",
        "severity": "Medium"
    })
    collector.save_ground_truth("038", {
        "incident_id": "038",
        "source": "https://zerodha.com/marketintel/bulletin/229363/post-mortem-of-technical-issue-august-29-2019",
        "root_cause": "Single order for 1M penny stock units generated 100k+ individual trades, overwhelming the Order Management System's limit.",
        "root_cause_timestamp": "2019-08-29T00:00:00Z",
        "category": "Software",
        "severity": "Medium",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 39. Cloudflare - Bot Management (2025)
    print("Populating Cloudflare 2025...")
    collector.create_incident_directory("039")
    collector.save_metadata("039", {
        "company": "Cloudflare",
        "category": "Software/Logic",
        "date": "2025-11-18",
        "severity": "Critical"
    })
    collector.save_ground_truth("039", {
        "incident_id": "039",
        "source": "https://blog.cloudflare.com/18-november-2025-outage/",
        "root_cause": "Bug in ClickHouse query logic used for Bot Management generated duplicate features, exceeding fixed-size configuration buffers.",
        "root_cause_timestamp": "2025-11-18T00:00:00Z",
        "category": "Software",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 40. Sentry - S3 Backup Leak (2016)
    print("Populating Sentry 2016...")
    collector.create_incident_directory("040")
    collector.save_metadata("040", {
        "company": "Sentry",
        "category": "Security/S3",
        "date": "2016-06-12",
        "severity": "High"
    })
    collector.save_ground_truth("040", {
        "incident_id": "040",
        "source": "https://blog.sentry.io/2016/06/14/security-incident-june-12-2016",
        "root_cause": "Misconfigured S3 bucket permissions for backups lead to public exposure of sensitive data.",
        "root_cause_timestamp": "2016-06-12T00:00:00Z",
        "category": "Security",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

if __name__ == "__main__":
    populate_batch_5()
    print("\n✅ Successfully populated Batch 5!")
