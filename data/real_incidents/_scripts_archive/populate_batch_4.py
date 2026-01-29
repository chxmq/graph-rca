from collect_incidents import IncidentDataCollector
import json
import os

def populate_batch_4():
    collector = IncidentDataCollector()
    
    # 21. CircleCI - Malware/Session Theft (2023)
    print("Populating CircleCI 2023...")
    collector.create_incident_directory("021")
    collector.save_metadata("021", {
        "company": "CircleCI",
        "category": "Security/Malware",
        "date": "2023-01-04",
        "severity": "Critical"
    })
    collector.save_ground_truth("021", {
        "incident_id": "021",
        "source": "https://circleci.com/blog/jan-4-2023-incident-report/",
        "root_cause": "Malware on an engineer's laptop stole a valid, 2FA-backed SSO session cookie, allowing unauthorized access to production systems.",
        "root_cause_timestamp": "2022-12-16T00:00:00Z",
        "category": "Security",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 22. Discord - Redis Failover (2020)
    print("Populating Discord 2020...")
    collector.create_incident_directory("022")
    collector.save_metadata("022", {
        "company": "Discord",
        "category": "Infrastructure/Redis",
        "date": "2020-01-01",
        "severity": "High"
    })
    collector.save_ground_truth("022", {
        "incident_id": "022",
        "source": "https://status.discordapp.com/incidents/qk9cdgnqnhcn",
        "root_cause": "Automated GCP migration of a Redis primary caused it to drop offline, triggering cascading failures in how Discord handles Redis failover.",
        "root_cause_timestamp": "2020-01-01T14:01:00Z",
        "category": "Infrastructure",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 23. Fastly - Undiscovered Software Bug (2021)
    print("Populating Fastly 2021...")
    collector.create_incident_directory("023")
    collector.save_metadata("023", {
        "company": "Fastly",
        "category": "Software/CDN",
        "date": "2021-06-08",
        "severity": "Critical"
    })
    collector.save_ground_truth("023", {
        "incident_id": "023",
        "source": "https://www.fastly.com/blog/summary-of-june-8-outage",
        "root_cause": "An undiscovered software bug in Fastly's edge cloud was triggered by a valid customer configuration change, causing a global outage.",
        "root_cause_timestamp": "2021-06-08T00:00:00Z",
        "category": "Software",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 24. Foursquare - MongoDB Memory Exhaustion (2010)
    print("Populating Foursquare 2010...")
    collector.create_incident_directory("024")
    collector.save_metadata("024", {
        "company": "Foursquare",
        "category": "Database/Memory",
        "date": "2010-10-04",
        "severity": "High"
    })
    collector.save_ground_truth("024", {
        "incident_id": "024",
        "source": "https://web.archive.org/web/20230602082218/https://news.ycombinator.com/item?id=1769761",
        "root_cause": "MongoDB ran out of memory due to a query pattern with low locality (fetching full history for every check-in), leading to catastrophic failure.",
        "root_cause_timestamp": "2010-10-04T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 25. Flowdock - COVID Traffic Spike (2020)
    print("Populating Flowdock 2020...")
    collector.create_incident_directory("025")
    collector.save_metadata("025", {
        "company": "Flowdock",
        "category": "Infrastructure/Traffic",
        "date": "2020-03-16",
        "severity": "Medium"
    })
    collector.save_ground_truth("025", {
        "incident_id": "025",
        "source": "https://www.flowdock.com/blog/2020/03/17/incident-report-2020-03-16/",
        "root_cause": "Unexpected traffic spike due to remote work shift (COVID-19) overloaded backend components.",
        "root_cause_timestamp": "2020-03-16T15:00:00Z",
        "category": "Infrastructure",
        "severity": "Medium",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 26. Heroku - Token Leak (2022)
    print("Populating Heroku 2022...")
    collector.create_incident_directory("026")
    collector.save_metadata("026", {
        "company": "Heroku",
        "category": "Security/Leaked Token",
        "date": "2022-04-01",
        "severity": "Critical"
    })
    collector.save_ground_truth("026", {
        "incident_id": "026",
        "source": "https://blog.heroku.com/april-2022-incident-review",
        "root_cause": "Leaked private tokens allowed attackers to access internal databases and private repositories.",
        "root_cause_timestamp": "2022-04-01T00:00:00Z",
        "category": "Security",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 27. Basecamp - DDoS Attack (2014)
    print("Populating Basecamp 2014...")
    collector.create_incident_directory("027")
    collector.save_metadata("027", {
        "company": "Basecamp",
        "category": "Network/DDoS",
        "date": "2014-03-24",
        "severity": "High"
    })
    collector.save_ground_truth("027", {
        "incident_id": "027",
        "source": "https://signalvnoise.com/posts/3729-basecamp-network-attack-postmortem",
        "root_cause": "Sustained DDoS attack on Basecamp's network infrastructure during a 100-minute window.",
        "root_cause_timestamp": "2014-03-24T00:00:00Z",
        "category": "Network",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 28. Bitly - Credential Exposure (2014)
    print("Populating Bitly 2014...")
    collector.create_incident_directory("028")
    collector.save_metadata("028", {
        "company": "Bitly",
        "category": "Security/Credentials",
        "date": "2014-05-08",
        "severity": "High"
    })
    collector.save_ground_truth("028", {
        "incident_id": "028",
        "source": "https://blog.bitly.com/post/85260908544/more-detail",
        "root_cause": "Hosted source code repository contained plaintext credentials granting access to Bitly's database backups.",
        "root_cause_timestamp": "2014-05-08T00:00:00Z",
        "category": "Security",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 29. Knight Capital - Software Bug ($460M Loss) (2012)
    print("Populating Knight Capital 2012...")
    collector.create_incident_directory("029")
    collector.save_metadata("029", {
        "company": "Knight Capital",
        "category": "Software/Deployment",
        "date": "2012-08-01",
        "severity": "Critical"
    })
    collector.save_ground_truth("029", {
        "incident_id": "029",
        "source": "https://web.archive.org/web/20120803152648/https://www.sec.gov/news/press/2012/2012-150.htm",
        "root_cause": "An outdated code path (Power Peg) was inadvertently re-enabled during a production deployment, causing millions of erroneous trades.",
        "root_cause_timestamp": "2012-08-01T09:30:00Z",
        "category": "Software",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })

    # 30. Keepthescore - Database Deletion (2020)
    print("Populating Keepthescore 2020...")
    collector.create_incident_directory("030")
    collector.save_metadata("030", {
        "company": "Keepthescore",
        "category": "Database/Human Error",
        "date": "2020-04-10",
        "severity": "High"
    })
    collector.save_ground_truth("030", {
        "incident_id": "030",
        "source": "https://keepthescore.co/blog/postmortem-how-i-deleted-the-production-database/",
        "root_cause": "Developing on production: running a script locally that was connected to the production database and executing a drop command.",
        "root_cause_timestamp": "2020-04-10T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })

if __name__ == "__main__":
    populate_batch_4()
    print("\n✅ Successfully populated Batch 4!")
