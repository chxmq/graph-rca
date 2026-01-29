from collect_incidents import IncidentDataCollector
import json
import os

def populate_batch_2():
    collector = IncidentDataCollector()
    
    # 7. Joyent Manta - DB Lock (2015)
    print("Populating Joyent 2015...")
    collector.create_incident_directory("007")
    collector.save_metadata("007", {
        "company": "Joyent",
        "category": "Database/Locking",
        "date": "2015-07-27",
        "severity": "High"
    })
    collector.save_ground_truth("007", {
        "incident_id": "007",
        "source": "https://www.joyent.com/blog/manta-postmortem-7-27-2015",
        "root_cause": "Locking conflict between PostgreSQL transaction wraparound maintenance (autovacuum) and a global lock query that blocked all operations.",
        "root_cause_timestamp": "2015-07-27T00:00:00Z",
        "category": "Database",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("007", """# Joyent Manta Postmortem July 27, 2015
Root Cause: Blocked locks on metadata servers.
Conflict: PG autovacuum held a lock, and a separate query tried to take a global lock, blocking subsequent requests.
""")

    # 8. Stack Exchange - StackEgg Load (2015)
    print("Populating Stack Exchange 2015...")
    collector.create_incident_directory("008")
    collector.save_metadata("008", {
        "company": "Stack Exchange",
        "category": "Network/Load Balancing",
        "date": "2015-03-31",
        "severity": "High"
    })
    collector.save_ground_truth("008", {
        "incident_id": "008",
        "source": "https://stackstatus.net/post/115305251014/outage-postmortem-march-31-2015",
        "root_cause": "Enabling the 'StackEgg' feature globally caused a massive traffic spike to load balancers, effectively behaving like a self-inflicted DDoS.",
        "root_cause_timestamp": "2015-03-31T00:00:00Z",
        "category": "Network",
        "severity": "High",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("008", """# Stack Exchange Outage March 31, 2015
Root Cause: Self-inflicted DDoS via feature flag.
Enabling StackEgg for all users simultaneously overwhelmed HAProxy fleet.
""")

    # 9. Facebook - Global BGP Outage (2021)
    print("Populating Facebook 2021...")
    collector.create_incident_directory("009")
    collector.save_metadata("009", {
        "company": "Facebook",
        "category": "Network/BGP",
        "date": "2021-10-04",
        "severity": "Critical"
    })
    collector.save_ground_truth("009", {
        "incident_id": "009",
        "source": "https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/",
        "root_cause": "A routine maintenance audit command inadvertently disconnected all Facebook data centers from the Internet by withdrawing all BGP routes.",
        "root_cause_timestamp": "2021-10-04T15:39:00Z",
        "category": "Network",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("009", """# Facebook Outage Oct 4, 2021
Root Cause: BGP route withdrawal.
A command meant to audit backbone capacity configuration triggered a bug that disconnected all datacenters.
Timeline:
- 15:39 UTC: Routes withdrawn
- Total global outage of Facebook, Instagram, WhatsApp
- Fixed by physical access to routers in Santa Clara data center.
""")

    # 10. Amazon S3 - US-EAST-1 Outage (2017)
    print("Populating Amazon S3 2017...")
    collector.create_incident_directory("010")
    collector.save_metadata("010", {
        "company": "Amazon",
        "category": "Infrastructure/Human Error",
        "date": "2017-02-28",
        "severity": "Critical"
    })
    collector.save_ground_truth("010", {
        "incident_id": "010",
        "source": "https://aws.amazon.com/message/41926/",
        "root_cause": "Human error: A typo in a command intended to remove a few billing servers inadvertently removed a large set of critical S3 storage systems.",
        "root_cause_timestamp": "2017-02-28T09:37:00Z",
        "category": "Infrastructure",
        "severity": "Critical",
        "logs_available": False,
        "root_cause_in_logs": False
    })
    collector.save_postmortem("010", """# AWS S3 Outage Feb 28, 2017
Root Cause: Human error (Typo).
Command to remove servers was entered incorrectly, removing too many servers in US-EAST-1.
Impact: Massive cascading failure affecting S3, EC2, EBS, and hundreds of downstream customers.
""")

    # 11. Homebrew - Token Leak (2018)
    print("Populating Homebrew 2018...")
    collector.create_incident_directory("011")
    collector.save_metadata("011", {
        "company": "Homebrew",
        "category": "Security/Leaked Token",
        "date": "2018-08-05",
        "severity": "Medium"
    })
    collector.save_ground_truth("011", {
        "incident_id": "011",
        "source": "https://brew.sh/2018/08/05/security-incident-disclosure/",
        "root_cause": "A GitHub personal access token with write scopes was leaked in Jenkins build logs, allowing an attacker to push malicious commits.",
        "root_cause_timestamp": "2018-08-05T00:00:00Z",
        "category": "Security",
        "severity": "Medium",
        "logs_available": True,
        "root_cause_in_logs": True
    })
    collector.save_postmortem("011", """# Homebrew Security Incident Aug 2018
Root Cause: Token leak in Jenkins.
An attacker found an API token in public build logs and used it to push to homebrew repositories.
""")

    # 12. NPM - Fastly Backend Reset (2014)
    print("Populating NPM 2014...")
    collector.create_incident_directory("012")
    collector.save_metadata("012", {
        "company": "NPM",
        "category": "Configuration/Varnish",
        "date": "2014-01-28",
        "severity": "High"
    })
    collector.save_ground_truth("012", {
        "incident_id": "012",
        "source": "https://blog.npmjs.org/post/74949623024/2014-01-28-outage-postmortem.html",
        "root_cause": "Fastly/Varnish 'restart' command reset the request backend to the first one in the VCL list (Manta) instead of the intended pool.",
        "root_cause_timestamp": "2014-01-28T00:00:00Z",
        "category": "Configuration",
        "severity": "High",
        "logs_available": True,
        "root_cause_in_logs": True
    })
    collector.save_postmortem("012", """# NPM Outage Jan 28, 2014
Root Cause: Varnish restart behavior.
The `restart` command in VCL reset `req.backend` to the top-most defined backend.
""")

if __name__ == "__main__":
    populate_batch_2()
    print("\n✅ Successfully populated Batch 2!")
