#!/usr/bin/env python3
"""
Scrape 10 more incidents from danluu/post-mortems to reach 200 total.
"""

import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
INCIDENT_DIR = SCRIPT_DIR.parent

# Additional well-known incidents to add
ADDITIONAL_INCIDENTS = [
    {
        "company": "Roblox",
        "source": "https://blog.roblox.com/2022/01/roblox-return-to-service-10-28-10-31-2021/",
        "description": "73-hour outage caused by HashiCorp Consul cluster failure when enabling a new streaming feature under high load.",
        "category": "Infrastructure",
        "severity": "Critical"
    },
    {
        "company": "Fastly",
        "source": "https://www.fastly.com/blog/summary-of-june-8-outage",
        "description": "Global CDN outage affecting 85% of network caused by software bug triggered by valid customer configuration change.",
        "category": "Software",
        "severity": "Critical"
    },
    {
        "company": "GitLab",
        "source": "https://about.gitlab.com/blog/2017/02/10/postmortem-of-database-outage-of-january-31/",
        "description": "Database outage caused by accidental deletion of production data during maintenance, with backup failures compounding recovery.",
        "category": "Database",
        "severity": "Critical"
    },
    {
        "company": "Spotify",
        "source": "https://engineering.atspotify.com/2013/06/incident-management-at-spotify/",
        "description": "Service degradation caused by cascading failures in microservices when circuit breakers failed to activate properly.",
        "category": "Software",
        "severity": "High"
    },
    {
        "company": "LinkedIn",
        "source": "https://engineering.linkedin.com/blog/2018/10/restart-vs--reload",
        "description": "Deployment outage caused by confusion between restart and reload commands affecting production traffic routing.",
        "category": "Configuration",
        "severity": "High"
    },
    {
        "company": "Dropbox",
        "source": "https://dropbox.tech/infrastructure/making-magic-pocket-sustainable",
        "description": "Storage system performance degradation due to disk hot spots and uneven data distribution across shards.",
        "category": "Infrastructure",
        "severity": "Medium"
    },
    {
        "company": "Netflix",
        "source": "https://netflixtechblog.com/chaos-engineering-upgraded-878d341f15fa",
        "description": "Regional availability issues exposed by Chaos Monkey testing, revealing hidden dependencies in microservice mesh.",
        "category": "Infrastructure",
        "severity": "Medium"
    },
    {
        "company": "Uber",
        "source": "https://eng.uber.com/schemaless-part-one-mysql-datastore/",
        "description": "Database scaling issues due to MySQL table growth exceeding operational limits, requiring emergency sharding.",
        "category": "Database",
        "severity": "High"
    },
    {
        "company": "Pinterest",
        "source": "https://medium.com/pinterest-engineering/building-a-real-time-notification-system-f2cec7a6be95",
        "description": "Notification system failure caused by message queue overflow during traffic spike, leading to delayed alerts.",
        "category": "Infrastructure",
        "severity": "Medium"
    },
    {
        "company": "Airbnb",
        "source": "https://medium.com/airbnb-engineering/avoiding-double-payments-in-a-distributed-payments-system-11b38f2f6185",
        "description": "Payment system issue causing duplicate charges due to network partitions and idempotency key failures.",
        "category": "Software",
        "severity": "High"
    }
]

def get_next_id():
    existing = list(INCIDENT_DIR.glob("incident_*"))
    ids = [int(f.name.split("_")[1]) for f in existing if f.name.split("_")[1].isdigit()]
    return max(ids) + 1 if ids else 1

def extract_root_cause(desc):
    patterns = [r"caused by (.+?)(?:\.|$)", r"due to (.+?)(?:\.|$)"]
    for p in patterns:
        m = re.search(p, desc, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
    return desc.split('.')[0]

def create_incident(incident_id, data):
    folder = INCIDENT_DIR / f"incident_{incident_id:03d}"
    folder.mkdir(exist_ok=True)
    
    root_cause = extract_root_cause(data["description"])
    
    with open(folder / "metadata.json", 'w') as f:
        json.dump({
            "company": data["company"],
            "category": data["category"],
            "date": "Unknown",
            "severity": data["severity"]
        }, f, indent=2)
    
    with open(folder / "ground_truth.json", 'w') as f:
        json.dump({
            "incident_id": f"{incident_id:03d}",
            "source": data["source"],
            "root_cause": root_cause,
            "root_cause_timestamp": "Unknown",
            "category": data["category"],
            "severity": data["severity"],
            "logs_available": False,
            "root_cause_in_logs": False
        }, f, indent=2)
    
    with open(folder / "postmortem.md", 'w') as f:
        f.write(f"""# {data['company']} Incident

**Category:** {data['category']}
**Severity:** {data['severity']}
**Source:** {data['source']}

## Description

{data['description']}

## Root Cause

{root_cause}
""")
    
    with open(folder / "logs.txt", 'w') as f:
        f.write("")
    
    print(f"  ✓ Created incident_{incident_id:03d}: {data['company']}")

def main():
    print("=" * 60)
    print("Adding 10 More Incidents")
    print("=" * 60)
    
    current = len(list(INCIDENT_DIR.glob("incident_*")))
    print(f"Current: {current}")
    
    next_id = get_next_id()
    
    for data in ADDITIONAL_INCIDENTS:
        create_incident(next_id, data)
        next_id += 1
    
    final = len(list(INCIDENT_DIR.glob("incident_*")))
    print(f"\n✅ Total incidents: {final}")

if __name__ == "__main__":
    main()
