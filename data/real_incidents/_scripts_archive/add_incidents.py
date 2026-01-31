#!/usr/bin/env python3
"""
===============================================================================
Manual Incident Curation Helper
===============================================================================

Use this script to manually add new incidents to the dataset, 
just like the original populate_batch_*.py scripts.

USAGE:
    python add_incidents.py

This will interactively walk you through adding incidents.
You can also edit this file to batch-add multiple incidents at once.
"""

import json
from pathlib import Path

# Setup
SCRIPT_DIR = Path(__file__).parent.absolute()
INCIDENT_DIR = SCRIPT_DIR.parent

def get_next_id() -> int:
    """Get the next available incident ID."""
    existing = list(INCIDENT_DIR.glob("incident_*"))
    if not existing:
        return 1
    ids = [int(f.name.split("_")[1]) for f in existing]
    return max(ids) + 1

def create_incident(
    incident_id: int,
    company: str,
    category: str,
    date: str,
    severity: str,
    source_url: str,
    root_cause: str,
    postmortem_summary: str,
    logs: str = ""
):
    """Create a new incident with all required files."""
    
    folder = INCIDENT_DIR / f"incident_{incident_id:03d}"
    folder.mkdir(exist_ok=True)
    
    # metadata.json
    metadata = {
        "company": company,
        "category": category,
        "date": date,
        "severity": severity
    }
    with open(folder / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ground_truth.json
    ground_truth = {
        "incident_id": f"{incident_id:03d}",
        "source": source_url,
        "root_cause": root_cause,
        "root_cause_timestamp": f"{date}T00:00:00Z",
        "category": category,
        "severity": severity,
        "logs_available": bool(logs),
        "root_cause_in_logs": bool(logs)
    }
    with open(folder / "ground_truth.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    # postmortem.md
    postmortem = f"""# {company} Incident - {date}

**Category:** {category}
**Severity:** {severity}
**Source:** {source_url}

## Summary

{postmortem_summary}

## Root Cause

{root_cause}
"""
    with open(folder / "postmortem.md", 'w') as f:
        f.write(postmortem)
    
    # logs.txt
    with open(folder / "logs.txt", 'w') as f:
        f.write(logs)
    
    print(f"✅ Created incident_{incident_id:03d}: {company} - {category}")
    return folder


def interactive_add():
    """Interactively add a new incident."""
    next_id = get_next_id()
    print(f"\n=== Adding Incident {next_id:03d} ===\n")
    
    company = input("Company name: ").strip()
    category = input("Category (Database/Infrastructure/Security/Memory/Network/Software/Kubernetes/Cloud): ").strip()
    date = input("Date (YYYY-MM-DD): ").strip()
    severity = input("Severity (Critical/High/Medium/Low): ").strip()
    source_url = input("Source URL: ").strip()
    
    print("\nEnter the root cause (one sentence):")
    root_cause = input("> ").strip()
    
    print("\nEnter postmortem summary (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    postmortem_summary = "\n".join(lines)
    
    create_incident(
        next_id, company, category, date, severity,
        source_url, root_cause, postmortem_summary
    )


def batch_add_examples():
    """
    TEMPLATE: Copy and modify this function to batch-add incidents.
    Uncomment and edit the examples below.
    """
    next_id = get_next_id()
    
    # Example incident - uncomment and modify:
    # create_incident(
    #     incident_id=next_id,
    #     company="Stripe",
    #     category="Database",
    #     date="2023-03-15",
    #     severity="Critical",
    #     source_url="https://stripe.com/blog/outage-postmortem",
    #     root_cause="Database connection pool exhaustion due to connection leak in payment processing service.",
    #     postmortem_summary="""
    #     At 14:30 UTC, our payment processing service began experiencing timeouts.
    #     Investigation revealed a connection leak introduced in a recent deployment.
    #     The leak caused gradual exhaustion of the database connection pool.
    #     Resolution: Rolled back to previous version and patched the leak.
    #     """
    # )
    
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("Manual Incident Curation Helper")
    print("=" * 60)
    
    current_count = len(list(INCIDENT_DIR.glob("incident_*")))
    print(f"\nCurrent incidents: {current_count}")
    print(f"Next ID: {get_next_id():03d}")
    
    print("\nOptions:")
    print("  1. Add incident interactively")
    print("  2. Edit this file for batch additions")
    print("  3. Exit")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        while True:
            interactive_add()
            again = input("\nAdd another? (y/n): ").strip().lower()
            if again != 'y':
                break
    elif choice == "2":
        print("\nEdit the batch_add_examples() function in this file.")
        print("Then run: python add_incidents.py --batch")
    else:
        print("Bye!")
