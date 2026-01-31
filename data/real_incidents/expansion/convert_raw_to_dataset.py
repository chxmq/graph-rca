#!/usr/bin/env python3
"""
===============================================================================
Convert Raw Incidents to Dataset Format
===============================================================================

Converts the 190 pre-scraped incidents from sources/raw/github/*.json
into proper incident_XXX folders with all required files.

Uses the description field to extract root cause (already human-readable).

RUN:
    cd data/real_incidents/expansion
    python convert_raw_to_dataset.py
"""

import json
import re
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
INCIDENT_DIR = SCRIPT_DIR.parent
RAW_DIR = INCIDENT_DIR / "sources" / "raw" / "github"

# Category mapping based on common keywords
CATEGORY_KEYWORDS = {
    "Database": ["database", "postgres", "mysql", "mongodb", "redis", "sql", "db", "cassandra"],
    "Network": ["network", "dns", "bgp", "routing", "connectivity", "latency", "packet"],
    "Infrastructure": ["infrastructure", "kubernetes", "k8s", "docker", "container", "cluster", "server"],
    "Security": ["security", "breach", "token", "credential", "leaked", "auth", "ssl", "certificate"],
    "Memory": ["memory", "oom", "heap", "gc", "garbage collection", "ram"],
    "Configuration": ["config", "configuration", "setting", "misconfiguration", "typo"],
    "Software": ["bug", "code", "software", "regression", "deploy", "release"],
    "Cloud": ["aws", "azure", "gcp", "cloud", "s3", "ec2"],
}

def detect_category(text: str) -> str:
    """Detect category from description text."""
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category
    return "Infrastructure"  # Default

def detect_severity(text: str) -> str:
    """Detect severity from description text."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["global", "complete", "total", "major", "critical"]):
        return "Critical"
    if any(w in text_lower for w in ["significant", "widespread", "severe"]):
        return "High"
    if any(w in text_lower for w in ["partial", "degraded", "intermittent"]):
        return "Medium"
    return "Medium"

def extract_root_cause(description: str) -> str:
    """Extract root cause from description - often already stated."""
    # If description mentions root cause patterns, extract that
    desc = description.strip()
    
    # Common patterns
    patterns = [
        r"caused by (.+?)(?:\.|$)",
        r"due to (.+?)(?:\.|$)",
        r"root cause[:\s]+(.+?)(?:\.|$)",
        r"issue was (.+?)(?:\.|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, desc, re.IGNORECASE)
        if match:
            return match.group(1).strip().capitalize()
    
    # Otherwise use first sentence as root cause
    first_sentence = desc.split('.')[0].strip()
    return first_sentence if len(first_sentence) < 200 else first_sentence[:200] + "..."

def get_next_incident_id() -> int:
    """Get next available incident ID."""
    existing = list(INCIDENT_DIR.glob("incident_*"))
    if not existing:
        return 1
    ids = [int(f.name.split("_")[1]) for f in existing if f.name.split("_")[1].isdigit()]
    return max(ids) + 1 if ids else 1

def convert_raw_incident(raw_file: Path, incident_id: int) -> bool:
    """Convert a single raw incident JSON to proper incident folder."""
    try:
        with open(raw_file) as f:
            data = json.load(f)
        
        company = data.get("company", "Unknown")
        description = data.get("description", "")
        source_url = data.get("url", "")
        
        if not description or len(description) < 20:
            return False
        
        # Skip if company looks like a category header
        if company in ["Config Errors", "Hardware/Power Failures", "Conflicts", "Time", "Database", "Uncategorized"]:
            return False
        
        # Extract/detect fields
        category = detect_category(description)
        severity = detect_severity(description)
        root_cause = extract_root_cause(description)
        
        # Create incident folder
        folder = INCIDENT_DIR / f"incident_{incident_id:03d}"
        folder.mkdir(exist_ok=True)
        
        # metadata.json
        metadata = {
            "company": company,
            "category": category,
            "date": "Unknown",
            "severity": severity
        }
        with open(folder / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # ground_truth.json
        ground_truth = {
            "incident_id": f"{incident_id:03d}",
            "source": source_url,
            "root_cause": root_cause,
            "root_cause_timestamp": "Unknown",
            "category": category,
            "severity": severity,
            "logs_available": False,
            "root_cause_in_logs": False
        }
        with open(folder / "ground_truth.json", 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        # postmortem.md
        postmortem = f"""# {company} Incident

**Category:** {category}
**Severity:** {severity}
**Source:** {source_url}

## Description

{description}

## Root Cause

{root_cause}
"""
        with open(folder / "postmortem.md", 'w') as f:
            f.write(postmortem)
        
        # logs.txt (empty for now)
        with open(folder / "logs.txt", 'w') as f:
            f.write("")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {raw_file}: {e}")
        return False

def main():
    print("=" * 60)
    print("Converting Raw Incidents to Dataset Format")
    print("=" * 60)
    
    # Get current count
    current_count = len(list(INCIDENT_DIR.glob("incident_*")))
    print(f"Current incidents: {current_count}")
    
    # Get raw files
    raw_files = sorted(RAW_DIR.glob("incident_*.json"))
    print(f"Raw files available: {len(raw_files)}")
    
    # Target: 200 total
    target = 200
    need = max(0, target - current_count)
    print(f"Target: {target}, Need to convert: {need}")
    
    if need == 0:
        print("Already at target!")
        return
    
    next_id = get_next_incident_id()
    converted = 0
    
    for raw_file in raw_files:
        if converted >= need:
            break
        
        # Skip files already used (roughly match by number)
        raw_num = int(raw_file.stem.split("_")[1])
        if raw_num <= 60:
            continue
        
        print(f"  Converting {raw_file.name} -> incident_{next_id:03d}")
        
        if convert_raw_incident(raw_file, next_id):
            converted += 1
            next_id += 1
    
    final_count = len(list(INCIDENT_DIR.glob("incident_*")))
    print(f"\n✅ Converted {converted} incidents")
    print(f"Total incidents: {final_count}")

if __name__ == "__main__":
    main()
