"""
Real Incident Data Collection Script

This script helps organize real incident data from public post-mortems
into a standardized format for evaluation.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class IncidentDataCollector:
    """Manages collection and organization of real incident data."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_incident_directory(self, incident_id: str) -> Path:
        """Create directory structure for a new incident."""
        incident_dir = self.base_dir / f"incident_{incident_id}"
        incident_dir.mkdir(exist_ok=True)
        return incident_dir
    
    def save_logs(self, incident_id: str, logs: str):
        """Save log entries for an incident."""
        incident_dir = self.create_incident_directory(incident_id)
        log_file = incident_dir / "logs.txt"
        log_file.write_text(logs)
        print(f"✓ Saved logs to {log_file}")
    
    def save_postmortem(self, incident_id: str, postmortem: str):
        """Save original post-mortem document."""
        incident_dir = self.create_incident_directory(incident_id)
        pm_file = incident_dir / "postmortem.md"
        pm_file.write_text(postmortem)
        print(f"✓ Saved post-mortem to {pm_file}")
    
    def save_metadata(self, incident_id: str, metadata: Dict):
        """Save incident metadata."""
        incident_dir = self.create_incident_directory(incident_id)
        meta_file = incident_dir / "metadata.json"
        
        # Ensure required fields
        required_fields = ["category", "date", "severity"]
        for field in required_fields:
            if field not in metadata:
                metadata[field] = "Unknown"
        
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {meta_file}")
    
    def save_ground_truth(self, incident_id: str, ground_truth: Dict):
        """Save ground truth annotation."""
        incident_dir = self.create_incident_directory(incident_id)
        gt_file = incident_dir / "ground_truth.json"
        
        # Validate ground truth schema
        required_fields = [
            "incident_id", "source", "root_cause", 
            "category", "severity", "logs_available",
            "root_cause_in_logs"
        ]
        
        for field in required_fields:
            if field not in ground_truth:
                raise ValueError(f"Missing required field: {field}")
        
        # Ensure incident_id matches
        ground_truth["incident_id"] = incident_id
        
        with open(gt_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        print(f"✓ Saved ground truth to {gt_file}")
    
    def create_incident(
        self,
        incident_id: str,
        logs: Optional[str] = None,
        postmortem: Optional[str] = None,
        metadata: Optional[Dict] = None,
        ground_truth: Optional[Dict] = None
    ):
        """Create a complete incident entry."""
        print(f"\n📁 Creating incident_{incident_id}...")
        
        if logs:
            self.save_logs(incident_id, logs)
        
        if postmortem:
            self.save_postmortem(incident_id, postmortem)
        
        if metadata:
            self.save_metadata(incident_id, metadata)
        
        if ground_truth:
            self.save_ground_truth(incident_id, ground_truth)
        
        print(f"✅ Incident {incident_id} created successfully\n")
    
    def list_incidents(self) -> List[str]:
        """List all collected incidents."""
        incidents = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith("incident_"):
                incident_id = item.name.replace("incident_", "")
                incidents.append(incident_id)
        return sorted(incidents)
    
    def get_incident_info(self, incident_id: str) -> Dict:
        """Get information about a specific incident."""
        incident_dir = self.base_dir / f"incident_{incident_id}"
        
        if not incident_dir.exists():
            raise ValueError(f"Incident {incident_id} not found")
        
        info = {
            "incident_id": incident_id,
            "has_logs": (incident_dir / "logs.txt").exists(),
            "has_postmortem": (incident_dir / "postmortem.md").exists(),
            "has_metadata": (incident_dir / "metadata.json").exists(),
            "has_ground_truth": (incident_dir / "ground_truth.json").exists(),
        }
        
        # Load metadata if available
        if info["has_metadata"]:
            with open(incident_dir / "metadata.json") as f:
                info["metadata"] = json.load(f)
        
        # Load ground truth if available
        if info["has_ground_truth"]:
            with open(incident_dir / "ground_truth.json") as f:
                info["ground_truth"] = json.load(f)
        
        return info


def create_ground_truth_template(incident_id: str) -> Dict:
    """Create a template for ground truth annotation."""
    return {
        "incident_id": incident_id,
        "source": "",  # e.g., "GitHub Post-Mortem #123"
        "root_cause": "",  # Detailed description
        "root_cause_timestamp": "",  # ISO format: "2024-01-15T10:23:45Z"
        "category": "",  # Database, Network, Application, Security, Infrastructure, Monitoring
        "severity": "",  # Critical, High, Medium, Low
        "logs_available": False,
        "root_cause_in_logs": False,
        "documentation_available": False,
        "notes": ""  # Additional context
    }


def main():
    """Example usage of the data collector."""
    collector = IncidentDataCollector()
    
    # Example: Create a sample incident
    sample_ground_truth = {
        "incident_id": "001",
        "source": "GitHub Post-Mortem #123",
        "root_cause": "Database connection pool exhaustion due to missing timeout configuration",
        "root_cause_timestamp": "2024-01-15T10:23:45Z",
        "category": "Database",
        "severity": "Critical",
        "logs_available": True,
        "root_cause_in_logs": True,
        "documentation_available": False,
        "notes": "Connection pool reached max size, no timeout caused indefinite waits"
    }
    
    sample_metadata = {
        "category": "Database",
        "date": "2024-01-15",
        "severity": "Critical",
        "duration_minutes": 45,
        "services_affected": ["api-server", "web-frontend"]
    }
    
    sample_logs = """2024-01-15T10:23:45Z [ERROR] Connection pool exhausted
2024-01-15T10:23:46Z [WARN] Database query timeout
2024-01-15T10:23:47Z [ERROR] API request failed: timeout
2024-01-15T10:24:00Z [CRITICAL] Service degradation detected"""
    
    sample_postmortem = """# Database Connection Pool Exhaustion

## Summary
Service outage caused by database connection pool exhaustion.

## Root Cause
Missing timeout configuration on connection pool allowed connections
to be held indefinitely, eventually exhausting the pool.

## Timeline
- 10:23:45 - Connection pool reaches maximum size
- 10:23:46 - First timeout errors appear
- 10:24:00 - Service degradation alert triggered
- 10:30:00 - Issue identified and mitigated
"""
    
    # Create the incident
    collector.create_incident(
        incident_id="001",
        logs=sample_logs,
        postmortem=sample_postmortem,
        metadata=sample_metadata,
        ground_truth=sample_ground_truth
    )
    
    # List all incidents
    print("📋 Collected incidents:")
    for inc_id in collector.list_incidents():
        info = collector.get_incident_info(inc_id)
        print(f"  - incident_{inc_id}: {info.get('metadata', {}).get('category', 'Unknown')}")


if __name__ == "__main__":
    main()
