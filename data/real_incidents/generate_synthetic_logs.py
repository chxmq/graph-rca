import os
import json
import ollama
from pathlib import Path

def generate_logs_for_incident(incident_id: str):
    incident_dir = Path(f"incident_{incident_id}")
    metadata_path = incident_dir / "metadata.json"
    postmortem_path = incident_dir / "postmortem.md"
    ground_truth_path = incident_dir / "ground_truth.json"
    logs_path = incident_dir / "logs.txt"

    if logs_path.exists() and logs_path.stat().st_size > 0:
        print(f"  Incident {incident_id}: logs.txt already exists. Skipping.")
        return

    print(f"  Incident {incident_id}: Generating logs...")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    with open(postmortem_path, 'r') as f:
        postmortem = f.read()
    with open(ground_truth_path, 'r') as f:
        gt = json.load(f)

    prompt = f"""Generate a semi-realistic log snippet (15-20 lines) for the following IT incident.
The logs should include timestamps, service names, log levels (INFO, WARNING, ERROR, CRITICAL), and messages that reflect the progression of the incident leading to the root cause.

Metadata: {json.dumps(metadata)}
Root Cause: {gt['root_cause']}
Post-mortem Summary:
{postmortem}

Format the output as a plain text log file. Do not include any explanation or markdown formatting. Just the log lines.
Example output:
2024-01-01 10:00:00 [INFO] service-a: Starting request handling
2024-01-01 10:00:05 [ERROR] service-b: Database connection timeout
"""

    try:
        # Using a higher timeout for generation
        client = ollama.Client(host="http://localhost:11434")
        response = client.generate(
            model="llama3.2:3b",
            prompt=prompt,
            options={"temperature": 0.7}
        )
        
        log_content = response['response'].strip()
        
        # Clean up any accidental markdown code blocks
        if log_content.startswith("```"):
            log_content = "\n".join(log_content.split("\n")[1:-1])

        with open(logs_path, 'w') as f:
            f.write(log_content)
        print(f"  Incident {incident_id}: logs.txt saved.")
    except Exception as e:
        print(f"  Incident {incident_id}: Failed to generate logs: {e}")

def main():
    # Use relative path to the directory where incidents are located
    # This script is located in data/real_incidents/
    base_dir = Path(__file__).parent.absolute()
    os.chdir(base_dir)
    
    incident_dirs = sorted([d.name for d in base_dir.glob("incident_*") if d.is_dir()])
    print(f"Found {len(incident_dirs)} incidents. Starting log synthesis...")
    
    for inc_dir in incident_dirs:
        incident_id = inc_dir.split("_")[1]
        generate_logs_for_incident(incident_id)

if __name__ == "__main__":
    main()
