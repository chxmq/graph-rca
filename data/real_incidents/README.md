# Real-World Incidents Dataset

A curated dataset of **200 real-world production incidents** from 50+ companies, used to evaluate GraphRCA's root cause analysis capabilities.

## Dataset Structure

```
real_incidents/
├── manage.py              # Dataset management utility
├── README.md
├── incident_001/
│   ├── logs.txt           # Synthetic log entries (LLM-generated)
│   ├── postmortem.md      # Incident summary from public postmortem
│   ├── metadata.json      # Company, date, severity, category
│   └── ground_truth.json  # Annotated root cause
├── incident_002/
│   └── ...
└── incident_200/
```

## Usage

```bash
# Show dataset statistics
python manage.py stats

# Validate all incident files
python manage.py validate

# Generate synthetic logs for incidents missing them
python manage.py generate-logs

# Export dataset summary
python manage.py export
```

## Categories

| Category | Count |
|----------|-------|
| Infrastructure | 66 |
| Database | 46 |
| Software | 31 |
| Network | 24 |
| Security | 17 |
| Configuration | 6 |
| Memory | 6 |
| Cloud | 3 |
| CI-CD | 1 |

## Data Sources

Incidents curated from public postmortems including:
- GitHub, Cloudflare, AWS, Google Cloud
- Microsoft Azure, Datadog, Roblox
- Stack Exchange, Stripe, Etsy, and others

## Methodology

1. **Source identification**: Locate public postmortem/incident report
2. **Metadata extraction**: Record company, date, category, severity
3. **Ground truth annotation**: Extract root cause statement
4. **Log synthesis**: Generate logs using LLM (Llama 3.2 3B)
5. **Quality validation**: Verify logs contain root cause evidence

See the paper for full methodology details.
