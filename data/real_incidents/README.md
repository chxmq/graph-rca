# Real Incidents Dataset

This directory contains 60 manually curated real-world incidents from production systems.

## Dataset Structure

Each incident is stored in a directory `incident_XXX/` with the following files:

- **`metadata.json`**: Company name, incident category, date, severity
- **`ground_truth.json`**: Root cause, timestamp, source URL, logs availability
- **`postmortem.md`**: Human-readable summary of the incident
- **`logs.txt`** (if available): Synthetic or real log evidence

## Dataset Statistics

- **Total Incidents**: 60
- **Categories**: Database, Software, Network, Infrastructure, Security, Hardware
- **Date Range**: 2009-2024
- **Companies**: GitHub, Cloudflare, AWS, Google, Microsoft, and 40+ others

## Generation Scripts

### Data Collection
- `collect_data.sh`: Automated scraping from GitHub and SRE Weekly
- `collect_incidents.py`: Python library for organizing incident data

### Manual Curation
- `populate_initial_incidents.py`: Batch 1 (incidents 001-010)
- `populate_batch_2.py`: Batch 2 (incidents 011-020)
- `populate_batch_3.py`: Batch 3 (incidents 021-030)
- `populate_batch_4.py`: Batch 4 (incidents 031-040)
- `populate_batch_5.py`: Batch 5 (incidents 041-050)
- `populate_batch_6.py`: Batch 6 (incidents 051-060)

### Log Synthesis
- `generate_synthetic_logs.py`: Generates semi-realistic logs for incidents

## Raw Sources

The `sources/` directory contains:
- `raw/github/`: 190 incident post-mortems from GitHub repositories
- `raw/sre_weekly/`: 49 incidents from SRE Weekly archives

## Usage

```python
from collect_incidents import IncidentDataCollector

collector = IncidentDataCollector()
# Access incident data programmatically
```

## Citation

If you use this dataset in your research, please cite:

```
@dataset{graphrca_incidents_2024,
  title={Real-World Production Incidents Dataset},
  author={GraphRCA Project},
  year={2024},
  url={https://github.com/chxmq/graph-rca}
}
```
