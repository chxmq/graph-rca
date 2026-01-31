# Dataset Expansion

Expand the GraphRCA incident dataset from 60 to 100+ incidents.

## Manual Curation (Recommended)

Same approach as the original 60 incidents - read postmortems, extract root cause manually.

### Interactive Mode

```bash
cd data/real_incidents/expansion
python add_incidents.py
```

This will prompt you for:
- Company name
- Category (Database, Infrastructure, Security, etc.)
- Date
- Severity
- Source URL
- Root cause (one sentence)
- Postmortem summary

### Batch Mode

Edit `add_incidents.py` and use the `create_incident()` function:

```python
create_incident(
    incident_id=61,
    company="Stripe",
    category="Database",
    date="2023-03-15",
    severity="Critical",
    source_url="https://stripe.com/blog/outage",
    root_cause="Database connection pool exhaustion due to connection leak.",
    postmortem_summary="Payment service timeouts caused by gradual pool exhaustion..."
)
```

## Good Sources for Postmortems

1. **[danluu/post-mortems](https://github.com/danluu/post-mortems)** - 100+ postmortem links
2. **Company engineering blogs:**
   - Cloudflare, GitHub, GitLab, Stripe
   - AWS, Google Cloud, Azure
   - Slack, Discord, Reddit
3. **SRE Weekly archives**

## Target Distribution

Aim for ~15-20 incidents per category:

| Category | Current | Target |
|----------|---------|--------|
| Database | ~12 | 20 |
| Infrastructure | ~10 | 20 |
| Security | ~8 | 15 |
| Memory | ~9 | 15 |
| Network | ~10 | 15 |
| Software | ~11 | 15 |
| **Total** | 60 | 100+ |

## After Adding Incidents

The overnight experiments will automatically pick up new incidents:

```bash
cd experiments/overnight_gpu
python run_overnight.py
```

No changes needed - it reads all `incident_*` folders automatically.
