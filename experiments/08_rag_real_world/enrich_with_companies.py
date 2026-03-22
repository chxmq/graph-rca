#!/usr/bin/env python3
"""
Enrich RAG comparison results with company names.
Creates the heterogeneous effects analysis data for the paper.

Usage: python enrich_with_companies.py
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"
DATA_DIR = Path(__file__).parent / "data"


def load_company_map() -> dict:
    """Load incident_id -> company mapping from metadata files."""
    company_map = {}
    for folder in INCIDENT_DIR.glob("incident_*"):
        meta_file = folder / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.load(open(meta_file))
                company_map[folder.name] = meta.get("company", "Unknown")
            except:
                pass
    return company_map


def analyze_results(rag_file: Path):
    """Run analysis on specific result file."""
    if not rag_file.exists():
        print(f"✗ {rag_file} not found.")
        return
    
    rag_data = json.load(open(rag_file))
    company_map = load_company_map()
    
    print(f"\nAnalyzing {rag_file.name}...")
    print(f"Loaded {len(company_map)} company mappings")
    print(f"Processing {len(rag_data.get('tests', []))} test results...")
    
    # Enrich results with company names
    company_results = defaultdict(lambda: {"baseline": [], "rag": [], "incidents": []})
    valid_count = 0
    
    for test in rag_data.get("tests", []):
        if "baseline_score" not in test:
            continue  # Skip errors
        
        incident_id = test["id"]
        company = company_map.get(incident_id, "Unknown")
        
        company_results[company]["baseline"].append(test["baseline_score"])
        company_results[company]["rag"].append(test["rag_score"])
        company_results[company]["incidents"].append(incident_id)
        valid_count += 1
    
    print(f"Valid tests: {valid_count}")
    print(f"Companies: {len(company_results)}")
    
    # Calculate per-company averages
    summary = []
    for company, data in company_results.items():
        if data["baseline"]:
            baseline_avg = sum(data["baseline"]) / len(data["baseline"]) * 100
            rag_avg = sum(data["rag"]) / len(data["rag"]) * 100
            improvement = rag_avg - baseline_avg
            summary.append({
                "company": company,
                "n": len(data["baseline"]),
                "baseline_pct": round(baseline_avg, 1),
                "rag_pct": round(rag_avg, 1),
                "improvement_pp": round(improvement, 1)
            })
    
    # Sort and categorize
    improved = [s for s in summary if s["improvement_pp"] > 5]
    degraded = [s for s in summary if s["improvement_pp"] < -5]
    neutral = [s for s in summary if -5 <= s["improvement_pp"] <= 5]
    
    print("\n" + "=" * 60)
    print("📈 RAG IMPROVED (Δ > 5pp):")
    for s in sorted(improved, key=lambda x: -x["improvement_pp"])[:6]:
        print(f"  {s['company']}: {s['baseline_pct']:.0f}% → {s['rag_pct']:.0f}% (+{s['improvement_pp']:.0f}pp)")
    
    print("\n📉 RAG DEGRADED (Δ < -5pp):")
    for s in sorted(degraded, key=lambda x: x["improvement_pp"])[:6]:
        print(f"  {s['company']}: {s['baseline_pct']:.0f}% → {s['rag_pct']:.0f}% ({s['improvement_pp']:.0f}pp)")
    
    print(f"\n⚖️ NEUTRAL (-5pp ≤ Δ ≤ 5pp): {len(neutral)} companies")
    print("=" * 60)
    
    # Save enriched data
    output = {
        "source": "02_rag_comparison.json",
        "valid_tests": valid_count,
        "overall_baseline_pct": round(rag_data.get("baseline_avg", 0) * 100, 1),
        "overall_rag_pct": round(rag_data.get("rag_avg", 0) * 100, 1),
        "improved": sorted(improved, key=lambda x: -x["improvement_pp"]),
        "degraded": sorted(degraded, key=lambda x: x["improvement_pp"]),
        "neutral": neutral,
        "all_companies": summary
    }
    
    out_path = DATA_DIR / f"heterogeneous_effects_{rag_file.stem}.json"
    json.dump(output, open(out_path, "w"), indent=2)
    print(f"✓ Saved to {out_path}")

def main():
    import sys
    
    # Default file or argument
    if len(sys.argv) > 1:
        files = [Path(sys.argv[1])]
    else:
        files = list(DATA_DIR.glob("rag_comparison_*.json"))
        
    if not files:
        # Fallback to old name if verified
        files = [DATA_DIR / "02_rag_comparison.json"]
        
    for f in files:
        if f.exists():
            analyze_results(f)
        else:
             print(f"No result files found in {DATA_DIR}")


if __name__ == "__main__":
    main()
