#!/usr/bin/env python3
"""
Statistical Analysis Script for GraphRCA Evaluation
Calculates confidence intervals, documentation dependency, and statistical rigor
"""

import json
import statistics
import math
from pathlib import Path
from collections import defaultdict

# Load RCA results
RESULTS_DIR = Path(__file__).parent / "eval_final_results"
RCA_FILE = RESULTS_DIR / "04_pipeline_rca.json"

# Ensure we can write to results directory
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(RCA_FILE) as f:
    rca_data = json.load(f)

# Documentation mapping (which scenarios have relevant docs)
# Based on corpus: Flask server operations, API endpoints, troubleshooting
DOCUMENTED_SCENARIOS = {
    "Database": ["DB Connection Pool Exhaustion", "Database Deadlock", "Replication Lag", "Query Degradation"],
    "Security": ["Brute Force Attack", "Certificate Expiration", "Token Cascade Failure"],
    "Application": ["Configuration Error", "Microservice Cascade", "Thread Pool Exhaustion", "Retry Storm"],
    "Monitoring": ["Alert Fatigue", "Metrics Pipeline Failure"],
    "Infrastructure": [],  # No docs for infrastructure scenarios
    "Memory": []  # No docs for memory scenarios
}

def calculate_confidence_interval(successes, total, confidence=0.95):
    """Calculate Wilson score confidence interval for proportion"""
    if total == 0:
        return (0.0, 0.0)
    
    # Z-score for 95% confidence = 1.96
    z = 1.96 if confidence == 0.95 else 2.576  # 99% = 2.576
    p = successes / total
    
    denominator = 1 + (z**2 / total)
    centre_adjusted_probability = (p + (z**2 / (2 * total))) / denominator
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
    upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
    
    return (max(0, lower_bound) * 100, min(1, upper_bound) * 100)

def analyze_rca_statistics():
    """Calculate comprehensive RCA statistics"""
    
    all_runs = []
    by_category = defaultdict(lambda: {"successes": 0, "total": 0, "runs": []})
    documented_runs = {"successes": 0, "total": 0}
    undocumented_runs = {"successes": 0, "total": 0}
    
    # Collect all run data
    for scenario in rca_data["results"]:
        category = scenario["category"]
        scenario_name = scenario["name"]
        has_docs = scenario_name in DOCUMENTED_SCENARIOS.get(category, [])
        
        for run in scenario["runs"]:
            success = run["success"]
            all_runs.append(success)
            by_category[category]["runs"].append(success)
            by_category[category]["successes"] += int(success)
            by_category[category]["total"] += 1
            
            if has_docs:
                documented_runs["successes"] += int(success)
                documented_runs["total"] += 1
            else:
                undocumented_runs["successes"] += int(success)
                undocumented_runs["total"] += 1
    
    # Overall statistics
    total_runs = len(all_runs)
    total_successes = sum(all_runs)
    overall_accuracy = (total_successes / total_runs) * 100 if total_runs > 0 else 0
    ci_lower, ci_upper = calculate_confidence_interval(total_successes, total_runs)
    
    # Category statistics
    category_stats = {}
    for category, data in by_category.items():
        if data["total"] > 0:
            accuracy = (data["successes"] / data["total"]) * 100
            ci_lower_cat, ci_upper_cat = calculate_confidence_interval(data["successes"], data["total"])
            category_stats[category] = {
                "accuracy": round(accuracy, 1),
                "successes": data["successes"],
                "total": data["total"],
                "ci_95_lower": round(ci_lower_cat, 1),
                "ci_95_upper": round(ci_upper_cat, 1)
            }
    
    # Documentation dependency
    doc_accuracy = (documented_runs["successes"] / documented_runs["total"] * 100) if documented_runs["total"] > 0 else 0
    no_doc_accuracy = (undocumented_runs["successes"] / undocumented_runs["total"] * 100) if undocumented_runs["total"] > 0 else 0
    
    doc_ci_lower, doc_ci_upper = calculate_confidence_interval(documented_runs["successes"], documented_runs["total"])
    no_doc_ci_lower, no_doc_ci_upper = calculate_confidence_interval(undocumented_runs["successes"], undocumented_runs["total"])
    
    # Standard deviation
    std_dev = statistics.stdev([1 if r else 0 for r in all_runs]) * 100 if len(all_runs) > 1 else 0
    
    return {
        "overall": {
            "accuracy": round(overall_accuracy, 1),
            "successes": total_successes,
            "total": total_runs,
            "std_dev": round(std_dev, 2),
            "ci_95_lower": round(ci_lower, 1),
            "ci_95_upper": round(ci_upper, 1),
            "ci_95": f"{round(ci_lower, 1)}%--{round(ci_upper, 1)}%"
        },
        "by_category": category_stats,
        "documentation_dependency": {
            "with_documentation": {
                "accuracy": round(doc_accuracy, 1),
                "successes": documented_runs["successes"],
                "total": documented_runs["total"],
                "ci_95": f"{round(doc_ci_lower, 1)}%--{round(doc_ci_upper, 1)}%"
            },
            "without_documentation": {
                "accuracy": round(no_doc_accuracy, 1),
                "successes": undocumented_runs["successes"],
                "total": undocumented_runs["total"],
                "ci_95": f"{round(no_doc_ci_lower, 1)}%--{round(no_doc_ci_upper, 1)}%"
            },
            "difference": round(doc_accuracy - no_doc_accuracy, 1)
        }
    }

def generate_latex_table(stats):
    """Generate LaTeX table with confidence intervals"""
    
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Root Cause Identification by Category with 95\\% Confidence Intervals}\n"
    latex += "\\label{tab:rca_statistical}\n"
    latex += "\\begin{tabular}{lccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Category} & \\textbf{Accuracy} & \\textbf{95\\% CI} & \\textbf{Runs} \\\\\n"
    latex += "\\midrule\n"
    
    for category in ["Database", "Security", "Application", "Monitoring", "Infrastructure", "Memory"]:
        if category in stats["by_category"]:
            cat_stat = stats["by_category"][category]
            latex += f"{category} & {cat_stat['accuracy']:.1f}\\% & "
            latex += f"({cat_stat['ci_95_lower']:.1f}--{cat_stat['ci_95_upper']:.1f}\\% ) & "
            latex += f"{cat_stat['total']} \\\\\n"
    
    latex += "\\midrule\n"
    latex += f"\\textbf{{Overall}} & \\textbf{{{stats['overall']['accuracy']:.1f}\\%}} & "
    latex += f"\\textbf{{({stats['overall']['ci_95_lower']:.1f}--{stats['overall']['ci_95_upper']:.1f}\\% )}} & "
    latex += f"\\textbf{{{stats['overall']['total']}}} \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def generate_doc_dependency_table(stats):
    """Generate LaTeX table for documentation dependency"""
    
    doc_stat = stats["documentation_dependency"]
    
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Documentation Dependency Analysis}\n"
    latex += "\\label{tab:doc_dependency}\n"
    latex += "\\begin{tabular}{lccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Condition} & \\textbf{Accuracy} & \\textbf{95\\% CI} & \\textbf{Runs} \\\\\n"
    latex += "\\midrule\n"
    
    latex += f"With Documentation & {doc_stat['with_documentation']['accuracy']:.1f}\\% & "
    latex += f"({doc_stat['with_documentation']['ci_95']}) & "
    latex += f"{doc_stat['with_documentation']['total']} \\\\\n"
    
    latex += f"Without Documentation & {doc_stat['without_documentation']['accuracy']:.1f}\\% & "
    latex += f"({doc_stat['without_documentation']['ci_95']}) & "
    latex += f"{doc_stat['without_documentation']['total']} \\\\\n"
    
    latex += "\\midrule\n"
    latex += f"\\textbf{{Difference}} & \\textbf{{{doc_stat['difference']:.1f} pp}} & & \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

if __name__ == "__main__":
    print("=" * 70)
    print("Statistical Analysis for GraphRCA RCA Evaluation")
    print("=" * 70)
    print()
    
    stats = analyze_rca_statistics()
    
    # Save JSON results
    output_file = RESULTS_DIR / "06_statistical_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Accuracy: {stats['overall']['accuracy']:.1f}%")
    print(f"95% Confidence Interval: {stats['overall']['ci_95']}")
    print(f"Standard Deviation: {stats['overall']['std_dev']:.2f}%")
    print(f"Total Runs: {stats['overall']['total']}")
    print(f"Successes: {stats['overall']['successes']}")
    
    print("\n" + "=" * 70)
    print("BY CATEGORY")
    print("=" * 70)
    for category, cat_stat in stats["by_category"].items():
        print(f"{category:15s} {cat_stat['accuracy']:5.1f}%  CI: ({cat_stat['ci_95_lower']:.1f}--{cat_stat['ci_95_upper']:.1f}%)  n={cat_stat['total']}")
    
    print("\n" + "=" * 70)
    print("DOCUMENTATION DEPENDENCY")
    print("=" * 70)
    doc_stat = stats["documentation_dependency"]
    print(f"With Documentation:    {doc_stat['with_documentation']['accuracy']:.1f}%  CI: ({doc_stat['with_documentation']['ci_95']})  n={doc_stat['with_documentation']['total']}")
    print(f"Without Documentation: {doc_stat['without_documentation']['accuracy']:.1f}%  CI: ({doc_stat['without_documentation']['ci_95']})  n={doc_stat['without_documentation']['total']}")
    print(f"Difference:            {doc_stat['difference']:.1f} percentage points")
    
    # Generate LaTeX tables
    latex_file = RESULTS_DIR / "statistical_tables.tex"
    with open(latex_file, 'w') as f:
        f.write("% Statistical Analysis Tables\n")
        f.write("% Generated automatically from RCA results\n\n")
        f.write(generate_latex_table(stats))
        f.write("\n\n")
        f.write(generate_doc_dependency_table(stats))
    
    print(f"\n✓ Saved LaTeX tables: {latex_file}")
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
