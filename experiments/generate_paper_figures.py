#!/usr/bin/env python3
"""
===============================================================================
GraphRCA Paper Figure Generator
===============================================================================

Generates all figures for the IEEE Access paper from experiment data.

Figures generated:
- fig1_throughput.png     - Batch inference throughput comparison
- fig2_field_accuracy.png - Log parsing field accuracy (BGL/HDFS)
- fig3_dag_scalability.png - DAG construction scalability
- fig4_rca_categories.png - RCA accuracy by failure category
- fig5_ablation.png       - Documentation ablation study
- fig6_summary.png        - Overall system performance summary

Usage:
    python generate_paper_figures.py
    
Output:
    experiments/figures/fig*.png
    experiments/figures/fig*.pdf
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def save_figure(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches='tight')
    print(f"  Saved: {name}.png, {name}.pdf")
    plt.close(fig)

def fig1_throughput():
    """Batch inference throughput comparison."""
    print("Generating fig1_throughput...")
    
    # Data from paper Table II
    batch_sizes = [1, 8, 16, 32]
    throughput = [0.40, 0.48, 0.82, 2.27]  # logs/s
    latency = [2515, 2069, 1224, 441]  # ms
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    # Throughput bars
    bars1 = ax1.bar(x - width/2, throughput, width, label='Throughput (logs/s)', color='steelblue')
    ax1.set_ylabel('Throughput (logs/s)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, 3)
    
    # Latency line on secondary axis
    ax2 = ax1.twinx()
    line = ax2.plot(x, latency, 'ro-', linewidth=2, markersize=8, label='Latency (ms)')
    ax2.set_ylabel('Latency (ms)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 3000)
    
    ax1.set_xlabel('Batch Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.set_title('Batch Inference Performance\n(Llama 3.2 3B on NVIDIA Quadro GV100)')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Annotate speedup
    ax1.annotate('5.7× speedup', xy=(3, 2.27), xytext=(2.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green', fontweight='bold')
    
    save_figure(fig, 'fig1_throughput')

def fig2_field_accuracy():
    """Log parsing field accuracy comparison."""
    print("Generating fig2_field_accuracy...")
    
    # Data from paper Table VII
    datasets = ['BGL', 'HDFS']
    drain_accuracy = [100.0, 100.0]
    graphrca_accuracy = [99.6, 99.2]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, drain_accuracy, width, label='Drain (Baseline)', color='gray')
    bars2 = ax.bar(x + width/2, graphrca_accuracy, width, label='GraphRCA (Ours)', color='steelblue')
    
    ax.set_ylabel('Field Extraction Accuracy (%)')
    ax.set_xlabel('Dataset')
    ax.set_title('Log Parsing Accuracy Comparison\n(Field-level extraction on LogHub datasets)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(98, 101)
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    
    save_figure(fig, 'fig2_field_accuracy')

def fig3_dag_scalability():
    """DAG construction scalability."""
    print("Generating fig3_dag_scalability...")
    
    # Data from paper Table V
    log_entries = [100, 500, 1000, 2000]
    dag_time_ms = [0.93, 4.91, 9.71, 19.68]
    std_dev = [0.03, 0.21, 0.51, 0.53]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.errorbar(log_entries, dag_time_ms, yerr=std_dev, fmt='bo-', 
               linewidth=2, markersize=8, capsize=5, label='Measured')
    
    # Theoretical O(n) line
    theoretical = [0.01 * n for n in log_entries]
    ax.plot(log_entries, theoretical, 'r--', linewidth=1.5, alpha=0.7, label='O(n) theoretical')
    
    ax.set_xlabel('Number of Log Entries')
    ax.set_ylabel('DAG Construction Time (ms)')
    ax.set_title('DAG Construction Scalability\n(Linear O(n) complexity confirmed)')
    ax.legend()
    ax.set_xlim(0, 2200)
    ax.set_ylim(0, 25)
    
    # Annotate key point
    ax.annotate('<20ms for 2000 entries', xy=(2000, 19.68), xytext=(1400, 22),
               arrowprops=dict(arrowstyle='->', color='green'),
               fontsize=10, color='green', fontweight='bold')
    
    save_figure(fig, 'fig3_dag_scalability')

def fig4_rca_categories():
    """RCA accuracy by failure category."""
    print("Generating fig4_rca_categories...")
    
    # Data from paper Table IV
    categories = ['Database', 'Security', 'Application', 'Infrastructure', 'Memory', 'Monitoring']
    accuracy = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]  # From run_log.txt
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    bars = ax.bar(categories, accuracy, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('RCA Accuracy (%)')
    ax.set_xlabel('Failure Category')
    ax.set_title('Root Cause Analysis Accuracy by Category\n(60 scenarios, 3 runs each)')
    ax.set_ylim(0, 100)
    
    # Add average line
    avg = np.mean(accuracy)
    ax.axhline(y=avg, color='red', linestyle='--', linewidth=2, label=f'Average: {avg:.1f}%')
    ax.legend()
    
    # Add value labels
    for bar in bars:
        ax.annotate(f'{bar.get_height():.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    save_figure(fig, 'fig4_rca_categories')

def fig5_ablation():
    """Documentation ablation study."""
    print("Generating fig5_ablation...")
    
    # Data from run_log.txt
    configs = ['Full Docs\n(3 docs)', 'Partial Docs\n(1 doc)', 'No Docs\n(0 docs)']
    accuracy = [93.3, 86.7, 100.0]  # From experiment
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['steelblue', 'orange', 'gray']
    bars = ax.bar(configs, accuracy, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('RCA Accuracy (%)')
    ax.set_xlabel('Documentation Configuration')
    ax.set_title('Documentation Ablation Study\n(Impact of RAG context on RCA accuracy)')
    ax.set_ylim(0, 110)
    
    # Add value labels
    for bar in bars:
        ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    
    # Note about surprising result
    ax.annotate('* No docs performed best\n(clear log evidence)', 
               xy=(2, 100), xytext=(1.5, 85),
               fontsize=9, style='italic', color='gray')
    
    save_figure(fig, 'fig5_ablation')

def fig6_summary():
    """Overall system performance summary."""
    print("Generating fig6_summary...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Throughput by batch size
    ax1 = axes[0, 0]
    batch_sizes = [1, 8, 16, 32]
    throughput = [0.40, 0.48, 0.82, 2.27]
    ax1.bar(batch_sizes, throughput, color='steelblue')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (logs/s)')
    ax1.set_title('Batch Inference Throughput')
    ax1.set_xticks(batch_sizes)
    
    # Subplot 2: Parsing accuracy
    ax2 = axes[0, 1]
    datasets = ['BGL', 'HDFS']
    accuracy = [99.6, 99.2]
    ax2.bar(datasets, accuracy, color='green')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Log Parsing Accuracy')
    ax2.set_ylim(98, 100.5)
    
    # Subplot 3: DAG scalability
    ax3 = axes[1, 0]
    entries = [100, 500, 1000, 2000]
    time_ms = [0.93, 4.91, 9.71, 19.68]
    ax3.plot(entries, time_ms, 'bo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Log Entries')
    ax3.set_ylabel('DAG Build Time (ms)')
    ax3.set_title('DAG Construction Scalability')
    
    # Subplot 4: Key metrics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    metrics = [
        ['Metric', 'Value'],
        ['RCA Accuracy', '77.8%'],
        ['Parsing Accuracy', '99.4%'],
        ['Max Throughput', '2.27 logs/s'],
        ['DAG Build (2K logs)', '<20ms'],
        ['Total Experiments', '200 incidents'],
    ]
    table = ax4.table(cellText=metrics, loc='center', cellLoc='center',
                     colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2)
    ax4.set_title('Key Performance Metrics', pad=20)
    
    plt.suptitle('GraphRCA System Performance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_figure(fig, 'fig6_summary')

def main():
    print("=" * 60)
    print("GraphRCA Paper Figure Generator")
    print("=" * 60)
    print(f"Output directory: {FIGURES_DIR}")
    print()
    
    fig1_throughput()
    fig2_field_accuracy()
    fig3_dag_scalability()
    fig4_rca_categories()
    fig5_ablation()
    fig6_summary()
    
    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
