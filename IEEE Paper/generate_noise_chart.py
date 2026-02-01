#!/usr/bin/env python3
"""Generate noise sensitivity chart for paper."""
import matplotlib.pyplot as plt
import numpy as np

# Data from experiment
noise_levels = [0, 100, 250, 500, 750, 1000]
accuracy = [95.0, 95.0, 85.0, 70.0, 70.0, 65.0]

# Create figure
plt.figure(figsize=(8, 5))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot line
plt.plot(noise_levels, accuracy, 'b-o', linewidth=2, markersize=8, label='Retrieval Accuracy')

# Fill regions
plt.axhspan(90, 100, alpha=0.2, color='green', label='Robust Zone')
plt.axhspan(60, 75, alpha=0.2, color='orange', label='Degraded Zone')

# Labels
plt.xlabel('Number of Decoy Documents', fontsize=12)
plt.ylabel('Retrieval Accuracy (%)', fontsize=12)
plt.title('Noise Sensitivity Analysis\nRAG Retrieval Accuracy vs. Corpus Contamination', fontsize=14)

# Axis settings
plt.xlim(-50, 1050)
plt.ylim(50, 100)
plt.xticks(noise_levels)
plt.yticks([50, 60, 70, 80, 90, 100])

# Legend
plt.legend(loc='lower left')

# Annotations
plt.annotate('95%', (0, 95), textcoords="offset points", xytext=(10, 5), fontsize=10)
plt.annotate('65%', (1000, 65), textcoords="offset points", xytext=(-30, -15), fontsize=10)

plt.tight_layout()
plt.savefig('fig9_noise_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig('fig9_noise_sensitivity.pdf', bbox_inches='tight')
print("Saved: fig9_noise_sensitivity.png and .pdf")
