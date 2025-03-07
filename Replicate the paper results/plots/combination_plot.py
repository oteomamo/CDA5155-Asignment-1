import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt errors

import matplotlib.pyplot as plt
import numpy as np

# Data
error_scenarios = [
    "Low Error\n(5% Stuck-Zero, 5% Saturation, 10% Noise)",
    "Moderate Error\n(10% Stuck-Zero, 15% Saturation, 20% Noise)",
    "High Error\n(20% Stuck-Zero, 30% Saturation, 40% Noise)"
]
mean_accuracy = [82.81, 72.50, 59.38]
std_deviation = [6.81, 3.60, 7.33]

# Line plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(error_scenarios, mean_accuracy, yerr=std_deviation, fmt='-o', capsize=10,
            color='blue', ecolor='red', elinewidth=2, markerfacecolor='green', markersize=8)

# Add labels and title
ax.set_ylabel('Mean Accuracy (%)', fontsize=14)
ax.set_xlabel('Error Scenario', fontsize=14)
#ax.set_title('Impact of Combinational Errors on Model Accuracy', fontsize=16)

# Annotate points
for x, y in zip(error_scenarios, mean_accuracy):
    ax.annotate(f'{y:.2f}%', xy=(x, y), xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('combination_errors_accuracy_line_plot.png', dpi=300)
plt.close()