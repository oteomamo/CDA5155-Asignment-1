import matplotlib
matplotlib.use('Agg')  # <-- Add this line at the very beginning
import matplotlib.pyplot as plt
import numpy as np

# Data preparation
noise_levels = [0.01, 0.05, 0.1, 0.2]

train_means = [
    np.mean([96.88, 96.88, 84.38, 89.06, 96.88]),
    np.mean([90.62, 100.0, 78.12, 90.62, 93.75]),
    np.mean([100.00, 98.44, 95.31, 100.00, 87.50]),
    np.mean([76.56, 98.44, 92.19, 95.31, 92.19])
]

train_stds = [
    np.std([96.88, 96.88, 84.38, 89.06]),
    np.std([90.62, 100.00, 78.12, 90.62, 93.75]),
    np.std([100.00, 98.44, 95.31, 100.00, 87.50]),
    np.std([76.56, 98.44, 92.19, 95.31, 92.19])
]

eval_means = [
    96.88,
    100.00,
    np.mean([96.88, 96.88, 98.44, 98.44, 96.88]),
    np.mean([95.31, 96.88, 96.88, 93.75, 96.88])
]

eval_stds = [
    0.0,
    0.0,
    np.std([96.88, 96.88, 98.44, 98.44, 96.88]),
    np.std([95.31, 96.88, 96.88, 93.75, 96.88])
]

# Plotting
plt.figure(figsize=(8, 5))
plt.errorbar(noise_levels, train_means, yerr=train_stds, label='Train & Eval Noise', marker='o', capsize=4)
plt.plot(noise_levels, eval_means, 'o--', label='Eval Noise Only')
plt.fill_between(noise_levels,
                 np.array(train_means) - np.array(train_stds),
                 np.array(train_means) + np.array(train_stds),
                 alpha=0.2)

plt.xticks(noise_levels)
plt.xlabel('Noise Standard Deviation (σ)')
plt.ylabel('Accuracy (%)')
#plt.title('Network Accuracy under Gaussian Noise')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig('accuracy_noise_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'accuracy_noise_plot.png'.")
