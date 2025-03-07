# Re-import necessary libraries
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt errors

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# Recreate the dataset for the new experiment "Stuck-at-High-Random"
data_high_random = {
    "TestName": (
        ["Stuck-at-High-Random-25%"] * 5 + ["Stuck-at-High-Random-50%"] * 5 +
        ["Stuck-at-High-Random-75%"] * 5 + ["Stuck-at-High-Random-90%"] * 5 +
        ["Stuck-at-High-Random-100%"] * 5
    ),
    "OriginalAccuracy": (
        [100.00] * 5 + [96.88] * 5 + [100.00] * 5 + [81.25] * 5 +
        [100.00] * 5
    ),
    "StuckTrainAccuracy": [
        82.81, 98.44, 100.00, 98.44, 95.31, 
        93.75, 70.31, 87.50, 85.94, 68.75, 
        50.00, 68.75, 48.44, 68.75, 67.19, 
        46.88, 37.50, 48.44, 50.00, 42.19, 
        28.12, 28.12, 28.12, 28.12, 28.12
    ]
}

# Convert to DataFrame
df_high_random = pd.DataFrame(data_high_random)

# Calculate mean and variance for each test category
summary_high_random = df_high_random.groupby("TestName").agg(
    MeanOriginalAccuracy=("OriginalAccuracy", "mean"),
    MeanStuckTrainAccuracy=("StuckTrainAccuracy", "mean"),
    VarianceOriginalAccuracy=("OriginalAccuracy", "std"),
    VarianceStuckTrainAccuracy=("StuckTrainAccuracy", "std")
).reset_index()

# Add an "Overall" category for Original Accuracy bar (replace None with NaN)
overall_accuracy_high_random = {
    "TestName": "Overall Accuracy",
    "MeanOriginalAccuracy": df_high_random["OriginalAccuracy"].mean(),
    "MeanStuckTrainAccuracy": np.nan,  # Use np.nan instead of None
    "VarianceOriginalAccuracy": df_high_random["OriginalAccuracy"].std(),
    "VarianceStuckTrainAccuracy": np.nan  # Use np.nan instead of None
}

# Append overall accuracy to the summary DataFrame
summary_high_random = pd.concat([pd.DataFrame([overall_accuracy_high_random]), summary_high_random], ignore_index=True)

# Sort the summary DataFrame in the required order
order_high_random = ["Overall Accuracy", "Stuck-at-High-Random-25%", "Stuck-at-High-Random-50%", 
                     "Stuck-at-High-Random-75%", "Stuck-at-High-Random-90%", "Stuck-at-High-Random-100%"]
summary_high_random["TestName"] = pd.Categorical(summary_high_random["TestName"], categories=order_high_random, ordered=True)
summary_high_random = summary_high_random.sort_values("TestName")

# Define colors
overall_color = "lightblue"
overall_variance_color = "blue"
stuck_color = "lightcoral"  # Different color for "Stuck-at-High-Random" experiment
stuck_variance_color = "darkred"

# Get x positions
x_high_random = np.arange(len(summary_high_random))

# Plot bars with a shaded variance region as full width of bars
fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

bar_width = 0.6  # Match the width of variance bars to the bars themselves

for i in range(len(summary_high_random)):
    if summary_high_random["TestName"].iloc[i] == "Overall Accuracy":  # Overall Accuracy
        ax.bar(x_high_random[i], summary_high_random["MeanOriginalAccuracy"].iloc[i], color=overall_color, width=bar_width)
        ax.bar(x_high_random[i], summary_high_random["VarianceOriginalAccuracy"].iloc[i] * 2, 
               bottom=summary_high_random["MeanOriginalAccuracy"].iloc[i] - summary_high_random["VarianceOriginalAccuracy"].iloc[i],
               color=overall_variance_color, width=bar_width)
    else:  # Stuck-at-High-Random categories
        ax.bar(x_high_random[i], summary_high_random["MeanStuckTrainAccuracy"].iloc[i], color=stuck_color, width=bar_width)
        ax.bar(x_high_random[i], summary_high_random["VarianceStuckTrainAccuracy"].iloc[i] * 2, 
               bottom=summary_high_random["MeanStuckTrainAccuracy"].iloc[i] - summary_high_random["VarianceStuckTrainAccuracy"].iloc[i],
               color=stuck_variance_color, width=bar_width)

# Labels and title
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x_high_random)
ax.set_xticklabels(summary_high_random["TestName"], rotation=45, ha="right")

# Ensure x and y axis lines are visible
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

# Enable grid for y-axis for better readability
ax.yaxis.grid(True, linestyle="--", alpha=0.7)

# Legend
legend_patches = [
    mpatches.Patch(color=overall_color, label="Overall Accuracy"),
    mpatches.Patch(color=overall_variance_color, label="Overall Variance"),
    mpatches.Patch(color=stuck_color, label="Stuck-at-High-Random Accuracy"),
    mpatches.Patch(color=stuck_variance_color, label="Stuck-at-High-Random Variance")
]
ax.legend(handles=legend_patches, loc="best", frameon=False)

# Save the figure
plt.savefig("accuracy_plot_high_random.png", bbox_inches="tight", dpi=300)

# Show the plot
plt.show()
