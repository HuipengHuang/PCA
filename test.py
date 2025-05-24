import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual data)
calibration_sizes = np.array([6000, 7000, 8000, 9000, 10000])
cov_gap = np.array([4.0, 3.8, 3.5, 3.2, 3.0])  # Example CovGap values

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(calibration_sizes, cov_gap, marker='o', linestyle='-', color='b', label='CovGap (%)')

# Add annotations

plt.text(8000, 3.6, "same\nhold-out", fontsize=10, color='green', ha='center')  # Multiline annotation

# Customize axes and labels
plt.xlabel("Size of Calibration Set", fontsize=12)
plt.ylabel("CovGap (%)", fontsize=12)
plt.title("CovGap vs. Calibration Set Size", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust x-axis ticks to match your example
plt.xticks(calibration_sizes)

# Add a legend (if needed)
plt.legend(loc='upper right')

# Save or display the plot
plt.tight_layout()

plt.show()