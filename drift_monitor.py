
import numpy as np
from scipy.stats import ks_2samp

# Reference data: original data distribution
np.random.seed(42)

reference_data = np.random.normal(
    loc=0,
    scale=1,
    size=1000
)

# Current data: newly received data
current_data = np.random.normal(
    loc=0.5,
    scale=1,
    size=1000
)

# Perform KS test
statistic, p_value = ks_2samp(
    reference_data,
    current_data
)

print("===== DATA DRIFT MONITORING =====")

print("KS Statistic:", statistic)
print("P-value:", p_value)

# Check drift
if p_value < 0.05:
    print("DATA DRIFT DETECTED!")
    drift_detected = True
else:
    print("No significant data drift detected.")
    drift_detected = False

# GitHub Actions output
import os

github_output = os.getenv("GITHUB_OUTPUT")

if github_output:
    with open(github_output, "a") as file:
        file.write(
            f"drift_detected={str(drift_detected).lower()}\n"
        )
