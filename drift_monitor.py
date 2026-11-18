
import numpy as np
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
import os

print("===== IRIS DATA DRIFT MONITORING =====")

# Load actual Iris dataset
iris = load_iris()
X = iris.data

# Reference data
reference_data = X[:75]

# Simulate current incoming data
np.random.seed(42)

current_data = X[75:].copy()

# Introduce a small distribution change
current_data[:, 0] += 0.5

# Drift threshold
threshold = 0.05

drift_detected = False

# Compare all four features
feature_names = iris.feature_names

for i in range(X.shape[1]):

    statistic, p_value = ks_2samp(
        reference_data[:, i],
        current_data[:, i]
    )

    print(f"\nFeature: {feature_names[i]}")
    print("KS Statistic:", statistic)
    print("P-value:", p_value)

    if p_value < threshold:
        print("DRIFT DETECTED")
        drift_detected = True
    else:
        print("No significant drift")

# Final result
print("\n===== FINAL DRIFT RESULT =====")

if drift_detected:
    print("DATA DRIFT DETECTED!")
else:
    print("NO DATA DRIFT DETECTED!")

# GitHub Actions output
github_output = os.getenv("GITHUB_OUTPUT")

if github_output:
    with open(github_output, "a") as file:
        file.write(
            f"drift_detected={str(drift_detected).lower()}\n"
        )
