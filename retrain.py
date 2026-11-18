
import joblib
import os
import json
import shutil

from pathlib import Path
from datetime import datetime

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("===== MODEL RETRAINING STARTED =====")

# Create model versions directory
version_dir = Path("model_versions")
version_dir.mkdir(exist_ok=True)

# Generate a unique model version
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

version_name = f"model_{timestamp}"

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed!")

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)

MIN_ACCURACY = 0.90

if accuracy >= MIN_ACCURACY:

    print("Model validation passed!")

    candidate_path = Path("model_candidate.pkl")

    # Save candidate model
    joblib.dump(model, candidate_path)

    # Preserve the existing production model
    current_model = Path("model.pkl")

    if current_model.exists():
        previous_path = version_dir / f"{version_name}_previous.pkl"
        shutil.copy2(current_model, previous_path)
        print("Previous model archived:", previous_path)

    # Save new version
    version_path = version_dir / f"{version_name}.pkl"
    shutil.copy2(candidate_path, version_path)

    # Save model metadata
    metadata = {
        "model_version": version_name,
        "algorithm": "RandomForestClassifier",
        "accuracy": accuracy,
        "minimum_accuracy": MIN_ACCURACY,
        "dataset": "Iris",
        "status": "promoted"
    }

    metadata_path = version_dir / f"{version_name}.json"

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    # Promote new model
    os.replace(candidate_path, current_model)

    print("New model version:", version_name)
    print("Model metadata saved:", metadata_path)
    print("New model promoted successfully!")

else:

    print("Model validation failed!")
    print("Existing model retained.")

    candidate_path = Path("model_candidate.pkl")

    if candidate_path.exists():
        candidate_path.unlink()

print("===== MODEL RETRAINING FINISHED =====")
