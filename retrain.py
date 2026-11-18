
import joblib
import os

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("===== MODEL RETRAINING STARTED =====")

# 1. Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create a new model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 4. Train the model
model.fit(X_train, y_train)

print("New model training completed!")

# 5. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)

# 6. Validate the model
MIN_ACCURACY = 0.90

if accuracy >= MIN_ACCURACY:

    print("Accuracy requirement passed!")

    # Save candidate model first
    candidate_path = "model_candidate.pkl"

    joblib.dump(model, candidate_path)

    # Promote candidate model
    os.replace(candidate_path, "model.pkl")

    print("New model promoted successfully!")
    print("Updated model saved as model.pkl")

else:

    print("Model validation failed!")
    print("Existing model will be retained.")

    if os.path.exists("model_candidate.pkl"):
        os.remove("model_candidate.pkl")

print("===== MODEL RETRAINING FINISHED =====")
