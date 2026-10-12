
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Step 1: Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Step 2: Load the diabetes dataset
data = load_diabetes(as_frame=True)

X = data.frame.drop(columns=["target"])
y = data.frame["target"]

# Step 3: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# Step 4: Create the machine learning pipeline
model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Evaluate the model
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Step 7: Save the trained model and metrics
output_dir = Path("artifacts")
output_dir.mkdir(exist_ok=True)

joblib.dump(model, output_dir / "model.joblib")

metrics = {
    "model_version": "v0.1",
    "model": "StandardScaler + LinearRegression",
    "rmse": float(rmse),
    "random_seed": RANDOM_SEED,
    "test_size": 0.2
}

with open(output_dir / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=2)

print(f"Model v0.1 RMSE: {rmse:.2f}")
print("Model and metrics saved successfully!")
