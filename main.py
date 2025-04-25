import pandas as pd # type: ignore
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
import json
import os

# Load dataset
data_path = 'data/tested.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Please download train.csv and place it in the 'data' folder.")

df = pd.read_csv(data_path)

# Separate features and target
y = df['Survived']
X = df.drop('Survived', axis=1)

# Preprocess the data
X_processed, encoders, scaler = preprocess_data(X)

# Train the model and get evaluation metrics
model, metrics = train_model(X_processed, y)

# Create outputs folder if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Save evaluation metrics
with open('outputs/evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("✅ Model trained successfully!")
print("📊 Evaluation metrics saved to 'outputs/evaluation_metrics.json'")
