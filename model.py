import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("fraud.csv")

# Use only a few important columns for demo
df = df[['V1', 'V2', 'V3', 'V4', 'Amount', 'Class']]

X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'fraud_model.pkl')
