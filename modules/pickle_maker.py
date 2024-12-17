import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from dill.tests.test_recursive import Model


"""
Use Min-Max Scaling or Standardization for Amount and Time since PCA-transformed features are likely scaled.
"""
# Load dataset
data = pd.read_csv('../Data/creditcard.csv')

# Scale 'Time' and 'Amount' features
scaler = StandardScaler()
data['Time'] = scaler.fit_transform(data[['Time']])
data['Amount'] = scaler.fit_transform(data[['Amount']])

# Prepare feature matrix (drop 'Class')
X = data.drop(columns=['Class'])

# Store true labels for evaluation
y_true = data['Class']

"""
isolation forrest
"""
# Initialize Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.001,  # Approx. percentage of anomalies in the data
    random_state=42
)

# Fit the model
iso_forest.fit(X)

# Predict anomaly scores
anomaly_scores = iso_forest.decision_function(X)

# Predict anomalies
predictions = iso_forest.predict(X)
predictions = [1 if x == -1 else 0 for x in predictions]  # Convert to binary


"""
Evaluate the Model
"""
# Compute Precision-Recall values
precision, recall, _ = precision_recall_curve(y_true, -anomaly_scores)  # Use negative scores

# Calculate AUPRC
auprc = auc(recall, precision)
print(f"AUPRC: {auprc:.3f}")

"""Vis Recall curve"""
# Plot the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'AUPRC = {auprc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()


"""Vis Anomolies"""
# Add predictions to the dataset
data['Anomaly'] = predictions

# Plot anomalies against Time and Amount
plt.figure(figsize=(10, 6))
plt.scatter(data['Time'], data['Amount'], c=data['Anomaly'], cmap='coolwarm', alpha=0.6)
plt.title('Anomalies Detected')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show()
