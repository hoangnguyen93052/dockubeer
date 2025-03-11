import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import seaborn as sns

# Generate synthetic dataset
def generate_data(n_samples=1000, n_features=2, n_clusters=3, random_state=42):
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.60, random_state=random_state)
    return pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])

# Visualizing the data
def plot_data(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Feature 1'], data['Feature 2'], alpha=0.5)
    plt.title('Generated Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid()
    plt.show()

# Anomaly Detection using Isolation Forest
def isolation_forest(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    data['Anomaly'] = model.fit_predict(data[['Feature 1', 'Feature 2']])
    return data

# Anomaly Detection using Local Outlier Factor
def local_outlier_factor(data):
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    data['Anomaly_LOF'] = model.fit_predict(data[['Feature 1', 'Feature 2']])
    return data

# Visualize the anomalies found
def plot_anomalies(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Feature 1'], data['Feature 2'], c=data['Anomaly'], cmap='coolwarm', alpha=0.5)
    plt.title('Anomalies Detected by Isolation Forest')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Anomaly (-1: Anomaly, 1: Normal)')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Feature 1'], data['Feature 2'], c=data['Anomaly_LOF'], cmap='coolwarm', alpha=0.5)
    plt.title('Anomalies Detected by Local Outlier Factor')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Anomaly (-1: Anomaly, 1: Normal)')
    plt.grid()
    plt.show()

# Main function to run the anomaly detection
def main():
    # Generate Data
    data = generate_data(n_samples=1000)
    plot_data(data)

    # Apply Isolation Forest
    data_if = isolation_forest(data.copy())
    print("Isolation Forest Results:")
    print(data_if['Anomaly'].value_counts())
    
    # Plot Isolation Forest results
    plot_anomalies(data_if)

    # Apply Local Outlier Factor
    data_lof = local_outlier_factor(data.copy())
    print("Local Outlier Factor Results:")
    print(data_lof['Anomaly_LOF'].value_counts())
    
    # Plot Local Outlier Factor results
    plot_anomalies(data_lof)

if __name__ == "__main__":
    main()