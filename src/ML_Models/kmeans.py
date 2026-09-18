import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Load dataset
df = pd.read_csv(
    r"C:\Users\asolo\Documents\ML\Placement-Prediction-System (3)\Placement-Prediction-System\data\placement_data.csv"
)

# Show available columns
print("Available columns:")
print(df.columns.tolist())

# Features to use
features = ["CGPA", "Internships", "Projects", "CodingTestScore"]
features = [f for f in features if f in df.columns]
print("\nFeatures used:", features)

if len(features) == 0:
    raise ValueError("None of the selected features exist in the dataset.")

# Select features and drop missing values
X = df[features].dropna()
print("\nSelected features sample:")
print(X.head())

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels
X["Cluster"] = clusters
print("\nCluster Assignments:")
print(X.head(10))

# Cluster centers (original scale)
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers, columns=features)
print("\nCluster Centers:")
print(centers_df)

# Cluster summary statistics
cluster_summary = X.groupby("Cluster").mean()
print("\nCluster Summary Statistics:")
print(cluster_summary)

# Ensure Charts folder exists (for plots)
charts_dir = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System (3)\Placement-Prediction-System\App\static\Charts"
os.makedirs(charts_dir, exist_ok=True)

# Ensure Data folder exists (for CSV)
data_dir = r"C:\Users\asolo\Documents\ML\Placement-Prediction-System (3)\Placement-Prediction-System\data"
os.makedirs(data_dir, exist_ok=True)

# Visualization (CGPA vs CodingTestScore)
if "CGPA" in features and "CodingTestScore" in features:
    plt.figure(figsize=(8,6))
    plt.scatter(X["CGPA"], X["CodingTestScore"], c=X["Cluster"], cmap="viridis", s=50)
    plt.xlabel("CGPA")
    plt.ylabel("Coding Test Score")
    plt.title("K-Means Clusters (CGPA vs CodingTestScore)")
    plt.tight_layout()
    chart_path = os.path.join(charts_dir, "kmeans_clusters.png")
    plt.savefig(chart_path, dpi=300)
    print(f"\nCluster plot saved at: {chart_path}")
    plt.show()

# Save clustered data into DATA folder
csv_path = os.path.join(data_dir, "clustered_data.csv")
X.to_csv(csv_path, index=False)
print(f"\nClustered data saved successfully at: {csv_path}")
