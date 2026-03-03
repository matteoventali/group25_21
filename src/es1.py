import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.cluster import KMeans             # Added from Prof's script
from sklearn.metrics import adjusted_rand_score # Added from Prof's script
from sklearn.metrics import euclidean_distances

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Output file for D3.js
OUTPUT_FILE = os.path.join(SCRIPT_DIR, '../dataset/wine_data.json')
# Input CSV file
CSV_FILENAME = os.path.join(SCRIPT_DIR, '../dataset/wine.csv') 
# Target column for Ground Truth (Producer)
TARGET_COLUMN = 'Producer' 

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
print(f"Loading {CSV_FILENAME}...")
try:
    df = pd.read_csv(CSV_FILENAME)
except FileNotFoundError:
    print("ERROR: File not found.")
    exit()

df = df.dropna()

# Separation of Target (True Labels) vs Data
if TARGET_COLUMN and TARGET_COLUMN in df.columns:
    # Save true labels for visualization and ARI calculation
    ground_truth = df[TARGET_COLUMN].values
    class_names = ["Vino " + str(x) for x in ground_truth] # Formatting for tooltip
    # Drop target from numerical data
    data_df = df.drop(columns=[TARGET_COLUMN])
else:
    print("Warning: Target column not found. Clustering comparison (ARI) will be skipped.")
    ground_truth = None
    class_names = ["Item " + str(i) for i in range(len(df))]
    data_df = df

# Select only numerical features
X_original = data_df.select_dtypes(include=[np.number])
X = X_original.values

print(f"Data shape: {X.shape}")

# StandardScaler (Common to both scripts)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. K-MEANS & ARI (From Professor's Script)
# ==========================================
print("Computing K-Means (k=3)...")
# Using same parameters as professor: init='k-means++', n_init=20
km = KMeans(n_clusters=3, init="k-means++", n_init=20, random_state=0)
kmeans_labels = km.fit_predict(X_scaled)

# Calculate Adjusted Rand Score if ground truth exists
if ground_truth is not None:
    ari = adjusted_rand_score(ground_truth, kmeans_labels)
    print(f"\n>>> Adjusted Rand Score (K-Means vs True): {ari:.4f} <<<\n")

# ==========================================
# 4. DIMENSIONALITY REDUCTION (PCA & MDS)
# ==========================================
print("Computing PCA...")
# Added random_state=0 as per professor's script
pca = PCA(n_components=2, random_state=0)
X_pca_raw = pca.fit_transform(X_scaled)

print("Computing MDS...")
mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0, normalized_stress='auto', n_init=4)
X_mds_raw = mds.fit_transform(X_scaled)

# --- COORDINATE NORMALIZATION (From Professor's Script) ---
# The professor normalizes the Projected Coordinates to [0, 1].
# We apply this to both PCA and MDS to make them visually comparable.
coord_scaler = MinMaxScaler()
X_pca = coord_scaler.fit_transform(X_pca_raw)
X_mds = coord_scaler.fit_transform(X_mds_raw)

# ==========================================
# 5. DISTORTION METRICS (Our Logic)
# ==========================================
# Note: We calculate distortion based on Normalized Distances to be scale-invariant.
print("Computing distortion scores (Stretched vs Compressed)...")

# 1. High-Dimensional Distances (Original Scaled Data)
D_high = euclidean_distances(X_scaled)

# 2. Low-Dimensional Distances (Using the MinMax Scaled Projections)
D_pca = euclidean_distances(X_pca)
D_mds = euclidean_distances(X_mds)

# 3. Normalize all distance matrices to [0, 1] for fair comparison
dist_scaler = MinMaxScaler()
D_high_norm = dist_scaler.fit_transform(D_high)
D_pca_norm = dist_scaler.fit_transform(D_pca)
D_mds_norm = dist_scaler.fit_transform(D_mds)

# 4. Calculate Error (LowDim - HighDim)
# Negative = Compressed (Blue), Positive = Stretched (Red)
E_pca = D_pca_norm - D_high_norm
E_mds = D_mds_norm - D_high_norm

score_pca = np.mean(E_pca, axis=1)
score_mds = np.mean(E_mds, axis=1)

# ==========================================
# 6. JSON EXPORT
# ==========================================
data_export = []
for i in range(len(X)):
    data_export.append({
        "id": i,
        # Coordinates are now [0,1] like the professor's CSV
        "pca_x": float(X_pca[i, 0]),
        "pca_y": float(X_pca[i, 1]),
        "mds_x": float(X_mds[i, 0]),
        "mds_y": float(X_mds[i, 1]),
        # Distortion scores for coloring
        "score_pca": float(score_pca[i]),
        "score_mds": float(score_mds[i]),
        # Info for Tooltip
        "class_name": class_names[i],       # True Label (Producer)
        "kmeans_label": int(kmeans_labels[i]) # Calculated K-Means Label
    })

with open(OUTPUT_FILE, 'w') as f:
    json.dump(data_export, f)

print(f"Done! JSON file saved to: {OUTPUT_FILE}")