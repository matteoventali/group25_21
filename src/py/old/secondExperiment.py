import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.datasets import load_wine

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Get the absolute path of the current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths for input dataset and output JSON
OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '../../dataset/firstExperiment.json'))
CSV_FILENAME = os.path.abspath(os.path.join(SCRIPT_DIR, '../../dataset/wine.csv'))

# Hyperparameters for K-NN topology analysis
TARGET_COLUMN = 'Producer'
K_NEIGHBORS = 10 
TOLERANCE = 80  

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
try:
    # Attempt to load local CSV dataset
    df = pd.read_csv(CSV_FILENAME)
    df = df.dropna()
    if TARGET_COLUMN in df.columns:
        class_names = ["Class " + str(c) for c in df[TARGET_COLUMN].values]
        data_df = df.drop(columns=[TARGET_COLUMN])
    else:
        class_names = ["Item " + str(i) for i in range(len(df))]
        data_df = df
except FileNotFoundError:
    # Fallback to Scikit-learn Wine dataset if CSV is missing
    print("Dataset not found. Using sklearn fallback...")
    wine = load_wine()
    data_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    class_names = ["Class " + str(c) for c in wine.target]

# Feature extraction and scaling
X = data_df.select_dtypes(include=[np.number]).values
N = X.shape[0]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. DIMENSIONALITY REDUCTION (PROJECTIONS)
# ==========================================
# Principal Component Analysis (Linear)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_pca[:, 0] = -X_pca[:, 0] # Flip axis for better visualization consistency

# Multidimensional Scaling (Non-linear)
mds = MDS(
    n_components=2, 
    dissimilarity='euclidean', 
    random_state=42, 
    normalized_stress='auto', 
    n_init=4, 
    max_iter=300
)
X_mds = mds.fit_transform(X_scaled)

# ==========================================
# 4. TOPOLOGICAL DISTORTION ANALYSIS (K-NN)
# ==========================================
def calculate_knn_distortion(D_high, D_low, K):
    """
    Calculates False Positives (Intrusions) and False Negatives (Extrusions)
    based on K-NN rank differences between high and low dimensional spaces.
    """
    N = D_high.shape[0]
    # Set diagonal to infinity to ignore self-distance in ranking
    np.fill_diagonal(D_high, np.inf)
    np.fill_diagonal(D_low, np.inf)

    # Convert distances to rank matrices (0 = closest, N-1 = furthest)
    R_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
    R_low = np.argsort(np.argsort(D_low, axis=1), axis=1)

    score_fp = np.zeros(N)
    score_fn = np.zeros(N)

    for i in range(N):
        # Identify K-nearest neighbors in both spaces
        U_i = set(np.where(R_high[i] < K)[0])
        V_i = set(np.where(R_low[i] < K)[0])

        # False Positives: Neighbors in 2D that were far in High-D (Intrusions)
        fp_indices = list(V_i - U_i)
        if fp_indices:
            score_fp[i] = np.sum(R_high[i, fp_indices] - K)

        # False Negatives: Neighbors in High-D that were lost in 2D (Extrusions)
        fn_indices = list(U_i - V_i)
        if fn_indices:
            score_fn[i] = np.sum(R_low[i, fn_indices] - K)

    # Net Score: Positive values = Intrusions (Blue), Negative = Extrusions (Red)
    point_scores = score_fp - score_fn
    total_distortion = np.sum(score_fp) + np.sum(score_fn)
    
    return point_scores, total_distortion

# Compute distance matrices and topological scores
D_high = euclidean_distances(X_scaled)
score_pca, global_pca = calculate_knn_distortion(D_high.copy(), euclidean_distances(X_pca), K_NEIGHBORS)
score_mds, global_mds = calculate_knn_distortion(D_high.copy(), euclidean_distances(X_mds), K_NEIGHBORS)

# Percentage of points within the acceptable topological error tolerance
pca_matches = np.sum(np.abs(score_pca) <= TOLERANCE) / N * 100
mds_matches = np.sum(np.abs(score_mds) <= TOLERANCE) / N * 100

# Identify the top 5 real neighbors in High-D for frontend visualization
neighbors_R13 = np.argsort(D_high, axis=1)[:, 0:5].tolist()

# ==========================================
# 5. DATA EXPORT
# ==========================================
export_data = {
    "stats": {
        "pca_global": int(global_pca),
        "mds_global": int(global_mds),
        "pca_matches": round(pca_matches, 1),
        "mds_matches": round(mds_matches, 1)
    },
    "points": []
}

# Construct JSON structure for each data point
for i in range(N):
    export_data["points"].append({
        "id": i,
        "pca_x": float(X_pca[i, 0]),
        "pca_y": float(X_pca[i, 1]),
        "mds_x": float(X_mds[i, 0]),
        "mds_y": float(X_mds[i, 1]),
        "score_pca": float(score_pca[i]),
        "score_mds": float(score_mds[i]),
        "class_name": class_names[i],
        "neighbors": neighbors_R13[i]
    })

# Ensure directory exists and write JSON file
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(export_data, f, indent=4)

print(f"K-NN Export successfully completed: {OUTPUT_FILE}")