import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', '..', 'dataset', 'wine.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'json')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'step1_results.json')

# --- ALGORITHM PARAMETERS ---
EPS = 1e-5
USE_KNN = True
K_NEIGHBORS = 15  # Number of nearest neighbors to consider

def calculate_metrics():
    # 1. Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Dataset
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # Assuming the first column is the label ('producer') and the rest are features
    labels = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    unique_labels = np.unique(labels)
    n_samples = len(X)

    # 3. Standardize Features
    print("Standardizing features for accurate Euclidean distance...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Compute Euclidean Distance Matrix
    print("Calculating distance matrix and attraction weights...")
    dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))

    # 5. Compute Weights: W = 1 / (dist + eps)
    weights = 1.0 / (dist_matrix + EPS)
    np.fill_diagonal(weights, 0) # Remove self-loops

    # 6. Build the K-NN Graph
    if USE_KNN:
        for i in range(n_samples):
            # Keep only the top K highest weights (nearest neighbors)
            idx_to_zero = np.argsort(weights[i])[:-K_NEIGHBORS]
            weights[i, idx_to_zero] = 0.0

    # 7. Calculate Precision, Recall, F-score, and extract neighbors
    print("Extracting neighbors and computing local metrics...")
    points_data = []
    
    for i in range(n_samples):
        current_label = labels[i]
        same_class_mask = (labels == current_label)
        diff_class_mask = (labels != current_label)
        
        # Identify connected neighbors (weight > 0)
        connected_mask = (weights[i] > 0)
        
        # --- EXTRACT NEIGHBORS (CRITICAL FOR FRONTEND) ---
        connected_indices = np.where(connected_mask)[0].tolist()
        if i in connected_indices:
            connected_indices.remove(i)
        
        # --- PRECISION ---
        tp_mask = connected_mask & same_class_mask
        tp_weight = np.sum(weights[i, tp_mask])
        
        fp_mask = connected_mask & diff_class_mask
        fp_weight = np.sum(weights[i, fp_mask])
        
        total_attraction = tp_weight + fp_weight
        precision = (tp_weight / total_attraction) if total_attraction > 0 else 0.0
        
        # --- RECALL ---
        fn_mask = same_class_mask & ~connected_mask
        fn_mask[i] = False # Exclude self
        
        tp_count = np.sum(tp_mask)
        fn_count = np.sum(fn_mask)
        
        total_relevant = tp_count + fn_count
        recall = (tp_count / total_relevant) if total_relevant > 0 else 0.0
        
        # --- F-SCORE ---
        if precision + recall > 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = 0.0
            
        points_data.append({
            "id": i,
            "label": str(current_label),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f_score": round(f_score, 4),
            "neighbors": connected_indices # Injected for frontend line drawing
        })

    # 8. Global F-score Assessment
    class_fscores = []
    for lbl in unique_labels:
        fscores_in_class = [p["f_score"] for p in points_data if p["label"] == str(lbl)]
        if fscores_in_class:
            class_fscores.append(np.mean(fscores_in_class))
            
    global_f_score = np.mean(class_fscores) if class_fscores else 0.0

    output_data = {
        "metadata": {
            "dataset": "wine.csv",
            "k_neighbors_used": K_NEIGHBORS if USE_KNN else "All (Dense)",
            "global_assessment": {
                "global_f_score": round(global_f_score, 4),
                "message": f"{round(global_f_score*100, 1)}% of the expected cluster structure is supported by original relationships."
            }
        },
        "points": points_data
    }

    # 9. Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Success! Relational data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    calculate_metrics()