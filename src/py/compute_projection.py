import os
import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, trustworthiness
from sklearn.preprocessing import StandardScaler

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', '..', 'dataset', 'wine.csv')
INPUT_JSON = os.path.join(BASE_DIR, '..', 'json', 'step1_results.json')
OUTPUT_JSON = os.path.join(BASE_DIR, '..', 'json', 'step2_final_data.json')

def run_projections():
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    X = df.iloc[:, 1:].values
    
    print("Standardizing features for unbiased projection...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Executing PCA (2 components)...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    print("Executing Euclidean MDS (2 components)...")
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto')
    mds_coords = mds.fit_transform(X_scaled)

    print(f"Loading relational metrics from: {INPUT_JSON}")
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Missing input! Please run compute_relationship.py first.")
        
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    # --- CALCULATE TRUSTWORTHINESS & CONTINUITY ---
    print("Calculating Global Trustworthiness and Continuity...")
    k = data["metadata"]["k_neighbors_used"]
    k_val = int(k) if str(k).isdigit() else 15

    # Trustworthiness: measures False Positives (HD is ground truth)
    pca_trust = trustworthiness(X_scaled, pca_coords, n_neighbors=k_val)
    mds_trust = trustworthiness(X_scaled, mds_coords, n_neighbors=k_val)
    
    # Continuity: measures False Negatives (Swapping inputs computes continuity)
    pca_cont = trustworthiness(pca_coords, X_scaled, n_neighbors=k_val)
    mds_cont = trustworthiness(mds_coords, X_scaled, n_neighbors=k_val)

    # Inject static global analytics
    data["metadata"]["global_assessment"]["pca"] = {
        "trustworthiness": round(pca_trust, 4),
        "continuity": round(pca_cont, 4)
    }
    data["metadata"]["global_assessment"]["mds"] = {
        "trustworthiness": round(mds_trust, 4),
        "continuity": round(mds_cont, 4)
    }

    # Inject coordinates
    print("Fusing projections with relational metrics...")
    for i, point in enumerate(data["points"]):
        point["pca_x"] = round(float(pca_coords[i, 0]), 4)
        point["pca_y"] = round(float(pca_coords[i, 1]), 4)
        point["mds_x"] = round(float(mds_coords[i, 0]), 4)
        point["mds_y"] = round(float(mds_coords[i, 1]), 4)

    data["metadata"]["projections_included"] = ["PCA", "Euclidean MDS"]

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Success! Final visual analytics payload saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    run_projections()