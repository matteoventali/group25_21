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
# 1. SETUP E CONFIGURAZIONE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '../../dataset/firstExperiment.json'))
CSV_FILENAME = os.path.abspath(os.path.join(SCRIPT_DIR, '../../dataset/wine.csv'))
TARGET_COLUMN = 'Producer'

K_NEIGHBORS = 10 # Il numero di "veri amici" da considerare per i FP/FN

# ==========================================
# 2. CARICAMENTO DATI
# ==========================================
try:
    df = pd.read_csv(CSV_FILENAME)
    df = df.dropna()
    if TARGET_COLUMN in df.columns:
        class_names = ["Class " + c for c in df[TARGET_COLUMN].astype(str).values]
        data_df = df.drop(columns=[TARGET_COLUMN])
    else:
        class_names = ["Item " + str(i) for i in range(len(df))]
        data_df = df
except FileNotFoundError:
    print("Dataset non trovato. Uso fallback...")
    wine = load_wine()
    data_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    class_names = ["Class " + str(c) for c in wine.target]

X = data_df.select_dtypes(include=[np.number]).values
N = X.shape[0]
X_scaled = StandardScaler().fit_transform(X)

# ==========================================
# 3. PROIEZIONI
# ==========================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_pca[:, 0] = -X_pca[:, 0]

mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto', n_init=4, max_iter=300)
X_mds = mds.fit_transform(X_scaled)

# ==========================================
# 4. LA MATEMATICA CORRETTA: RANGHI K-NN
# ==========================================
def calculate_knn_distortion(D_high, D_low, K):
    N = D_high.shape[0]
    np.fill_diagonal(D_high, np.inf)
    np.fill_diagonal(D_low, np.inf)

    # Trasforma le distanze in classifiche (0 = più vicino, N-1 = più lontano)
    R_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
    R_low = np.argsort(np.argsort(D_low, axis=1), axis=1)

    score_fp = np.zeros(N)
    score_fn = np.zeros(N)

    for i in range(N):
        # Chi sono i K vicini reali e quelli apparenti?
        U_i = set(np.where(R_high[i] < K)[0])
        V_i = set(np.where(R_low[i] < K)[0])

        # Falsi Positivi (Intrusioni): Sono tra i K nel 2D, ma non in R13
        fp_indices = list(V_i - U_i)
        if fp_indices:
            score_fp[i] = np.sum(R_high[i, fp_indices] - K)

        # Falsi Negativi (Estrusioni): Erano tra i K in R13, ma sono spariti nel 2D
        fn_indices = list(U_i - V_i)
        if fn_indices:
            score_fn[i] = np.sum(R_low[i, fn_indices] - K)

    # Punteggio Netto: Positivo = FP dominanti (Blu), Negativo = FN dominanti (Rosso)
    point_scores = score_fp - score_fn
    global_score = np.sum(score_fp) + np.sum(score_fn)
    
    return point_scores, global_score

D_high = euclidean_distances(X_scaled)
score_pca, global_pca = calculate_knn_distortion(D_high.copy(), euclidean_distances(X_pca), K_NEIGHBORS)
score_mds, global_mds = calculate_knn_distortion(D_high.copy(), euclidean_distances(X_mds), K_NEIGHBORS)

# Assessment: Punti con 0 errori topologici nei primi K vicini
# Assessment: Punti con un errore topologico accettabile (Tolleranza)
TOLERANCE = 10  # Puoi alzare o abbassare questo valore (es. tra 5 e 15)

pca_matches = np.sum(np.abs(score_pca) <= TOLERANCE) / N * 100
mds_matches = np.sum(np.abs(score_mds) <= TOLERANCE) / N * 100

# Trova i veri vicini in R13 per il frontend (le linee)
neighbors_R13 = np.argsort(D_high, axis=1)[:, 0:5].tolist()

# ==========================================
# 5. ESPORTAZIONE
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

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(export_data, f, indent=4)
print(f"Esportazione K-NN completata in {OUTPUT_FILE}")