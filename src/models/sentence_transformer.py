# pip install sentence-transformers numpy scipy
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian

# 1) Embed your feature phrases (expand cryptic names first if needed)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # fast & good
phrases = [
    "chest pain type 2 vs type 1 baseline",
    "maximum heart rate achieved",
    "resting ECG shows left ventricular hypertrophy",
    "fasting blood sugar > 120 mg/dL",
    "exercise-induced angina (yes/no)"
]
E = model.encode(phrases, normalize_embeddings=True)   # shape: (p, d), L2-normalized rows
p = E.shape[0]

# 2) Build a PSD kernel: K = E E^T   (guaranteed PSD if rows are vectors)
K = E @ E.T               # shape: (p, p)
K = 0.5 * (K + K.T)       # symmetrize for safety

# 2a) (Optional) Normalize scale so gamma has a sane range
tr = np.trace(K)
if tr > 0:
    K = K * (p / tr)      # now trace(K) ~ p

# 2b) (Optional) Sparsify to top-k neighbors per row to avoid over-smoothing
def topk_sparsify(K_dense, k=15):
    p = K_dense.shape[0]
    rows, cols, data = [], [], []
    for i in range(p):
        # ignore self (i), keep top-k others
        row = K_dense[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(row, -k)[-k:]
        idx = idx[np.argsort(row[idx])[::-1]]  # sort descending
        rows.extend([i]*len(idx))
        cols.extend(idx.tolist())
        data.extend(K_dense[i, idx].tolist())
    # add symmetric edges and self-diagonal to keep PSD-ish behavior
    # (strict PSD isn’t guaranteed after sparsify, but this works well in practice)
    S = csr_matrix((data, (rows, cols)), shape=K_dense.shape)
    S = 0.5 * (S + S.T)
    S = S + csr_matrix(np.eye(p)) * 1e-6  # tiny diagonal lift
    return S

K_sparse = topk_sparsify(K, k=3)  # small k for this demo

# 3) Quick sanity checks
print("K (dense) shape:", K.shape)
print("K (dense) trace:", float(np.trace(K)))
print("K (sparse) nnz :", K_sparse.nnz)

# 4) (Mini test) Smooth a noisy target vector t over the feature graph
#    This shows how K can stabilize noisy LLM targets BEFORE you use them.
rng = np.random.default_rng(0)
t_raw = rng.normal(size=p)                  # pretend this came from an LLM
L = sparse_laplacian(K_sparse, normed=False)  # graph Laplacian from sparsified kernel
rho = 0.1                                   # smoothing strength (tune by CV in practice)

# Solve: t_smooth = argmin_u ||u - t_raw||^2 + rho * u^T L u
# => (I + rho * L) u = t_raw
from scipy.sparse.linalg import spsolve
I = csr_matrix(np.eye(p))
t_smooth = spsolve(I + rho * L, t_raw)

print("\nRaw targets:   ", np.round(t_raw, 3))
print("Smoothed t:    ", np.round(t_smooth, 3))

# 5) (If you’ve added SKR to your model)
#    Pass `kernel=K` (or a dense K from K_sparse.toarray()) and a gamma>0 into your model,
#    and cross-validate gamma together with your existing eta.
