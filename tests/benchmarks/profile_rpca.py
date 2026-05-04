import numpy as np
from scipy import sparse
import time

N = 151
r = 15
alpha = 1.0

# Generate random rho
np.random.seed(0)
rho = np.random.rand(N, N, N)

# Generate sparse matrix
S_dense = np.random.rand(N, N)
S_dense[S_dense < 0.95] = 0  # 95% sparsity
S_sp = sparse.csr_matrix(S_dense)

def apply_sparse_axis(S_sp, rho, axis):
    rho_m = np.moveaxis(rho, axis, 0)
    rest = rho_m.shape[1:]
    out = (S_sp @ rho_m.reshape(N, -1)).reshape(N, *rest)
    return np.moveaxis(out, 0, axis)

t0 = time.time()
for _ in range(15):  # 15 terms
    for axis in range(3):
        _ = apply_sparse_axis(S_sp, rho, axis)
t1 = time.time()
print(f"Time for 15 terms (3 axes each): {t1 - t0:.4f} s")
