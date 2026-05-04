import numpy as np
from scipy import sparse
import time


def apply_sparse_axis_csr(S_sp, rho, axis):
    rho_m = np.moveaxis(rho, axis, 0)
    rest = rho_m.shape[1:]
    out = (S_sp @ rho_m.reshape(rho_m.shape[0], -1)).reshape(rho_m.shape[0], *rest)
    return np.moveaxis(out, 0, axis)


def apply_sparse_axis_dense(S_dense, rho, axis):
    tmp = np.tensordot(S_dense, rho, axes=([1], [axis]))
    return np.moveaxis(tmp, 0, axis)


def benchmark(N=151, sparsity=0.95, repeats=15, seed=0):
    np.random.seed(seed)
    rho = np.random.rand(N, N, N)

    S_dense = np.random.rand(N, N)
    S_dense[S_dense < sparsity] = 0.0
    S_sp = sparse.csr_matrix(S_dense)

    t0 = time.time()
    for _ in range(repeats):
        for axis in range(3):
            apply_sparse_axis_csr(S_sp, rho, axis)
    t_csr = time.time() - t0

    t0 = time.time()
    for _ in range(repeats):
        for axis in range(3):
            apply_sparse_axis_dense(S_dense, rho, axis)
    t_dense = time.time() - t0

    print(f"CSR reshape: {t_csr:.4f} s")
    print(f"Dense tensordot: {t_dense:.4f} s")


if __name__ == "__main__":
    benchmark()
