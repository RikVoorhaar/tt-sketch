# %%
import dask.array as da
import numpy as np
from scipy.linalg import logm

A = da.random.normal(size=(100, 1000))
B = da.random.normal(size=(100, 1000))
ABt = A @ B.T

res = np.mean(ABt)

from tt_sketch.drm.fast_lazy_gaussian import inds_to_normal
from tt_sketch.tensor import dense_to_sparse, DenseTensor

# %%
# Shape of problem
n_dim = 2
dim = 6
n_samples = 1
shape = (dim,) * n_dim
seed = 179
print(f"{shape=}, total dim={np.prod(shape):.5e}")

# Construct the dense tensor
def f(X):
    return np.sqrt(np.abs(np.sum(X, axis=0)))


vals = [np.linspace(-0.2, 2, s) for s in shape]
grid = np.stack(np.meshgrid(*vals))
X = f(grid)
X /= np.linalg.norm(X)

X_tensor = DenseTensor(X.shape, X)
X_dask = da.from_array(X_tensor, chunks=3)


def block_func(X, block_info=None, block_id=None):
    print("-" * 30)
    print("block_info:")
    print(block_info)
    print("block_id:")
    print(block_id)
    print("x_type:")
    print(type(X.item()))
    print("x")
    print(X.item())
    return X.item().data


m = X_dask.map_blocks(
    block_func, dtype=np.float64, meta=np.array([])
)
m.compute(scheduler="single-threaded")
# %%

# %%