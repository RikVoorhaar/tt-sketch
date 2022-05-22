# %%
import numpy as np
from tt_sketch.drm import SparseGaussianDRM
from tt_sketch.drm import DenseGaussianDRM
from tt_sketch.sketching_methods import tensor_train_sketch, dense_sketch
from tt_sketch.drm import (
    TensorTrainDRM,
)
from tt_sketch.sketched_tensor_train import SketchedTensorTrain
from tt_sketch.tt_svd import tt_svd
from tt_sketch.tensor import TensorTrain, DenseTensor
from tt_sketch.utils import matricize

n_dim = 4
dim = 20
shape = tuple(range(dim, dim + n_dim))
rank = [6, 7, 8]
tt = TensorTrain.random(shape, rank)

left_rank = [6, 7, 8]
right_rank = [9, 10, 11]
sktt = stream_sketch(
    tt,
    left_rank,
    right_rank,
    left_sketch_type=DenseGaussianDRM,
    right_sketch_type=DenseGaussianDRM,
)
tt_sketched = TensorTrain(sktt.C_cores)

X0 = tt.to_numpy()
X1 = tt_sketched.to_numpy()
np.linalg.norm(X0 - X1) / np.linalg.norm(X0)
# %%
from tt_sketch.sketched_tensor_train import tt_cores_from_sketch

X_dense = tt.dense()
left_sketch = DenseGaussianDRM(left_rank, shape, transpose=False)
right_sketch = DenseGaussianDRM(right_rank, shape, transpose=True)

phi_mats = left_sketch.sketch_dense(X_dense)
omega_mats = right_sketch.sketch_dense(X_dense)

print([M.shape for M in phi_mats])
print([M.shape for M in omega_mats])
X = X_dense.data
n_dims = len(X.shape)
mu = 2
X_tens = matricize(X, range(mu + 1), mat_shape=False)
X_tens = X_tens.reshape(
    np.prod(X_tens.shape[:mu], dtype=int),
    X_tens.shape[mu],
    X_tens.shape[mu + 1],
)

Psi_cores, Omega_mats = dense_sketch(X_dense, left_sketch, right_sketch)
print("Psi_cores:", [Y.shape for Y in Psi_cores])
print("Omega_mats:", [Z.shape for Z in Omega_mats])
C_cores = tt_cores_from_sketch(Psi_cores, Omega_mats)
tt_sketched = TensorTrain(C_cores)
np.linalg.norm(tt_sketched.to_numpy() - X)
# %%
X_dense.data
tuple(range(len(X_dense.shape) - 1, -1, -1))
reversed(range(len(X_dense.shape)))
# %%
from tt_sketch.utils import right_mul_pinv, matricize
import scipy.linalg

m, n, r = (10, 20, 122)
Y = np.random.normal(size=(m, n))
Z = np.random.normal(size=(r, n))
YZ = right_mul_pinv(Y, Z)
lstsq = scipy.linalg.lstsq(Z.T, Y.T, cond=1e-12)
np.linalg.norm(YZ - lstsq[0].T)
# %%
tt_sparse = tt.dense().to_sparse()
sktt = stream_sketch(
    tt_sparse,
    left_rank,
    right_rank,
    left_sketch_type=DenseGaussianDRM,
    right_sketch_type=DenseGaussianDRM,
)

tt_sketched = TensorTrain(sktt.C_cores)

X0 = tt.to_numpy()
X1 = tt_sketched.to_numpy()
np.linalg.norm(X0 - X1) / np.linalg.norm(X0)
# %%
from tt_sketch.drm.dense_gaussian_drm import DenseGaussianDRM

dg = DenseGaussianDRM(right_rank, shape, transpose=False)
mu = 1
partial_contracts = tt.partial_dense()
[C.shape for C in partial_contracts], [sm.shape for sm in dg.sketching_mats]
sketch = [(sm @ C).T for sm, C in zip(dg.sketching_mats, partial_contracts)]
[s.shape for s in sketch]
# %%
ttdrm = TensorTrainDRM(right_rank, shape, transpose=False)
sparse_sketch_vecs = ttdrm.sketch_tt(tt)
[s.shape for s in sparse_sketch_vecs]
# %%
tuple(range(5, 10)[::-1])
# %%
left_sm = TensorTrainDRM(left_rank, shape, is_left_sketch=True)
right_sm = TensorTrainDRM(right_rank, shape, is_left_sketch=False)
Psi_cores, Omega_mats = tensor_train_sketch(tt, left_sm, right_sm)
# %%
[Y.shape for Y in Psi_cores], [Z.shape for Z in Omega_mats]
# %%

left_contracts = left_sm.sketch_tt(tt)
right_contracts = right_sm.sketch_tt(tt)
[C.shape for C in left_contracts], [C.shape for C in right_contracts]
# %%
[C.shape for C in tt.T.cores]

# %%
