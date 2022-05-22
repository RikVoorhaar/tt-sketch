# %%
from typing import Dict, List, Tuple

import numpy as np
from tt_sketch.drm import (
    ALL_DRM,
    DenseGaussianDRM,
    SparseGaussianDRM,
    SparseSignDRM,
    TensorTrainDRM,
)
from tt_sketch.drm_base import DRM
from tt_sketch.sketched_tensor_train import (
    SketchedTensorTrain,
    get_drm_capabilities,
)
from tt_sketch.sketching_methods import (
    CansketchSparse,
    CansketchTT,
    tensor_train_sketch,
    sparse_sketch,
    sparse_sketch_parallel,
)
from tt_sketch.tensor import SparseTensor, Tensor, TensorTrain
from tt_sketch.utils import ArrayList

sparse_drm_list = [
    drm_type for drm_type in ALL_DRM if issubclass(drm_type, CansketchSparse)
]
import pandas as pd

import sys

sys.path.append("../tests")
from test_sketching_matrix import (
    general_blocked_sketch,
    general_rank_increase,
    test_exact_recovery_tt,
    test_exact_recovery_sparse,
)

# %%
# mprun -f tensor_train_sketch
# left_drm_type = TensorTrainDRM
# right_drm_type = DenseGaussianDRM
# rank = 4
# n_dims = 6
# seed = 180
# X_shape = tuple(range(7, 7 + n_dims))
# X_tt = TensorTrain.random(X_shape, rank)

# left_rank = tuple(range(rank, rank + n_dims - 1))
# right_rank = tuple(range(rank + 1, rank + n_dims))

# left_drm = left_drm_type(left_rank, X_shape, transpose=False, seed=179)
# right_drm = right_drm_type(right_rank, X_shape, transpose=True, seed=12)
# tensor_train_sketch(X_tt, left_drm, right_drm)

# # %%
# """Let's start of with some memory profiling. How do we do that in Python?"""

# pd.DataFrame(get_drm_capabilities())

# test_exact_recovery_tt(2, 2, "TensorTrainDRM|TensorTrainDRM")

# # %%
left_drm_type = TensorTrainDRM
right_drm_type = DenseGaussianDRM
rank = 4
n_dims = 6
seed = 180
X_shape = tuple(range(10, 10 + n_dims))
X_tt = TensorTrain.random(X_shape, rank)
X_dense = X_tt.to_numpy()

left_rank = tuple(range(rank, rank + n_dims - 1))
right_rank = tuple(range(rank + 1, rank + n_dims))

X_sparse = X_tt.dense().to_sparse()

left_drm = left_drm_type(left_rank, X_shape, transpose=False, seed=179)
right_drm = right_drm_type(right_rank, X_shape, transpose=True, seed=12)
print("sketch_size: ", sum([s.size for s in left_drm.sketch_sparse(X_sparse)]))

memory_footprint_per_index = (sum(left_rank) + sum(right_rank)) * 8
memory_footprint_per_index * X_sparse.nnz

# %time sparse_sketch_parallel(X_sparse, left_drm, right_drm,max_mem_size_MB=10,DEBUG_COPY=True)
sparse_sketch_parallel(X_sparse, left_drm, right_drm, max_mem_size_MB=10)
Psi_cores2, Omega_mats2 = sparse_sketch(X_sparse, left_drm, right_drm)
# [np.linalg.norm(Y1-Y2) for Y1,Y2 in zip(Psi_cores1, Psi_cores2)]
# [np.linalg.norm(Z1-Z2) for Z1,Z2 in zip(Omega_mats1, Omega_mats2)]

# %%
# %%
# %%
# %%
Psi_cores, Omega_mats = sparse_sketch(X_sparse, left_drm, right_drm)

Y_list = [Psi_cores[0] for _ in range(10)]
np.sum(Y_list, axis=0).shape, Psi_cores[0].shape

# %%
shape = X_sparse.shape
shape

# %%
left_prods = []
prod = 1
for n in shape:
    left_prods = 1