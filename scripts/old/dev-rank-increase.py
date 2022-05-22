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
from tt_sketch.sketching_methods import CansketchSparse, CansketchTT
from tt_sketch.tensor import SparseTensor, Tensor, TensorTrain
from tt_sketch.utils import ArrayList

sparse_drm_list = [
    drm_type for drm_type in ALL_DRM if issubclass(drm_type, CansketchSparse)
]
import pandas as pd

import sys

sys.path.append("../tests")
from test_sketching_matrix import general_blocked_sketch, general_rank_increase

pd.DataFrame(get_drm_capabilities())


# %%
# sparse_drm_list = [DenseGaussianDRM,SparseGaussianDRM,SparseSignDRM]

left_drm = TensorTrainDRM
right_drm = DenseGaussianDRM
rank = 4
n_dims = 4
seed = 180
X_shape = tuple(range(7, 7 + n_dims))
X_tt = TensorTrain.random(X_shape, rank)
X_dense = X_tt.to_numpy()

left_rank = tuple(range(rank, rank + n_dims - 1))
right_rank = tuple(range(rank + 1, rank + n_dims))

X_sparse = X_tt.dense().to_sparse()

# general_rank_increase(
#     X_dense, X_sparse, left_rank, right_rank, seed, left_drm, right_drm
# )
general_blocked_sketch(X_sparse, seed, left_drm, right_drm)
# %%
DenseGaussianDRM.__name__
# %%
from itertools import chain

chain.from_iterable(zip(range(10), range(10)))
# %%
sparse_drm = right_drm(right_rank, X_shape, transpose=True)
print(f"{sparse_drm.nnz=}")
new_right_rank = tuple(r + 2 for r in right_rank)
sparse_drm2 = sparse_drm.increase_rank(new_right_rank)
print(f"{sparse_drm2.nnz=}")
# %%
np.prod(X_sparse.nnz)
# %%
tt_sketch_top_left = SketchedTensorTrain(
    left_sketch, right_sketch, Psi_cores1, Omega_mats1
)
tt_sketch4 = tt_sketch_top_left.increase_rank(
    X_sparse, new_left_rank, new_right_rank
)
tt_sketch4.Psi_cores[1]
# %%
left_rank, new_left_rank
right_rank, new_right_rank
# %%
for i, (Y1, Y2) in enumerate(zip(tt_sketch4.Psi_cores, Psi_cores2)):
    assert np.allclose(Y1, Y2)

for i, (Z1, Z2) in enumerate(zip(tt_sketch4.Omega_mats, Omega_mats2)):
    assert np.allclose(Z1, Z2)
# %%
[Y.shape for Y in Psi_cores2]
