# %%
"""There seem to be problems with consistent seeding; need to fix."""
from tests.test_sketching_matrix import test_tt_cores_contraction
from tt_sketch.sketch import hmt_sketch, orthogonal_sketch, stream_sketch
import numpy as np
from tt_sketch.tensor import (
    CPTensor,
    DenseTensor,
    SparseTensor,
    TensorTrain,
    TuckerTensor,
)
from tt_sketch.drm import TensorTrainDRM


import sys

import numpy as np
import pytest

import itertools
from tt_sketch.drm import (
    ALL_DRM,
    SparseGaussianDRM,
    TensorTrainDRM,
    DenseGaussianDRM,
)
from tt_sketch.drm_base import CanSlice, CanIncreaseRank
from tt_sketch.sketching_methods.abstract_methods import (
    CansketchSparse,
    CansketchTT,
    CansketchCP,
)
from tt_sketch.tensor import (
    SparseTensor,
    TensorTrain,
    CPTensor,
    TensorSum,
    DenseTensor,
    TuckerTensor,
)
from tt_sketch.sketch_dispatch import SketchMethod
from tt_sketch.sketch import (
    stream_sketch,
    blocked_stream_sketch,
    orthogonal_sketch,
    hmt_sketch,
)


# %%
shape = (2, 3, 2)
rank = 4
tensor = SparseTensor.random(shape=shape, nnz=3)
tts = stream_sketch(
    tensor,
    left_rank=3,
    right_rank=6,
    left_drm_type=SparseGaussianDRM,
    right_drm_type=SparseGaussianDRM,
)
tts.increase_rank(tensor, new_left_rank=5, new_right_rank=8)

# %%
n_dims = 3
rank = 2

seed = 180
X_shape = tuple(range(9, 9 + n_dims))
X_tt = TensorTrain.random(X_shape, rank)
X = X_tt.to_numpy()

left_rank = tuple(range(rank, rank + n_dims - 1))
right_rank = tuple(range(rank + 1, rank + n_dims))

X_sparse = X_tt.dense().to_sparse()

d = X_tt.ndim
left_drm_type = DenseGaussianDRM
right_drm_type = DenseGaussianDRM
left_drm = left_drm_type(left_rank, transpose=False, shape=X_shape, seed=seed)
right_seed = np.mod(seed + hash(str(d)), 2**32)
right_drm = right_drm_type(
    right_rank, transpose=True, shape=X_shape, seed=right_seed
)
tts = stream_sketch(
    X_sparse,
    left_rank=left_rank,
    right_rank=right_rank,
    left_drm=left_drm,
    right_drm=right_drm,
)
new_left_rank = tuple(r + 2 for r in left_rank)
new_right_rank = tuple(r + 3 for r in right_rank)
tts2 = tts.increase_rank(X_sparse, new_left_rank, new_right_rank)

tts2.error(X_sparse)
# %%

drm = DenseGaussianDRM((7, 5), transpose=True, shape=X_shape, seed=seed)
print(drm.rank, drm.true_rank)
drm_s = drm.slice((4, 2), (7, 5))
print(drm_s.rank, drm_s.true_rank)

# %%
nnz = 100
shape = (10, 10, 10, 10)
sparse_tensor = SparseTensor.random(shape, nnz)
sparse_tensor.entries *= np.logspace(0, -50, nnz)  # make entries decay fast

tt_sketched = stream_sketch(sparse_tensor, left_rank=10, right_rank=15)
other_sparse_tensor = SparseTensor.random(shape, 10) * 1e-6
sparse_tensor_sum = sparse_tensor + other_sparse_tensor

# Updating an existing sketch
tt_sketched_updated = tt_sketched + other_sparse_tensor
print(tt_sketched_updated.error(sparse_tensor_sum, relative=True))

# Sketching the sum of two tensors directly
tt_sketched2 = stream_sketch(sparse_tensor_sum, left_rank=10, right_rank=15)
print(tt_sketched2.error(sparse_tensor_sum, relative=True))

# %%




# %%
n_dims = 4
rank = tuple(range(3, 3 + n_dims - 1))[::-1]
rank = (3, 2, 5)
left_rank = tuple(r + 2 for r in rank)
right_rank = tuple(r + 1 for r in rank)
X_shape = tuple(range(9, 9 + n_dims))
X_tt = TensorTrain.random(X_shape, rank)
tts1 = stream_sketch(X_tt, left_rank=left_rank, right_rank=right_rank)
tt1 = tts1.to_tt()

tts2 = tts1 * 2
tt2 = tts2.to_tt()
assert tt2.error(tt1 * 2) < 1e-10

tts3 = tts1.T
tt3 = tts3.to_tt()
print()

[O.shape for O in tts3.sketch_.Omega_mats], [O.shape for O in tts1.Omega_mats]
tts3.shape, tts3.sketch_.shape
tts3.left_rank, tts3.sketch_.left_rank, tts3.right_rank, tts3.sketch_.right_rank, left_rank, right_rank
# tts1.left_rank, tts1.sketch_.left_rank, tts1.right_rank, tts1.sketch_.right_rank

# %%

sparse = SparseTensor.random(tt1.shape, 10)
tt_entries = tt1.gather(sparse.indices)
np.dot(tt_entries, sparse.entries), tt1.dot(sparse)

# %%
indices = np.ravel_multi_index(sparse.indices, sparse.shape)

# %%
sparse.dot(sparse), np.dot(sparse.entries, sparse.entries), sparse.norm()**2