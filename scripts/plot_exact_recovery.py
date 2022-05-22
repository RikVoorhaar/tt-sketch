# %%
from itertools import product
import numpy as np
from tqdm import tqdm
from tt_sketch.tensor_train_sketching_matrix import TensorTrainDRM
from tt_sketch.tensor import dense_to_sparse
from tt_sketch.lazy_gaussian_sketching_matrix import GaussianDRM
from tt_sketch.sketched_tensor_train import (
    SketchedTensorTrain,
)
from tt_sketch.tt_svd import tt_svd
from tt_sketch.recursive_big_sketch import recursive_big_sketch
from tt_sketch.tensor import TensorTrain

# Shape of problem
n_dim = 4
dim = 20
n_samples = 1
shape = (dim,) * n_dim
seed = 179
print(f"{shape=}, total dim={np.prod(shape):.5e}")

# Store results in dictionary
results = dict()
rank_range = list(range(2, dim))
i_range = np.arange(n_samples)
methods = ["classical", "gauss-gauss", "tt-gauss", "tt-tt", "recursive"]
for name in methods:
    results[name] = np.ones((np.max(rank_range) + 1, len(i_range))) * np.nan


# Construct the dense tensor
X = TensorTrain.random(shape, 10)
X = X.to_numpy()

for r, i in tqdm(list(product(rank_range, i_range))):
    tt = tt_svd(X, r)
    error_tt = np.linalg.norm(tt.to_numpy() - X)
    results["classical"][r, i] = error_tt

    X_sparse = dense_to_sparse(X)

    left_rank = (r,) * (len(shape) - 1)
    right_rank = (r * 2 + 1,) * (len(shape) - 1)
    tt_sketched = stream_sketch(
        X_sparse,
        left_rank,
        right_rank,
        left_sketch_type=GaussianDRM,
        right_sketch_type=GaussianDRM,
    )
    tt2 = TensorTrain(tt_sketched.C_cores)
    error = np.linalg.norm(tt2.to_numpy() - X)
    results["gauss-gauss"][r, i] = error

    tt_sketched = stream_sketch(
        X_sparse,
        left_rank,
        right_rank,
        left_sketch_type=TensorTrainDRM,
        right_sketch_type=GaussianDRM,
    )
    tt2 = TensorTrain(tt_sketched.C_cores)
    error = np.linalg.norm(tt2.to_numpy() - X)
    results["tt-gauss"][r, i] = error

    tt_sketched = stream_sketch(
        X_sparse,
        left_rank,
        right_rank,
        left_sketch_type=TensorTrainDRM,
        right_sketch_type=TensorTrainDRM,
    )
    tt2 = TensorTrain(tt_sketched.C_cores)
    error = np.linalg.norm(tt2.to_numpy() - X)
    results["tt-tt"][r, i] = error

    cores = recursive_big_sketch(X, left_rank, right_rank)
    tt3 = TensorTrain(cores)
    error = np.linalg.norm(tt3.to_numpy() - X)
    results["recursive"][r, i] = error
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i, name in enumerate(methods):
    A = results[name]
    A = A[min(rank_range) :, :]
    A_medians = np.median(A, axis=1)
    A_20 = np.percentile(A, 20, axis=1)
    A_80 = np.percentile(A, 80, axis=1)
    plt.fill_between(rank_range, A_20, A_80, alpha=0.2)
    plt.plot(rank_range, A_medians, "-o", label=name)
plt.ylabel("Error (l2)")
plt.xlabel("tt-rank (uniform)")
plt.title(
    f"Big sketch (oversampling=2x) reconstruction error on {shape} random TT of rank 10"
)
plt.legend()
plt.yscale("log")
plt.savefig("exact_recovery.pdf", bbox_inches="tight")
