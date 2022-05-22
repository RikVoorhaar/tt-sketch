# %%
from itertools import permutations, product

import numpy as np
from tqdm import tqdm
from tt_sketch.drm import DenseGaussianDRM, TensorTrainDRM
from tt_sketch.tensor import DenseTensor
from experiment_base import (
    Experiment,
    experiment_recursive_sketch,
    experiment_tensor_sketch,
    experiment_tt_svd,
)

# Shape of problem
n_dim = 5
dim = 10
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
def f(X):
    return np.sqrt(np.abs(np.sum(X, axis=0)))


vals = [np.linspace(-0.2, 2, s) for s in shape]
grid = np.stack(np.meshgrid(*vals))
X = f(grid)
X /= np.linalg.norm(X)

tensor = DenseTensor(X.shape, X).to_sparse()
experiment = Experiment("results/sum-sqrt.csv")
# %%
ranks = range(1, 16)
sketch_types = [
    DenseGaussianDRM,
    TensorTrainDRM,
]
runs = range(10)

for rank, run in tqdm(list(product(ranks, runs)), desc="recursive big-sketch"):
    experiment.do_experiment(
        tensor,
        "recursive_sketch",
        experiment_recursive_sketch,
        left_rank=rank,
        right_rank=2*rank,
        run=run,
    )


for rank, sketch_type, run in tqdm(
    list(product(ranks, sketch_types, runs)), desc="Sparse-sketch"
):
    experiment.do_experiment(
        tensor,
        "sketched_dense",
        experiment_tensor_sketch,
        left_rank=rank,
        right_rank=rank * 2,
        left_sketch_type=sketch_type,
        run=run,
    )

for rank in tqdm(ranks, desc="tt_svd"):
    experiment.do_experiment(tensor, "tt_svd", experiment_tt_svd, rank=rank)

# %%
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(8, 4))

df = pd.read_csv("results/sum-sqrt.csv")
rsketch = df[df["name"] == "recursive_sketch"]
ranks = rsketch["left_rank"].unique()

ttsvd = df[df["name"] == "tt_svd"]
plt.plot(ranks, ttsvd.error.values, "o", label="TTSVD")

error_gb = rsketch.groupby(rsketch["left_rank"]).error
errors08 = error_gb.quantile(0.8).values
errors05 = error_gb.quantile(0.5).values
errors02 = error_gb.quantile(0.2).values
# plt.fill_between(ranks, errors02, errors08, alpha=0.3)
plt.errorbar(
    ranks,
    errors05,
    yerr=np.stack([errors02, errors08]),
    label="OTTS",
    capsize=3,
    linestyle="",
)


ssketch = df[df["name"] == "sketched_dense"]
drms = {
    "DenseGaussianDRM": "STTA, Gaussian DRM",
    "TensorTrainDRM": "STTA, TT-DRM",
}
for drm, drm_name in drms.items():
    error_gb = (
        ssketch[ssketch["left_sketch_type"] == drm].groupby("left_rank").error
    )
    errors08 = error_gb.quantile(0.8).values
    errors05 = error_gb.quantile(0.5).values
    errors02 = error_gb.quantile(0.2).values
    plt.errorbar(
        ranks,
        errors05,
        yerr=np.stack([errors02, errors08]),
        label=drm_name,
        capsize=3,
        linestyle="",
    )

    # plt.plot(ranks, errors05, "-o", label=drm_name)

plt.xticks(ranks)
plt.ylabel("L2-error")
plt.xlabel("TT-rank")
plt.yscale("log")
plt.title(
    f"Approximation of square-root-sum tensor"
)
plt.legend()
plt.savefig("results/plot-sum-sqrt.pdf", transparent=True, bbox_inches="tight")
plt.show()
