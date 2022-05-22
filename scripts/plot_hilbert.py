# %%
from itertools import product
from functools import reduce
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tt_sketch.drm import (
    DenseGaussianDRM,
    TensorTrainDRM,
)

from experiment_base import (
    Experiment,
    experiment_recursive_sketch,
    experiment_tensor_sketch,
    experiment_tt_svd,
)


def hilbert_tensor(n_dims: int, size: int) -> DenseTensor:
    grid = np.meshgrid(*([np.arange(size)] * n_dims))
    hilbert = 1 / (np.sum(np.array(grid), axis=0) + 1)
    return DenseTensor((size,) * n_dims, hilbert)


size = 5
n_dims = 7
hilbert = hilbert_tensor(n_dims, size).to_sparse()
experiment = Experiment("results/hilbert.csv")

# %%
ranks = range(1, 16)
sketch_types = [
    DenseGaussianDRM,
    # SparseGaussianDRM,
    # SparseSignDRM,
    TensorTrainDRM,
]
runs = range(10)

for rank, run in tqdm(list(product(ranks, runs)), desc="Orthogonalized sketch"):
    experiment.do_experiment(
        hilbert,
        "recursive_sketch",
        experiment_recursive_sketch,
        left_rank=rank,
        right_rank=2 * rank,
        run=run,
    )


for rank, sketch_type, run in tqdm(
    list(product(ranks, sketch_types, runs)), desc="STTA"
):
    experiment.do_experiment(
        hilbert,
        "sketched_dense",
        experiment_tensor_sketch,
        left_rank=rank,
        right_rank=rank * 2,
        left_sketch_type=sketch_type,
        run=run,
    )

for rank in tqdm(ranks, desc="tt_svd"):
    experiment.do_experiment(hilbert, "tt_svd", experiment_tt_svd, rank=rank)


# %%
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(8, 4))

df = pd.read_csv("results/hilbert.csv")
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

plt.xticks(ranks)
plt.ylabel("L2-error")
plt.xlabel("TT-rank")
plt.yscale("log")
plt.legend()
plt.title("Approximation of Hilbert tensor")
plt.savefig("results/plot-hilbert.pdf", transparent=True, bbox_inches="tight")
plt.show()

# %%
