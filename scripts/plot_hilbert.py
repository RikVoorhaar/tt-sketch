# %%
from itertools import product
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tt_sketch.drm import (
    DenseGaussianDRM,
    TensorTrainDRM,
)
from tt_sketch.tensor import DenseTensor
from tt_sketch.utils import hilbert_tensor

from experiment_base import (
    Experiment,
    experiment_orthogonal_sketch,
    experiment_stream_sketch,
    experiment_tt_svd,
    experiment_hmt_sketch,
)
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


size = 5
n_dims = 7
tensor = DenseTensor(hilbert_tensor(n_dims, size)).to_sparse()
csv_filename = "results/hilbert.csv"
experiment = Experiment(csv_filename)

# %%
ranks = range(1, 16)
drm_types = [
    DenseGaussianDRM,
    TensorTrainDRM,
]
runs = range(20)

for rank, run, drm_type in tqdm(
    list(product(ranks, runs, drm_types)), desc="OTTS"
):
    experiment.do_experiment(
        tensor,
        "OTTS",
        experiment_orthogonal_sketch,
        left_rank=rank,
        right_rank=rank * 2,
        left_drm_type=drm_type,
        right_drm_type=drm_type,
        run=run,
    )


for rank, run, drm_type in tqdm(
    list(product(ranks, runs, drm_types)), desc="STTA"
):
    experiment.do_experiment(
        tensor,
        "STTA",
        experiment_stream_sketch,
        left_rank=rank,
        right_rank=rank * 2,
        left_drm_type=drm_type,
        right_drm_type=drm_type,
        run=run,
    )


for rank, run, drm_type in tqdm(
    list(product(ranks, runs, drm_types)), desc="HMT"
):
    experiment.do_experiment(
        tensor,
        "HMT",
        experiment_hmt_sketch,
        rank=rank,
        drm_type=drm_type,
        run=run,
    )

for rank in tqdm(ranks, desc="TT-SVD"):
    experiment.do_experiment(tensor, "TT-SVD", experiment_tt_svd, rank=rank)


# %%
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(csv_filename)
df
# %%
plt.figure(figsize=(8, 4))
ttsvd = df[df["name"] == "TT-SVD"]

ssketch = df[df["name"] == "OTTS"]
plot_ranks = ssketch["left_rank"].unique()
plt.plot(plot_ranks, ttsvd.error.values, "-o", label="TT-SVD", ms=3)

# drms = {
#     "DenseGaussianDRM": "OTTS, Gaussian DRM",
#     "TensorTrainDRM": "OTTS, TT-DRM",
# }
# for i, (drm, drm_name) in enumerate(drms.items()):
#     error_gb = (
#         ssketch[ssketch["left_drm_type"] == drm].groupby("left_rank").error
#     )
#     errors05 = error_gb.quantile(0.5).values
#     errors08 = error_gb.quantile(0.8).values - errors05
#     errors02 = errors05 - error_gb.quantile(0.2).values
#     plt.errorbar(
#         plot_ranks - 0.05 * (i + 0.5),
#         errors05,
#         yerr=np.stack([errors02, errors08]),
#         label=drm_name,
#         capsize=3,
#         linestyle="",
#     )

ssketch = df[df["name"] == "HMT"]
drms = {
    "DenseGaussianDRM": "TT-HMT, Gaussian DRM",
    "TensorTrainDRM": "TT-HMT, TT-DRM",
}
for i, (drm, drm_name) in enumerate(drms.items()):
    error_gb = ssketch[ssketch["drm_type"] == drm].groupby("rank").error
    errors05 = error_gb.quantile(0.5).values
    errors08 = error_gb.quantile(0.8).values - errors05
    errors02 = errors05 - error_gb.quantile(0.2).values
    plt.errorbar(
        plot_ranks - 0.15 * (1.5 - i),
        errors05,
        yerr=np.stack([errors02, errors08]),
        label=drm_name,
        capsize=3,
        linestyle="",
    )


ssketch = df[df["name"] == "STTA"]
drms = {
    "DenseGaussianDRM": "STTA, Gaussian DRM",
    "TensorTrainDRM": "STTA, TT-DRM",
}
for i, (drm, drm_name) in enumerate(drms.items()):
    error_gb = (
        ssketch[ssketch["left_drm_type"] == drm].groupby("left_rank").error
    )
    errors05 = error_gb.quantile(0.5).values
    errors08 = error_gb.quantile(0.8).values - errors05
    errors02 = errors05 - error_gb.quantile(0.2).values
    plt.errorbar(
        plot_ranks + 0.15 * (i + 0.5),
        errors05,
        yerr=np.stack([errors02, errors08]),
        label=drm_name,
        capsize=3,
        linestyle="",
    )

plt.xticks(ranks)
plt.ylabel("Relative error")
plt.xlabel("TT-rank")
plt.yscale("log")
plt.legend()
plt.title("Approximation of Hilbert tensor")
plt.savefig("results/plot-hilbert.pdf", transparent=True, bbox_inches="tight")
plt.show()

# %%
