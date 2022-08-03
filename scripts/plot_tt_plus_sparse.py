# %%
"""Experiment of compressing a TT + dense tensor"""
import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from tt_sketch.tensor import TensorTrain, SparseTensor

from experiment_base import (
    Experiment,
    experiment_orthogonal_sketch,
    experiment_stream_sketch,
    experiment_tt_svd,
    experiment_hmt_sketch
)
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

n_dims = 5
dim = 10
shape = (dim,) * n_dims

tt_rank = 5

np.random.seed(179)
tensor_tt = TensorTrain.random(
    shape, rank=tt_rank
) 

nnz = 100
tot_dim = np.prod(shape)
inds_dense = np.random.choice(tot_dim, nnz, replace=False)
inds = np.stack(np.unravel_index(inds_dense, shape))
entries = np.random.normal(size=nnz)
entries = entries * np.logspace(-3, -20, nnz)
tensor_sparse = SparseTensor(shape, inds, entries)

tensor = tensor_sparse + tensor_tt

csv_filename = "results/tt_plus_sparse.csv"
experiment = Experiment(csv_filename)

error_just_tt = tensor_tt.error(tensor, relative=True)
# %%

ranks = range(1, 31)
runs = range(30)

for rank, run in tqdm(list(product(ranks, runs)), desc="OTTS"):
    experiment.do_experiment(
        tensor,
        "OTTS",
        experiment_orthogonal_sketch,
        left_rank=rank,
        right_rank=rank * 2,
        run=run,
    )

for rank, run in tqdm(list(product(ranks, runs)), desc="HMT"):
    experiment.do_experiment(
        tensor,
        "HMT",
        experiment_hmt_sketch,
        rank=rank,
        run=run,
    )
    
for rank, run in tqdm(list(product(ranks, runs)), desc="STTA"):
    experiment.do_experiment(
        tensor,
        "STTA",
        experiment_stream_sketch,
        left_rank=rank,
        right_rank=rank * 2,
        run=run,
    )

for rank in tqdm(ranks, desc="TT-SVD"):
    experiment.do_experiment(tensor, "TT-SVD", experiment_tt_svd, rank=rank)

# %%
import pandas as pd

df = pd.read_csv(csv_filename)

plt.figure(figsize=(10, 4))
ttsvd = df[df["name"] == "TT-SVD"]

# rsketch = df[df["name"] == "OTTS"]

# error_gb = rsketch.groupby(rsketch["left_rank"]).error
# errors05 = error_gb.quantile(0.5).values
# errors08 = error_gb.quantile(0.8).values - errors05
# errors02 = errors05 - error_gb.quantile(0.2).values
# plt.errorbar(
#     plot_ranks - 0.05,
#     errors05,
#     yerr=np.stack([errors02, errors08]),
#     label="OTTS, TT-DRM",
#     capsize=3,
#     linestyle="",
# )
plt.axhline(error_just_tt, label="baseline", linestyle="--", color="k", alpha=0.7)

ssketch = df[df["name"] == "HMT"]
plot_ranks = ssketch["rank"].unique()
plt.plot(plot_ranks, ttsvd.error.values, "-o", label="TT-SVD",ms=3)
error_gb = ssketch.groupby("rank").error
errors05 = error_gb.quantile(0.5).values
errors08 = error_gb.quantile(0.8).values - errors05
errors02 = errors05 - error_gb.quantile(0.2).values
plt.errorbar(
    plot_ranks - 0.1,
    errors05,
    yerr=np.stack([errors02, errors08]),
    label="TT-HMT, TT-DRM",
    capsize=3,
    linestyle="",
)

ssketch = df[df["name"] == "STTA"]
error_gb = ssketch.groupby("left_rank").error
errors05 = error_gb.quantile(0.5).values
errors08 = error_gb.quantile(0.8).values - errors05
errors02 = errors05 - error_gb.quantile(0.2).values
plt.errorbar(
    plot_ranks + 0.1,
    errors05,
    yerr=np.stack([errors02, errors08]),
    label="STTA, TT-DRM",
    capsize=3,
    linestyle="",
)


plt.xticks(ranks)
plt.ylabel("Relative error")
plt.xlabel("TT-rank")
plt.yscale("log")
plt.title(f"Approximation of TT + sparse")
plt.legend()
plt.savefig("results/plot-tt-plus-sparse.pdf", transparent=True, bbox_inches="tight")
plt.show()

# %%
