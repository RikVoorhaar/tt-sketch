# %%
"""Experiment of compressing a CP tensor"""
import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from tt_sketch.tensor import CPTensor

from experiment_base import (
    Experiment,
    experiment_stream_sketch,
    experiment_orthogonal_sketch,
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
cp_rank = 100

np.random.seed(179)
cp_norms = 1 / np.arange(1, cp_rank + 1) ** 5
cores = []
for n in shape:
    core = np.random.normal(size=(n, cp_rank))
    core = core / np.linalg.norm(core, axis=0)
    cores.append(core)
cores[0] = cp_norms * cores[0]

tensor = CPTensor(cores)
tensor_dense = tensor.dense()

csv_filename = "results/cp_tensor.csv"
experiment = Experiment(csv_filename)
# %%

ranks = range(1, 31)
runs = range(20)

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
# plot_ranks = rsketch["left_rank"].unique()

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


ssketch = df[df["name"] == "HMT"]
plot_ranks  = ssketch["rank"].unique()
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
plt.title(f"Approximation of CP tensor")
plt.legend()
plt.savefig("results/plot-cp.pdf", transparent=True, bbox_inches="tight")
plt.show()

# %%
