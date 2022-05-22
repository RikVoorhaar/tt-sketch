# %%
"""Experiment of compressing a CP tensor"""
import numpy as np
from itertools import product
from tqdm import tqdm
from tt_sketch.drm import DenseGaussianDRM, TensorTrainDRM
import matplotlib.pyplot as plt
from tt_sketch.tensor import CPTensor
from tt_sketch.sketched_tensor_train import SketchedTensorTrain

from experiment_base import (
    Experiment,
    experiment_recursive_sketch,
    experiment_tensor_sketch,
    experiment_tt_svd,
)

n_dims = 5
dim = 10
shape = (dim,) * n_dims
cp_rank = 100

np.random.seed(179)
cp_norms = 1 / np.arange(1, cp_rank + 1) ** 2
cores = []
for n in shape:
    core = np.random.normal(size=(n, cp_rank))
    core = core / np.linalg.norm(core, axis=0)
    cores.append(core)
cores[0] = cp_norms * cores[0]

tensor_cp = CPTensor(cores)
tensor_dense = tensor_cp.dense()

csv_filename = "results/cp_tensor.csv"
experiment = Experiment(csv_filename)
# %%

ranks = range(1, 30)
sketch_types = [
    TensorTrainDRM,
]
runs = range(20)

for rank, run in tqdm(list(product(ranks, runs)), desc="recursive big-sketch"):
    experiment.do_experiment(
        tensor_dense,
        "recursive_sketch",
        experiment_recursive_sketch,
        left_rank=rank,
        right_rank=rank*2,
        run=run,
    )


for rank, sketch_type, run in tqdm(
    list(product(ranks, sketch_types, runs)), desc="Sparse-sketch"
):
    experiment.do_experiment(
        tensor_cp,
        "sketched_dense",
        experiment_tensor_sketch,
        left_rank=rank,
        right_rank=rank * 2,
        left_sketch_type=sketch_type,
        run=run,
    )

for rank in tqdm(ranks, desc="tt_svd"):
    experiment.do_experiment(tensor_cp, "tt_svd", experiment_tt_svd, rank=rank)

# %%
import pandas as pd

df = pd.read_csv(csv_filename)
rsketch = df[df["name"] == "recursive_sketch"]
ranks = rsketch["left_rank"].unique()
plt.figure(figsize=(8,4))

ttsvd = df[df["name"] == "tt_svd"]
plt.plot(ranks, ttsvd.error.values, "o", label="TTSVD")

error_gb = rsketch.groupby(rsketch["left_rank"]).error
errors08 = error_gb.quantile(0.8).values
errors05 = error_gb.quantile(0.5).values
errors02 = error_gb.quantile(0.2).values
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
plt.title(f"Approximation of CP tensor")
plt.legend()
plt.savefig("results/plot-cp.pdf", transparent=True, bbox_inches="tight")
plt.show()
