# %%
"""In this experiment we compare the effect of different levels of
right-oversampling. On both the median error, and the variance in the error."""

"""As target tensor let's pick something for which we get a decent amount of
variance. For example the sum-of-tt tensor"""

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from tt_stream_sketch.drm import DenseGaussianDRM, TensorTrainDRM
import matplotlib.pyplot as plt
from tt_stream_sketch.tensor import TensorTrain, TensorSum
from tt_stream_sketch.sketched_tensor_train import SketchedTensorTrain

from experiment_base import (
    Experiment,
    experiment_recursive_sketch,
    experiment_tensor_sketch,
    experiment_tt_svd,
)

n_dims = 5
dim = 10
shape = (dim,) * n_dims

tt_rank = 3
num_tts = 20

np.random.seed(179)
tt_sum = []
coeffs = np.logspace(0, -8, num_tts)
for coeff in coeffs:
    tensor_tt = TensorTrain.random(shape, rank=tt_rank)
    tensor_tt[0] = tensor_tt[0] * coeff
    tt_sum.append(tensor_tt)
tt_sum = TensorSum(tt_sum)
tt_sum_dense = tt_sum.dense()
csv_filename = "results/right_oversampling.csv"
experiment = Experiment(csv_filename)

# %%
left_rank = 10
right_ranks = range(10, 31)

runs = range(20)

for right_rank, run in tqdm(
    list(product(right_ranks, runs)), desc="recursive big-sketch"
):
    experiment.do_experiment(
        tt_sum_dense,
        "recursive_sketch",
        experiment_recursive_sketch,
        left_rank=left_rank,
        right_rank=right_rank,
        run=run,
    )

for right_rank, run in tqdm(
    list(product(right_ranks, runs)), desc="Sparse-sketch"
):
    experiment.do_experiment(
        tt_sum,
        "sketched_dense",
        experiment_tensor_sketch,
        left_rank=left_rank,
        right_rank=right_rank,
        left_sketch_type=TensorTrainDRM,
        right_sketch_type=TensorTrainDRM,
        run=run,
    )

experiment.do_experiment(
    tt_sum_dense, "tt_svd", experiment_tt_svd, rank=left_rank
)

# %%
df = pd.read_csv(csv_filename)

plt.figure(figsize=(8, 4))
ttsvd = df[df["name"] == "tt_svd"]
plt.axhline(ttsvd["error"].iloc[0], ls="--", color="k", label="TT-SVD")

rsketch = df[df["name"] == "recursive_sketch"]
right_ranks = rsketch["right_rank"].unique()

error_gb = rsketch.groupby(rsketch["right_rank"]).error
errors05 = error_gb.quantile(0.5).values
plt.plot(errors05, marker=".", label="OTTS median")
errors08 = error_gb.quantile(0.8).values
plt.plot(errors08, "--", label="OTTS 80th percentile")
errors02 = error_gb.quantile(0.2).values
plt.plot(errors02, "--", label="OTTS 20th percentile")
plt.yscale("log")
plt.xticks(right_ranks - left_rank)
plt.xlabel("Right oversampling $\ell$")
plt.legend()


ssketch = df[df["name"] == "sketched_dense"]
error_gb = ssketch.groupby(ssketch["right_rank"]).error
errors05 = error_gb.quantile(0.5).values
plt.plot(errors05, marker=".", label="STTA median")
errors08 = error_gb.quantile(0.8).values
plt.plot(errors08, "-.", label="STTA 80th percentile")
errors02 = error_gb.quantile(0.2).values
plt.plot(errors02, "-.", label="STTA 20th percentile")


plt.yscale("log")
plt.xticks(right_ranks - left_rank)
plt.xlabel("Right oversampling $\ell$")
plt.ylabel("L2 error")
plt.legend()
plt.title("Effect of right oversampling $\ell$ on approximation error")

plt.savefig(
    "results/plot-right-oversampling.pdf", transparent=True, bbox_inches="tight"
)
plt.show()
# %%
