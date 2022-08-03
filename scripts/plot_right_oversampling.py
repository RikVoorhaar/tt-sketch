# %%
"""In this experiment we compare the effect of different levels of
right-oversampling. On both the median error, and the variance in the error."""

"""As target tensor let's pick something for which we get a decent amount of
variance. For example the sum-of-tt tensor"""

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from typing import List
from tt_sketch.drm import TensorTrainDRM
import matplotlib.pyplot as plt
from tt_sketch.tensor import TensorTrain, TensorSum, Tensor

from experiment_base import (
    Experiment,
    experiment_orthogonal_sketch,
    experiment_stream_sketch,
    experiment_tt_svd,
)

n_dims = 5
dim = 10
shape = (dim,) * n_dims

tt_rank = 3
num_tts = 20

np.random.seed(179)
tt_summands_list: List[Tensor] = []
coeffs = np.logspace(0, -20, num_tts)
for coeff in coeffs:
    tensor_tt = TensorTrain.random(shape, rank=tt_rank) * coeff
    tt_summands_list.append(tensor_tt)
tensor = TensorSum(tt_summands_list)
csv_filename = "results/right_oversampling.csv"
experiment = Experiment(csv_filename)

# %%
left_rank = 10
# right_ranks = range(11, 31)
oversampling_params = range(2, 41, 2)

runs = range(100)

# for right_rank, run in tqdm(list(product(right_ranks, runs)), desc="OTTS"):
#     experiment.do_experiment(
#         tensor,
#         "OTTS",
#         experiment_orthogonal_sketch,
#         left_rank=left_rank,
#         right_rank=right_rank,
#         run=run,
#     )

for oversampling, run in tqdm(
    list(product(oversampling_params, runs)), desc="STTA"
):
    experiment.do_experiment(
        tensor,
        "STTA",
        experiment_stream_sketch,
        left_rank=left_rank,
        right_rank=left_rank + oversampling,
        run=run,
    )

experiment.do_experiment(tensor, "tt_svd", experiment_tt_svd, rank=left_rank)
experiment.do_experiment(
    tensor, "tt_svd", experiment_tt_svd, rank=left_rank - 1
)
experiment.do_experiment(
    tensor, "tt_svd", experiment_tt_svd, rank=left_rank - 2
)

# %%
df = pd.read_csv(csv_filename)

plt.figure(figsize=(8, 4))
ttsvd = df[(df["name"] == "tt_svd") & (df["rank"] == left_rank-2)]
# plt.axhline(ttsvd["error"].iloc[0], ls="--", color="k", label="TT-SVD")

# rsketch = df[df["name"] == "OTTS"]
# right_ranks = rsketch["right_rank"].unique()
# plot_ranks = right_ranks - left_rank
# error_gb = rsketch.groupby(rsketch["right_rank"]).error
# errors05 = error_gb.quantile(0.5).values
# plt.plot(plot_ranks, errors05, marker=".", label="OTTS median")
# errors08 = error_gb.quantile(0.8).values
# plt.plot(plot_ranks, errors08, "--", label="OTTS 80th percentile")
# errors02 = error_gb.quantile(0.2).values
# plt.plot(plot_ranks, errors02, "--", label="OTTS 20th percentile")


ssketch = df[df["name"] == "STTA"]
right_ranks = ssketch["right_rank"].unique()
plot_ranks = right_ranks - left_rank
error_gb = ssketch.groupby(ssketch["right_rank"]).error
errors05 = error_gb.quantile(0.5).values
plt.plot(plot_ranks, errors05, marker=".", label="STTA median")
errors08 = error_gb.quantile(0.8).values
plt.plot(plot_ranks, errors08, "-.", label="80th percentile")
errors02 = error_gb.quantile(0.2).values
plt.plot(plot_ranks, errors02, "-.", label="20th percentile")

plt.yscale("log")
plt.xticks(plot_ranks)
plt.xlabel("Right oversampling $\ell$")
plt.ylabel("L2 error")
plt.legend()
plt.title("Effect of right oversampling $\ell$ on approximation error")

plt.savefig(
    "results/plot-right-oversampling.pdf", transparent=True, bbox_inches="tight"
)
plt.show()
# %%
