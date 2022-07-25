# %%
"""In this experiment we see how much the error can improve when we use TT-SVD
rounding to a fixed rank after first oversampling. Benchmark is then the TT-SVD error."""


from pydoc import describe
import re
import numpy as np
from typing import List
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from tt_sketch.tensor import TensorTrain, TensorSum, Tensor

from experiment_base import (
    Experiment,
    experiment_stream_sketch,
    experiment_tt_svd,
    experiment_hmt_sketch,
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
tt_sum_dense = tensor.dense()
csv_filename = "results/recompression.csv"
experiment = Experiment(csv_filename)

# %%

recompression_rank = 10
sketch_ranks = range(recompression_rank, recompression_rank + 20)
runs = range(30)

for rank, run in tqdm(list(product(sketch_ranks, runs)), desc="HMT"):
    experiment.do_experiment(
        tensor,
        "HMT",
        experiment_hmt_sketch,
        rank=rank,
        recompression_rank=recompression_rank,
        run=run,
    )

for rank, run in tqdm(list(product(sketch_ranks, runs)), desc="STTA+3"):
    experiment.do_experiment(
        tensor,
        "STTA+3",
        experiment_stream_sketch,
        left_rank=rank,
        right_rank=rank + 3,
        recompression_rank=recompression_rank,
        run=run,
    )

for rank, run in tqdm(list(product(sketch_ranks, runs)), desc="STTAx2"):
    experiment.do_experiment(
        tensor,
        "STTAx2",
        experiment_stream_sketch,
        left_rank=rank,
        right_rank=rank + 3,
        recompression_rank=recompression_rank,
        run=run,
    )
experiment.do_experiment(
    tensor, "tt_svd", experiment_tt_svd, rank=recompression_rank
)

# %%
import pandas as pd
plt.figure(figsize=(8, 4))

df = pd.read_csv(csv_filename, index_col=0)
tt_svd_erorr = df[df["name"]=="tt_svd"].iloc[0]["error"]
# plt.axhline(tt_svd_erorr, ls="--", color="k", label="TT-SVD")

label = "HMT"
rank_label = "rank"
plot_pairs = [("HMT", "rank"), ("STTA+3", "left_rank"), ("STTAx2", "left_rank")]
markers = ["s","v","^"]
for i, (label, rank_label) in enumerate(plot_pairs):
    df_method = df[df["name"] ==label].copy()
    df_method["log_error"] = np.log10(df_method["error"] / tt_svd_erorr)
    ranks = df_method[rank_label].unique()
    error80 = df_method.groupby(rank_label)["log_error"].quantile(0.8)
    error50 = df_method.groupby(rank_label)["log_error"].median()
    error20 = df_method.groupby(rank_label)["log_error"].quantile(0.2)

    jitter = np.linspace(-0.2, 0.2, len(plot_pairs))[i]
    plt.errorbar(
        ranks+jitter,
        error50,
        yerr=[error50 - error20, error80 - error50],
        capsize=2,
        label=label,
        linestyle="",
        marker=markers[i],
        ms="3"
    )
plt.xticks(ranks)
plt.yscale("log")
plt.xlabel("TT-rank")
plt.ylabel("Error relative to TT-SVD")
plt.legend()

locs_new = np.array([10,3,1.8,1.33,1.15,1.075,1.04,1.02,1.01,1.005])
labels_new =[str(s) for s in locs_new]
locs_new = np.log10(locs_new)
plt.yticks(locs_new, labels_new)
plt.minorticks_off()

# %%
locs_new
10**np.logspace(np.log10(1),np.log10(0.002), 10)