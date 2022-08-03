# %%
"""
In this script we investigate the relationship between error and timing for the
three different method. We want to see if OTTS makes sense versus the HMT
method. Despite the fact that OTTS is better, it's also a little slower. We need
to see if there are situations where OTTS can lead to a better approximation in
the same amount of time as HMT.
"""

from unittest import result
import numpy as np
import pandas as pd
from typing import List
from experiment_base import (
    Experiment,
    experiment_orthogonal_sketch,
    experiment_stream_sketch,
    experiment_hmt_sketch,
    experiment_tt_svd,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from tt_sketch.sketch import stream_sketch, orthogonal_sketch, hmt_sketch
from itertools import product
import time
from tt_sketch.tensor import TensorTrain, TensorSum, Tensor, SparseTensor

n_dims = 5
dim = 1000
shape = (dim,) * n_dims

tt_rank = 100
num_tts = 10

np.random.seed(179)
tt_summands_list: List[Tensor] = []
coeffs = np.logspace(0, -10, num_tts)
for coeff in coeffs:
    tensor_tt = TensorTrain.random(shape, rank=tt_rank) * coeff
    tt_summands_list.append(tensor_tt)
tensor = TensorSum(tt_summands_list)
# tt_sum_dense = tensor.dense()

# n_dims = 6
# dim = 5
# shape = (dim,) * n_dims

# tt_rank = 3

# np.random.seed(179)
# tensor_tt = TensorTrain.random(shape, rank=tt_rank)

# nnz = 100
# tot_dim = np.prod(shape)
# inds_dense = np.random.choice(tot_dim, nnz, replace=False)
# inds = np.stack(np.unravel_index(inds_dense, shape))
# entries = np.random.normal(size=nnz)
# entries = entries * np.logspace(-3, -20, nnz)
# tensor_sparse = SparseTensor(shape, inds, entries)

# tensor = tensor_sparse + tensor_tt


# %%
import cProfile
cProfile.run('hmt_sketch(tensor, rank=10)')
# %%
max_rank = 5
results = []
runs = range(10)
ranks = range(max_rank, max_rank + 20)
# ranks = [200]


def hmt_func(tensor, left_rank, right_rank):
    return hmt_sketch(tensor, rank=left_rank)


method_dict = {
    "OTTS": orthogonal_sketch,
    "STTA": stream_sketch,
    "HMT": hmt_func,
}
for run, r, method in tqdm(list(product(runs, ranks, method_dict.keys()))):
    current_time = time.perf_counter()
    stt = method_dict[method](tensor, left_rank=r, right_rank=r+1)
    if not isinstance(stt, TensorTrain):
        tt = stt.to_tt()
    else:
        tt = stt
    orthogonalize = method != "STTA"
    # tt = tt.round(max_rank=max_rank, orthogonalize=orthogonalize)
    time_taken = time.perf_counter() - current_time
    error = tt.error(tensor, fast=False, relative=True)
    results.append(
        {
            "method": method,
            "rank": r,
            "time": time_taken,
            "error": error,
            "run": run,
        }
    )
# %%
shape_string = "x".join(str(x) for x in shape)
ranks_string = str(min(ranks)) + "-" + str(max(ranks))
plt.figure(figsize=(7, 5))
title = f"shape={shape_string}, {tt_rank=}, {num_tts=}, ranks={ranks_string}"
plt.title(title)
results_df = pd.DataFrame(results)
plot_df = (
    results_df.groupby(["method", "rank"])[["error", "time"]]
    .median()
    .reset_index()
)
for method in method_dict.keys():
    plt.plot(
        plot_df.loc[plot_df["method"] == method, "time"],
        plot_df.loc[plot_df["method"] == method, "error"],
        "o",
        label=method,
    )
plt.legend()
plt.yscale("log")
plt.xlabel("Time (s)")
plt.ylabel("Error")

# %%
