# %%
"""
Simple timings experiment. We compare the time required to sketch a TT of
varying rank. The rank of the sketches also vary.
"""

import numpy as np
import pandas as pd
from experiment_base import (
    Experiment,
    experiment_orthogonal_sketch,
    experiment_stream_sketch,
    experiment_hmt_sketch,
    experiment_tt_svd,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from tt_sketch.sketch import (
    SketchedTensorTrain,
    stream_sketch,
    orthogonal_sketch,
    hmt_sketch,
)
from itertools import product

from tt_sketch.tensor import TensorTrain

csv_filename = "results/timings150.csv"

num_runs = 100
runs = list(range(num_runs))
shape = (100,) * 5
experiment = Experiment(csv_filename)
tt_rank = 150
sketch_ranks = np.arange(5, tt_rank + 0.1, 10, dtype=int)
SEED = 179

# %%


def tt_exp_decay(shape, tt_rank, min_svdval=-20, seed=None):
    tensor = TensorTrain.random(
        shape, rank=tt_rank, orthog=True, trim=True, seed=seed
    )
    for i, C in enumerate(tensor.cores):
        C_shape = C.shape
        left_mat_shape = (C_shape[0] * C.shape[1], C_shape[2])
        right_mat_shape = (C_shape[0], C_shape[1] * C.shape[2])
        if min(left_mat_shape) > min(right_mat_shape):
            mat_shape = left_mat_shape
        else:
            mat_shape = right_mat_shape
        C = C.reshape(mat_shape)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        S *= np.logspace(0, min_svdval, len(S))
        C = (U @ np.diag(S) @ Vt).reshape(C_shape)
        tensor.cores[i] = C
    return tensor


def tt_error_func(tt1, tt2):
    if isinstance(tt1, SketchedTensorTrain):
        tt1 = tt1.to_tt()
    tt2_norm = tt2.norm()
    if tt2_norm == 0:
        return np.infa
    error = tt1.add(-tt2).norm() / tt2_norm
    return error


tensors = []
for i in tqdm(runs, desc="creating target tensors"):
    tensor = tt_exp_decay(shape, min_svdval=-10, tt_rank=tt_rank, seed=SEED + i)
    tensors.append(tensor)

# %%
# for tt_rank, sketch_rank, run in tqdm(
#     list(product(tt_ranks, sketch_ranks, runs)), desc="OTTS"
# ):
#     # tensor = TensorTrain.random(shape, rank=tt_rank)
#     tensor = tt_exp_decay(shape, tt_rank, seed=SEED+run)
#     experiment.do_experiment(
#         tensor,
#         "OTTS",
#         experiment_orthogonal_sketch,
#         left_rank=sketch_rank,
#         right_rank=sketch_rank * 2,
#         tensor_rank=tt_rank,
#         sketch_rank=sketch_rank,
#         run=run,
#     )


for run, sketch_rank in tqdm(list(product(runs, sketch_ranks)), desc="STTAx2"):
    # tensor = TensorTrain.random(shape, rank=tt_rank)
    # tensor = tt_exp_decay(shape, tt_rank, seed=SEED + run)
    tensor = tensors[run]
    experiment.do_experiment(
        tensor,
        "STTAx2",
        experiment_stream_sketch,
        left_rank=sketch_rank,
        right_rank=sketch_rank * 2,
        tensor_rank=tt_rank,
        sketch_rank=sketch_rank,
        run=run,
        error_func=tt_error_func,
    )

for run, sketch_rank in tqdm(list(product(runs, sketch_ranks)), desc="STTA+3"):
    # tensor = TensorTrain.random(shape, rank=tt_rank)
    # tensor = tt_exp_decay(shape, tt_rank, seed=SEED + run)
    tensor = tensors[run]
    experiment.do_experiment(
        tensor,
        "STTA+3",
        experiment_stream_sketch,
        left_rank=sketch_rank,
        right_rank=sketch_rank + 3,
        tensor_rank=tt_rank,
        sketch_rank=sketch_rank,
        run=run,
        error_func=tt_error_func,
    )

# for run, sketch_rank in tqdm(list(product(runs, sketch_ranks)), desc="OTTS+3"):
#     # tensor = TensorTrain.random(shape, rank=tt_rank)
#     # tensor = tt_exp_decay(shape, tt_rank, seed=SEED + run)
#     tensor = tensors[run]
#     experiment.do_experiment(
#         tensor,
#         "OTTS+3",
#         experiment_orthogonal_sketch,
#         left_rank=sketch_rank,
#         right_rank=sketch_rank + 3,
#         tensor_rank=tt_rank,
#         sketch_rank=sketch_rank,
#         run=run,
#         error_func=tt_error_func,
#     )

# for run, sketch_rank in tqdm(list(product(runs, sketch_ranks)), desc="OTTSx2"):
#     # tensor = TensorTrain.random(shape, rank=tt_rank)
#     # tensor = tt_exp_decay(shape, tt_rank, seed=SEED + run)
#     tensor = tensors[run]
#     experiment.do_experiment(
#         tensor,
#         "OTTSx2",
#         experiment_orthogonal_sketch,
#         left_rank=sketch_rank,
#         right_rank=sketch_rank * 2,
#         tensor_rank=tt_rank,
#         sketch_rank=sketch_rank,
#         run=run,
#         error_func=tt_error_func,
#     )

for run, sketch_rank in tqdm(list(product(runs, sketch_ranks)), desc="HMT"):
    # tensor = TensorTrain.random(shape, rank=tt_rank, seed=SEED + run)
    # tensor = tt_exp_decay(shape, tt_rank)
    tensor = tensors[run]
    experiment.do_experiment(
        tensor,
        "HMT",
        experiment_hmt_sketch,
        rank=sketch_rank,
        tensor_rank=tt_rank,
        sketch_rank=sketch_rank,
        run=run,
        error_func=tt_error_func,
    )

# --------------------------------

# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(csv_filename)
label_dic = {
    "STTAx2": "STTAx2, TT-DRM",
    "STTA+3": "STTA+3, TT-DRM",
    "HMT": "TT-HMT, TT-DRM",
}
df = df[df["name"].isin(label_dic)]


def make_percentile_function(percentile):
    def f(x):
        return np.percentile(x, percentile)

    return f


percentile80 = make_percentile_function(80)
percentile20 = make_percentile_function(20)

timing_df = (
    df.groupby(["name", "sketch_rank", "tensor_rank"])[["time_taken", "error"]]
    .agg(
        min_time=("time_taken", "min"),
        time50=("time_taken", "median"),
        std_time=("time_taken", "std"),
        error50=("error", "median"),
        std_error=("error", "std"),
        time20=("time_taken", percentile20),
        time80=("time_taken", percentile80),
        error80=("error", percentile80),
        error20=("error", percentile20),
    )
    .reset_index()
)
max_xtick = np.ceil(timing_df.time50.max() * 10 + 1) / 10
timing_df.name.unique()
# %%
markers = ["s", "o", "D"]
for i, label in enumerate(("HMT", "STTA+3", "STTAx2")):
    sub_df = timing_df[(timing_df.name == label)]

    plt.plot(
        sub_df.time50,
        sub_df.error50,
        # xerr=[sub_df.time50 - sub_df.time20, sub_df.time80 - sub_df.time50],
        # yerr=[sub_df.error50 - sub_df.error20, sub_df.error80 - sub_df.error50],
        # capsize=3,
        marker=markers[i],
        ms=4,
        label=label_dic[label],
        # linestyle="",
    )

plt.ylabel("Relative error")
plt.xlabel("Time taken (s)")
plt.legend()
plt.yscale("log")
# plt.xscale("log")
plt.xlim(0, None)
plt.title("Error vs. time taken")
plt.xticks(np.arange(0, max_xtick, 0.1))
plt.savefig("results/timings1.pdf", transparent=True, bbox_inches="tight")

# %%
markers = ["s", "o", "D"]
for i, label in enumerate(("HMT", "STTA+3", "STTAx2")):
    sub_df = timing_df[(timing_df.name == label)]

    plt.plot(
        sub_df.sketch_rank,
        sub_df.time50,
        # yerr=[sub_df.time50 - sub_df.time20, sub_df.time80 - sub_df.time50],
        # capsize=3,
        label=label_dic[label],
        marker=markers[i],
        ms=4
        # linestyle="",
    )

plt.xlabel("TT-rank of sketch")
plt.ylabel("Time taken (s)")
plt.legend()
plt.yticks(np.arange(0, max_xtick, 0.2))
# plt.yscale("log")
plt.ylim(0, None)
plt.xlim(0, None)
# plt.xticks(np.concatenate([np.arange(5, 151, 10)]))
plt.xticks(timing_df.sketch_rank.unique())
plt.title("Time taken vs. sketch rank")
plt.savefig("results/timings2.pdf", transparent=True, bbox_inches="tight")
# %%
timing_df.sketch_rank.unique()
