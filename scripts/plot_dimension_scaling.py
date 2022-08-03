# %%
import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from tt_sketch.drm.tensor_train_drm import TensorTrainDRM
from tt_sketch.tensor import TensorTrain, TensorSum, Tensor
from tt_sketch.sketch import SketchedTensorTrain, stream_sketch, hmt_sketch
from experiment_base import (
    Experiment,
    experiment_stream_sketch,
    experiment_tt_round,
    experiment_hmt_sketch,
)

csv_filename = "results/dimension_scaling.csv"
experiment = Experiment(csv_filename)

seed = 179
tt_rank = 30
round_rank = 10
dim = 30
n_dims_list = 2 ** np.arange(2, 14)
runs = range(30)
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
        S = np.logspace(0, min_svdval, len(S))
        S = S * np.sqrt(min(mat_shape))
        C = (U @ np.diag(S) @ Vt).reshape(C_shape)
        tensor.cores[i] = C
    return tensor


def tt_error_func(tt1, tt2, relative=True):
    if isinstance(tt1, SketchedTensorTrain):
        tt1 = tt1.to_tt()
    tt2_norm = tt2.norm()
    if tt2_norm == 0:
        return np.inf
    error = tt1.add(-tt2).norm()
    if relative:
        return error / tt2_norm
    return error


def make_tensor(n_dims, tt_rank, dim, seed=None):
    shape = (dim,) * n_dims
    return tt_exp_decay(shape, tt_rank, seed=seed)


# %%
for n_dims in tqdm(n_dims_list):
    tensor = make_tensor(n_dims, tt_rank, dim, seed=seed + n_dims)
    experiment.do_experiment(
        tensor,
        "TT-SVD",
        experiment_tt_round,
        rank=round_rank,
        n_dims=n_dims,
        error_func=tt_error_func,
    )
    experiment.do_experiment(
        tensor,
        "TT-SVD",
        experiment_tt_round,
        rank=round_rank - 1,
        n_dims=n_dims,
        error_func=tt_error_func,
    )
    experiment.do_experiment(
        tensor,
        "TT-SVD",
        experiment_tt_round,
        rank=round_rank - 2,
        n_dims=n_dims,
        error_func=tt_error_func,
    )
    for run in runs:
        experiment.do_experiment(
            tensor,
            "HMT",
            experiment_hmt_sketch,
            rank=round_rank,
            run=run,
            n_dims=n_dims,
            error_func=tt_error_func,
        )
        experiment.do_experiment(
            tensor,
            "STTA",
            experiment_stream_sketch,
            left_rank=round_rank,
            right_rank=round_rank * 2,
            run=run,
            n_dims=n_dims,
            error_func=tt_error_func,
        )


# %%
import pandas as pd

df = pd.read_csv(csv_filename, index_col=0)
df.groupby(["name", "n_dims"]).median().reset_index()
n_dims_plot = df["n_dims"].unique()


# %%
df["error"].array
# %%
from matplotlib.ticker import ScalarFormatter, NullFormatter, AutoMinorLocator

plt.figure(figsize=(8, 4))
marker_size = 4
markers = ["o", "o", "D", "s"]
jitter_dict = {
    "HMT": 1 / 1.05,
    "STTA": 1.05,
}
for name, marker in zip(
    ("TT-SVD, rank 8", "TT-SVD, rank 9", "HMT", "STTA"), markers
):
    if name == "TT-SVD, rank 8":
        df_subset = df[(df["rank"] == 8) & (df["name"] == "TT-SVD")]
    elif name == "TT-SVD, rank 9":
        df_subset = df[(df["rank"] == 9) & (df["name"] == "TT-SVD")]
    else:
        df_subset = df[df["name"] == name]
    tt_svd = df[(df["rank"] == 10) & (df["name"] == "TT-SVD")]["error"].array
    label_name = "TT-HMT" if name == "HMT" else name
    df_gb_median = df_subset.groupby("n_dims").median()
    if name.startswith("TT-SVD"):
        continue
        plt.plot(
            df_gb_median.index,
            df_gb_median["error"] / tt_svd,
            marker=marker,
            markersize=marker_size,
            label=label_name,
        )
    else:
        error80 = (
            df[df["name"] == name].groupby("n_dims")["error"].quantile(0.8)
        )
        error20 = (
            df[df["name"] == name].groupby("n_dims")["error"].quantile(0.2)
        )
        plt.errorbar(
            df_gb_median.index * jitter_dict[name],
            df_gb_median["error"] / tt_svd,
            yerr=(
                (df_gb_median["error"] - error20) / tt_svd,
                (error80 - df_gb_median["error"]) / tt_svd,
            ),
            capsize=5,
            marker=marker,
            markersize=marker_size,
            label=label_name,
            linewidth=1,
            # linestyle="",

        )

plt.xscale("log")

ax = plt.gca()
ax.set_xticks(n_dims_plot)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_tick_params(which="minor", size=0)
ax.xaxis.set_tick_params(which="minor", width=0)

ax.set_yticks(np.arange(2, 18 + 0.1, 2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

plt.legend()
plt.title("Dependence of the error on the tensor order")
plt.xlabel("Order of tensor")
plt.ylabel("Normalized error")
plt.savefig(
    "results/plot-dimension-scaling.pdf", transparent=True, bbox_inches="tight"
)

# %%
