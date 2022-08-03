# %%
import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from tt_sketch.drm import TensorTrainDRM, DenseGaussianDRM
from tt_sketch.tensor import TensorTrain, TensorSum, Tensor
from tt_sketch.sketch import SketchedTensorTrain, stream_sketch, hmt_sketch
from tt_sketch.utils import process_tt_rank
from experiment_base import (
    Experiment,
    experiment_stream_sketch,
    experiment_tt_round,
    experiment_hmt_sketch,
)

csv_filename = "results/dimension_scaling_guassian.csv"
experiment = Experiment(csv_filename)

seed = 179
tt_rank = 10
round_rank = 8
dim = 5
n_dims_list = np.arange(4, 13, dtype=int)
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
    tensor.orthogonalize()
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


def make_tensor(n_dims, tt_rank, dim, seed=None, min_svdval=-20):
    shape = (dim,) * n_dims
    return tt_exp_decay(shape, tt_rank, seed=seed, min_svdval=min_svdval)



# %%
for n_dims in tqdm(n_dims_list):
    tensor = make_tensor(
        n_dims, tt_rank, dim, seed=seed + n_dims, min_svdval=-5
    )
    round_rank_trim = process_tt_rank(round_rank, tensor.shape, trim=True)
    right_rank = tuple(r*2 for r in round_rank_trim)
    experiment.do_experiment(
        tensor,
        "TT-SVD",
        experiment_tt_round,
        rank=round_rank_trim,
        n_dims=n_dims,
        error_func=tt_error_func,
    )
    for run in runs:
        experiment.do_experiment(
            tensor,
            "HMT",
            experiment_hmt_sketch,
            drm_type=DenseGaussianDRM,
            rank=round_rank_trim,
            run=run,
            n_dims=n_dims,
            error_func=tt_error_func,
        )
        experiment.do_experiment(
            tensor,
            "STTA",
            experiment_stream_sketch,
            drm_type=DenseGaussianDRM,
            left_rank=round_rank_trim,
            right_rank=right_rank,
            run=run,
            n_dims=n_dims,
            error_func=tt_error_func,
        )


# %%
import pandas as pd

df = pd.read_csv(csv_filename, index_col=0)
df.groupby(["name", "n_dims"]).median().reset_index()
n_dims_plot = df["n_dims"].unique()

df[df["name"] == "TT-SVD"]
# %%
from matplotlib.ticker import ScalarFormatter, NullFormatter, AutoMinorLocator

plt.figure(figsize=(8, 4))
marker_size = 4
markers = ["D", "s"]
jitter_dict = {
    "HMT": -0.07,
    "STTA": 0.07,
}
for name, marker in zip(("HMT", "STTA"), markers):
    df_subset = df[df["name"] == name]
    tt_svd = df[ (df["name"] == "TT-SVD")][
        "error"
    ].array
    # tt_svd = 1
    label_name = "TT-HMT" if name == "HMT" else name
    df_gb_median = df_subset.groupby("n_dims").median()
    error80 = df[df["name"] == name].groupby("n_dims")["error"].quantile(0.8)
    error20 = df[df["name"] == name].groupby("n_dims")["error"].quantile(0.2)
    plt.errorbar(
        df_gb_median.index + jitter_dict[name],
        df_gb_median["error"] / tt_svd,
        yerr=(
            (df_gb_median["error"] - error20) / tt_svd,
            (error80 - df_gb_median["error"]) / tt_svd,
        ),
        capsize=5,
        marker=marker,
        markersize=marker_size,
        label=label_name,
        linestyle="",
    )


ax = plt.gca()
ax.set_xticks(n_dims_plot)

plt.axhline(1, color="k", linestyle="--",alpha=0.7)

plt.legend()
plt.title("Scaling of error with number of modes")
plt.xlabel("Number of modes")
plt.ylabel("Normalized error")
plt.savefig(
    "results/plot-dimension-scaling-gaussian.pdf",
    transparent=True,
    bbox_inches="tight",
)

# %%
