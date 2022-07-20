# %%
"""
Simple timings experiment. We compare the time required to sketch a TT of varying rank. The rank of the sketches also vary.
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
from tt_sketch.sketch import stream_sketch, orthogonal_sketch, hmt_sketch
from itertools import product

from tt_sketch.tensor import TensorTrain

csv_filename = "results/timings.csv"

sketch_ranks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
tt_ranks = [50]
runs = range(5)
shape = (50,) * 4
experiment = Experiment(csv_filename)

for tt_rank, sketch_rank, run in tqdm(
    list(product(tt_ranks, sketch_ranks, runs)), desc="OTTS"
):
    tensor = TensorTrain.random(shape, rank=tt_rank)
    experiment.do_experiment(
        tensor,
        "OTTS",
        experiment_orthogonal_sketch,
        left_rank=sketch_rank,
        right_rank=sketch_rank * 2,
        tensor_rank=tt_rank,
        sketch_rank=sketch_rank,
        run=run,
    )


for tt_rank, sketch_rank, run in tqdm(
    list(product(tt_ranks, sketch_ranks, runs)), desc="STTA"
):
    tensor = TensorTrain.random(shape, rank=tt_rank)
    experiment.do_experiment(
        tensor,
        "STTA",
        experiment_stream_sketch,
        left_rank=sketch_rank,
        right_rank=sketch_rank * 2,
        tensor_rank=tt_rank,
        sketch_rank=sketch_rank,
        run=run,
    )

for tt_rank, sketch_rank, run in tqdm(
    list(product(tt_ranks, sketch_ranks, runs)), desc="HMT"
):
    tensor = TensorTrain.random(shape, rank=tt_rank)
    experiment.do_experiment(
        tensor,
        "HMT",
        experiment_hmt_sketch,
        rank=sketch_rank,
        tensor_rank=tt_rank,
        sketch_rank=sketch_rank,
        run=run,
    )

for tt_rank, sketch_rank,run in tqdm(
    list(product(tt_ranks, sketch_ranks, runs)), desc="TT-SVD"
):
    tensor = TensorTrain.random(shape, rank=tt_rank)
    experiment.do_experiment(
        tensor,
        "TT-SVD",
        experiment_tt_svd,
        rank=sketch_rank,
        tensor_rank=tt_rank,
        sketch_rank=sketch_rank,
        run=run,
    )

# %%

df = pd.read_csv(csv_filename)

timing_df = df.groupby(["name", "sketch_rank", "tensor_rank"]).time_taken.median().reset_index()

sub_df = timing_df[(timing_df.name == "HMT") & (timing_df.tensor_rank==50)]
plt.plot(sub_df.sketch_rank, sub_df.time_taken, label="HMT")

sub_df = timing_df[(timing_df.name == "OTTS") & (timing_df.tensor_rank==50)]
plt.plot(sub_df.sketch_rank, sub_df.time_taken, label="OTTS")

sub_df = timing_df[(timing_df.name == "STTA") & (timing_df.tensor_rank==50)]
plt.plot(sub_df.sketch_rank, sub_df.time_taken, label="STTA")

sub_df = timing_df[(timing_df.name == "TT-SVD") & (timing_df.tensor_rank==50)]
plt.plot(sub_df.sketch_rank, sub_df.time_taken, label="TT-SVD")

plt.xlabel("sketch_rank")
plt.ylabel("time taken (s)")
plt.legend()
plt.yscale('log')