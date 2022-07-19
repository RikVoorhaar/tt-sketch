# %%
import logging
from itertools import product
from time import perf_counter
from typing import Tuple

import numpy as np
from tqdm import tqdm
from tt_sketch.tensor import TensorSum, TensorTrain
from tt_sketch.tt_gmres import ROUNDING_MODE, round_tt_sum, tt_sum_gmres

from cookie_problem import prepare_cookie_problem
from experiment_base import Experiment

LOG_FILE = "cookie.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

map_sum, B, precond_map = prepare_cookie_problem(100, 4)
x0 = TensorTrain.zero(shape=map_sum.in_shape, rank=1)

csv_filename = "results/cookie_2x2.csv"
experiment = Experiment(csv_filename)


def experiment_cookie(
    _,
    rounding_method: ROUNDING_MODE,
    rounding_method_final: ROUNDING_MODE,
    max_rank: int,
    final_round_ranks: Tuple[int, ...],
    **kwargs,
):
    logging.info("\n")
    logging.info("-" * 80)
    logging.info(f"Rounding method: {rounding_method}")
    logging.info(f"Rounding method final: {rounding_method_final}")
    logging.info(f"Max rank: {max_rank}")
    logging.info(f"Final round ranks: {final_round_ranks}")
    logging.info(f"Kwargs: {kwargs}")
    logging.info("\n")
    result, history = tt_sum_gmres(
        A=map_sum,
        b=B,
        precond=precond_map,
        x0=x0,
        max_rank=max_rank,
        final_round_rank=None,
        maxiter=50,
        tolerance=0,
        symmetric=True,
        rounding_method=rounding_method,
        rounding_method_final=None,
        verbose=True,
    )
    experiment_result = {}
    experiment_result["last_residual"] = history["residual_norm"][-1]
    experiment_result["gmres_time"] = np.sum(history["step_time"])
    # experiment_result["final_round_time"] = history["final_round_time"]

    true_errors = []
    precond_errors = []
    round_timings = []
    for rank in final_round_ranks:
        current_time = perf_counter()
        result_rounded = round_tt_sum(
            result, rank, method=rounding_method_final, eps=None  # type: ignore
        )
        round_timings.append(perf_counter() - current_time)
        true_error = map_sum(result_rounded).error(B, relative=True)
        precond_error = TensorSum(
            [precond_map(t) for t in map_sum(result_rounded).tensors]
        ).error(precond_map(B), relative=True)
        true_errors.append(true_error)
        precond_errors.append(precond_error)
        logging.info(f"Rank {rank}: true error: {true_error:.4e}")
        logging.info(f"Rank {rank}: preconditioned error: {precond_error:.4e}")

    experiment_result["true_error"] = np.array(true_errors)
    experiment_result["precond_error"] = np.array(precond_errors)
    experiment_result["final_round_time"] = np.array(round_timings)

    return experiment_result


sketch_ranks = list(range(10, 101, 5))
pairwise_ranks = list(range(10, 51, 5))
round_methods = ["sketch", "pairwise"]
final_round_methods = ["sketch", "pairwise"]
final_round_ranks = tuple(range(10, 101, 5))
runs = list(range(20))


all_experiments = []
for run in runs:
    all_experiments.extend(
        list(
            product(["sketch"], ["sketch"], sketch_ranks, [final_round_ranks], [run])
        )
    )
    all_experiments.extend(
        list(
            product(
                ["pairwise"], ["sketch"], pairwise_ranks, [final_round_ranks], [run]
            )
        )
    )
    all_experiments.extend(
        list(
            product(
                ["sketch"], ["pairwise"], sketch_ranks, [final_round_ranks], [run]
            )
        )
    )
    all_experiments.extend(
        list(
            product(
                ["pairwise"],
                ["pairwise"],
                pairwise_ranks,
                [final_round_ranks],
                [run],
            )
        )
    )

for (
    rounding_method,
    rounding_method_final,
    max_rank,
    final_round_ranks,
    run,
) in tqdm(all_experiments):
    experiment.do_experiment(
        None,
        "cookie_2x2",
        experiment_cookie,
        rounding_method=rounding_method,
        rounding_method_final=rounding_method_final,
        max_rank=max_rank,
        final_round_ranks=final_round_ranks,
        run=run,
    )
