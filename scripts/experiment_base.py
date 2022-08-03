import time
from functools import reduce
from os.path import isfile
from typing import Callable, Dict, Optional, Any

import pandas as pd
from tt_sketch.sketch import stream_sketch, orthogonal_sketch, hmt_sketch
from tt_sketch.tensor import Tensor, TensorTrain
from tt_sketch.tt_svd import tt_svd


class Experiment:
    data: pd.DataFrame
    filename: str
    autosave: bool

    def __init__(
        self,
        filename: str,
        data: Optional[pd.DataFrame] = None,
        force_overwrite: bool = False,
    ):
        self.filename = filename
        if data is None:
            if isfile(filename) and not force_overwrite:
                data = pd.read_csv(filename)
            else:
                data = pd.DataFrame(columns=["name", "error", "time_taken"])

        self.data = data

    @classmethod
    def load(cls, filename):
        data = pd.read_csv(filename)
        return cls(filename, data)

    def save(self):
        self.data.to_csv(self.filename)

    def check_experiment_done(self, conds):
        masks = []
        for column, value in conds.items():
            try:
                masks.append(self.data[column] == value)
            except KeyError:
                return False
        mask = reduce(lambda x, y: x & y, masks)
        return mask.sum() > 0

    def do_experiment(
        self,
        input,
        name: str,
        experiment_func: Callable[..., Dict[str, Any]],
        **kwargs,
    ):
        """Do an experiment, and store results+metadata.

        ``input`` should be something tensor-like that ``experiment_func``
        understands.

        ``experiment_func`` has signature ``(input, **kwargs) -> Dict[str, Any]``.
        """

        row = dict()
        for key, value in kwargs.items():
            try:  # Check if values is numeric
                float(value)
            except (ValueError, TypeError):
                if not isinstance(value, str):
                    try:
                        value = value.__name__
                    except AttributeError:
                        value = str(value)
            row[key] = value
        row["name"] = name
        if self.check_experiment_done(row):
            return

        # Do the experiment
        result = experiment_func(input, **kwargs)
        for key, value in result.items():
            row[key] = value
        # row["error"] = result

        row_df = pd.DataFrame([row.values()], columns=row.keys())
        self.data = pd.concat([self.data, row_df], ignore_index=True)

        self.save()


def experiment_stream_sketch(
    input_tensor: Tensor,
    left_rank=None,
    right_rank=None,
    left_drm_type=None,
    right_drm_type=None,
    error_func=None,
    recompression_rank=None,
    **kwargs,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    tt_sketched = stream_sketch(
        input_tensor,
        left_rank=left_rank,
        right_rank=right_rank,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
    )
    if recompression_rank is not None:
        tt_sketched = tt_sketched.to_tt().round(max_rank=recompression_rank)
        assert max(tt_sketched.rank) <= recompression_rank
    time_taken = time.perf_counter() - start_time

    if error_func is not None:
        error = error_func(tt_sketched, input_tensor)
    else:
        error = tt_sketched.error(input_tensor, relative=True)
    return {"error": error, "time_taken": time_taken}


def experiment_orthogonal_sketch(
    input_tensor: Tensor,
    left_rank=None,
    right_rank=None,
    left_drm_type=None,
    right_drm_type=None,
    error_func=None,
    recompression_rank=None,
    **kwargs,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    tt_sketched = orthogonal_sketch(
        input_tensor,
        left_rank=left_rank,
        right_rank=right_rank,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
    )
    if recompression_rank is not None:
        tt_sketched = tt_sketched.round(max_rank=recompression_rank)
    time_taken = time.perf_counter() - start_time

    if error_func is not None:
        error = error_func(tt_sketched, input_tensor)
    else:
        error = tt_sketched.error(input_tensor, relative=True)
    return {"error": error, "time_taken": time_taken}


def experiment_hmt_sketch(
    input_tensor: Tensor,
    rank=None,
    drm_type=None,
    error_func=None,
    recompression_rank=None,
    **kwargs,
):
    start_time = time.perf_counter()
    tt_sketched = hmt_sketch(input_tensor, rank=rank, drm_type=drm_type)
    if recompression_rank is not None:
        tt_sketched = tt_sketched.round(max_rank=recompression_rank)
    time_taken = time.perf_counter() - start_time
    if error_func is not None:
        error = error_func(tt_sketched, input_tensor)
    else:
        error = tt_sketched.error(input_tensor, relative=True)
    return {"error": error, "time_taken": time_taken}


def experiment_tt_svd(
    input_tensor: Tensor, rank=None, error_func=None, **kwargs
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    tt = tt_svd(input_tensor, rank=rank)
    time_taken = time.perf_counter() - start_time
    if error_func is not None:
        error = error_func(tt, input_tensor)
    else:
        error = tt.error(input_tensor, relative=True)
    return {"error": error, "time_taken": time_taken}

def experiment_tt_round(
    input_tensor: TensorTrain, rank=None, error_func=None, **kwargs
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    tt = input_tensor.round(max_rank=rank)
    time_taken = time.perf_counter() - start_time
    if error_func is not None:
        error = error_func(tt, input_tensor)
    else:
        error = tt.error(input_tensor, relative=True)
    return {"error": error, "time_taken": time_taken}
