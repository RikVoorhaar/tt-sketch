import time
from functools import reduce
from os.path import isfile
from typing import Callable, Optional

import pandas as pd
from tt_sketch.sketch import stream_sketch, orthogonal_sketch
from tt_sketch.tensor import Tensor
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
        self, input, name: str, experiment_func: Callable[..., float], **kwargs
    ):
        """Do an experiment, and store results+metadata.

        ``input`` should be something tensor-like that ``experiment_func``
        understands.

        ``experiment_func`` has signature ``(input, **kwargs) -> float``.
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
        time_zero = time.perf_counter()
        result = experiment_func(input, **kwargs)
        time_taken = time.perf_counter() - time_zero
        row["error"] = result
        row["time_taken"] = time_taken

        row_df = pd.DataFrame([row.values()], columns=row.keys())
        self.data = pd.concat([self.data, row_df], ignore_index=True)

        self.save()


def experiment_stream_sketch(
    input_tensor: Tensor,
    left_rank=None,
    right_rank=None,
    left_drm_type=None,
    right_drm_type=None,
    error_func: Optional[Callable[..., float]] = None,
    **kwargs,
) -> float:
    tt_sketched = stream_sketch(
        input_tensor,
        left_rank=left_rank,
        right_rank=right_rank,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
    )

    if error_func is not None:
        error = error_func(input_tensor, tt_sketched)
    else:
        error = tt_sketched.relative_error(input_tensor)
    return error


def experiment_orthogonal_sketch(
    input_tensor: Tensor,
    left_rank=None,
    right_rank=None,
    left_drm_type=None,
    right_drm_type=None,
    error_func: Optional[Callable[..., float]] = None,
    **kwargs,
) -> float:
    tt_sketched = orthogonal_sketch(
        input_tensor,
        left_rank=left_rank,
        right_rank=right_rank,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
    )

    if error_func is not None:
        error = error_func(input_tensor, tt_sketched)
    else:
        error = tt_sketched.relative_error(input_tensor)
    return error


def experiment_tt_svd(input_tensor: Tensor, rank=None, **kwargs):
    tt = tt_svd(input_tensor, rank=rank)
    error = tt.relative_error(input_tensor)
    return error
