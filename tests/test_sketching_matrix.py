import sys

import numpy as np
import pytest

# sys.path.append("..")

import itertools
from tt_sketch.drm import (
    ALL_DRM,
    SparseGaussianDRM,
    TensorTrainDRM,
    DenseGaussianDRM,
)
from tt_sketch.drm_base import CanSlice, CanIncreaseRank
from tt_sketch.sketching_methods.abstract_methods import (
    CansketchSparse,
    CansketchTT,
    CansketchCP,
)
from tt_sketch.tensor import (
    SparseTensor,
    TensorTrain,
    CPTensor,
    TensorSum,
    DenseTensor,
    TuckerTensor,
)
from tt_sketch.sketch_dispatch import SketchMethod
from tt_sketch.sketch import (
    stream_sketch,
    blocked_stream_sketch,
    orthogonal_sketch,
    hmt_sketch,
)

DRM_DICT = {drm_type.__name__: drm_type for drm_type in ALL_DRM}
AVAILABLE_SKETCHING_METHODS = list(SketchMethod.__members__.values())


def general_rank_increase(
    X_dense,
    X_tensor,
    left_rank,
    right_rank,
    seed,
    left_drm_type,
    right_drm_type,
):
    shape = X_tensor.shape
    d = len(shape)

    left_drm = left_drm_type(
        left_rank, transpose=False, shape=X_dense.shape, seed=seed
    )
    right_seed = np.mod(seed + hash(str(d)), 2**32)
    right_drm = right_drm_type(
        right_rank, transpose=True, shape=X_dense.shape, seed=right_seed
    )
    tt_sketch1 = stream_sketch(
        X_tensor,
        left_rank,
        right_rank,
        left_drm=left_drm,
        right_drm=right_drm,
        seed=seed,
    )

    left_rank_p1 = (1,) + left_rank
    right_rank_p1 = right_rank + (1,)

    new_left_rank = tuple(r + 2 for r in left_rank)
    new_right_rank = tuple(r + 3 for r in right_rank)

    tt_sketch3 = tt_sketch1.increase_rank(
        X_tensor, new_left_rank, new_right_rank
    )

    new_left_rank_p1 = (1,) + new_left_rank
    new_right_rank_p1 = new_right_rank + (1,)

    left_drm2 = left_drm.increase_rank(new_left_rank)
    right_drm2 = right_drm.increase_rank(new_right_rank)

    tt_sketch2 = stream_sketch(
        X_tensor,
        new_left_rank,
        new_right_rank,
        left_drm=left_drm2,
        right_drm=right_drm2,
        seed=seed,
    )

    for tt_sketch_other in [tt_sketch2, tt_sketch3]:

        for i, (Y1, Y2) in enumerate(
            zip(tt_sketch1.Psi_cores, tt_sketch_other.Psi_cores)
        ):
            assert np.allclose(Y1, Y2[: left_rank_p1[i], :, : right_rank_p1[i]])

        for i, Y in enumerate(tt_sketch_other.Psi_cores):
            assert Y.shape == (
                new_left_rank_p1[i],
                shape[i],
                new_right_rank_p1[i],
            )
        for i, Z in enumerate(tt_sketch_other.Omega_mats):
            assert Z.shape == (new_left_rank[i], new_right_rank[i])

    # Check slicing cancels out rank increase
    left_drm4 = left_drm2.slice(None, left_rank)
    right_drm4 = right_drm2.slice(None, right_rank)

    tt_sketch4 = stream_sketch(
        X_tensor,
        left_rank,
        right_rank,
        left_drm=left_drm4,
        right_drm=right_drm4,
        seed=seed,
    )
    for i, (Y1, Y2) in enumerate(
        zip(tt_sketch1.Psi_cores, tt_sketch4.Psi_cores)
    ):
        assert np.all(Y1 == Y2)

    for i, (Y1, Y2) in enumerate(
        zip(tt_sketch1.Omega_mats, tt_sketch4.Omega_mats)
    ):
        assert np.all(Y1 == Y2)


# TODO: Make particular tests for slicing, right now general_blocked_sketch does
# test slicing, but only indirectly


def general_blocked_sketch(X_tensor, seed, left_drm_type, right_drm_type):
    shape = X_tensor.shape
    n_ranks = len(shape) - 1

    left_rank = (9,) * n_ranks
    right_rank = (8,) * n_ranks
    sketched_tt, left_drm, right_drm = stream_sketch(
        X_tensor,
        left_rank,
        right_rank,
        seed=seed,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
        return_drm=True,
    )

    slice_left0 = (0,) * n_ranks
    slice_left1 = (3,) * n_ranks
    slice_left2 = (6,) * n_ranks
    slice_left3 = (9,) * n_ranks

    slice_right0 = (0,) * n_ranks
    slice_right1 = (4,) * n_ranks
    slice_right2 = (6,) * n_ranks
    slice_right3 = (8,) * n_ranks

    left_slices = [
        (slice_left0, slice_left1, slice_left2, slice_left3),
        (slice_left0, slice_left1, slice_left3),
        (slice_left0, slice_left3),
    ]

    right_slices = [
        (slice_right0, slice_right1, slice_right2, slice_right3),
        (slice_right0, slice_right2, slice_right3),
        (slice_right0, slice_right3),
    ]

    for ls in left_slices:
        for rs in right_slices:
            sketched_tt_block = blocked_stream_sketch(
                X_tensor, left_drm, right_drm, ls, rs
            )
            for Y1, Y2 in zip(
                sketched_tt_block.Psi_cores, sketched_tt.Psi_cores
            ):
                assert np.allclose(Y1, Y2)
            for Z1, Z2 in zip(
                sketched_tt_block.Omega_mats, sketched_tt.Omega_mats
            ):
                assert np.allclose(Z1, Z2)


def general_to_hmt(
    tensor,
    left_rank,
    right_rank,
    seed=None,
    left_drm_type=None,
    right_drm_type=None,
):
    return hmt_sketch(tensor, right_rank, seed, right_drm_type)


SKETCH_FUNC_DISPATCHER = {
    SketchMethod.hmt: general_to_hmt,
    SketchMethod.streaming: stream_sketch,
    SketchMethod.orthogonal: orthogonal_sketch,
}


def general_exact_recovery(
    X_dense,
    X_tensor,
    left_rank,
    right_rank,
    seed,
    left_drm_type,
    right_drm_type,
    sketch_method: SketchMethod,
):
    sketch_func = SKETCH_FUNC_DISPATCHER[sketch_method]
    tt_sketched = sketch_func(
        X_tensor,
        left_rank,
        right_rank,
        seed=seed,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
    )
    error = tt_sketched.error(X_dense)
    tt_sketched_dense = tt_sketched.to_numpy()
    assert error < 1e-9

    tt_sketched2 = sketch_func(
        X_tensor,
        left_rank,
        right_rank,
        seed=seed,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
    )
    # same seed should give exact same result. But in the case of parallel
    # execution, we can sum in different order giving same result only up to
    # machine epsilon
    assert tt_sketched2.error(tt_sketched) < 1e-9

    tt_sketched3 = sketch_func(
        X_tensor,
        left_rank,
        right_rank,
        seed=seed + 1,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
    )
    tt_sketched_dense3 = tt_sketched3.to_numpy()
    # different seed should give different result
    assert not np.all(tt_sketched_dense == tt_sketched_dense3)


sparse_drm_list = [
    drm_type.__name__
    for drm_type in ALL_DRM
    if issubclass(drm_type, CansketchSparse)
]
sparse_drm_pairs = [
    "|".join(s) for s in itertools.permutations(sparse_drm_list, 2)
]
sparse_drm_pairs.extend(["|".join([s, s]) for s in sparse_drm_list])
assert len(sparse_drm_pairs) > 0


@pytest.mark.parametrize("drm_types", sparse_drm_pairs)
@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("rank", [2, 5])
@pytest.mark.parametrize("sketch_method", AVAILABLE_SKETCHING_METHODS)
def test_exact_recovery_sparse(n_dims, rank, drm_types, sketch_method):
    seed = 180
    X_shape = tuple(range(9, 9 + n_dims))
    X_tt = TensorTrain.random(X_shape, rank)
    X = X_tt.to_numpy()

    left_rank = tuple(range(rank, rank + n_dims - 1))
    right_rank = tuple(range(rank + 1, rank + n_dims))

    X_sparse = X_tt.dense().to_sparse()

    drm_types = drm_types.split("|")
    left_drm = DRM_DICT[drm_types[0]]
    right_drm = DRM_DICT[drm_types[1]]
    general_exact_recovery(
        X,
        X_sparse,
        left_rank,
        right_rank,
        seed,
        left_drm,
        right_drm,
        sketch_method,
    )

    if issubclass(left_drm, CanSlice) and issubclass(right_drm, CanSlice):
        general_blocked_sketch(X_sparse, seed, left_drm, right_drm)

    if issubclass(left_drm, CanIncreaseRank) and issubclass(
        right_drm, CanIncreaseRank
    ):
        general_rank_increase(
            X, X_sparse, left_rank, right_rank, seed, left_drm, right_drm
        )


@pytest.mark.parametrize("direction", ["left", "right"])
def test_massive_oversample(direction):
    n_dims = 4
    rank = 5
    seed = 180
    X_shape = tuple(range(5, 5 + n_dims))
    X_tt = TensorTrain.random(X_shape, rank)
    X = X_tt.to_numpy()
    if direction == "left":
        left_rank = 100
        right_rank = 90
    if direction == "right":
        left_rank = 90
        right_rank = 100
    stt = stream_sketch(
        X_tt,
        left_rank,
        right_rank,
        seed=seed,
    )

    left_tt = TensorTrain(stt.C_cores(direction="left"))
    right_tt = TensorTrain(stt.C_cores(direction="right"))

    assert np.allclose(left_tt.to_numpy(), right_tt.to_numpy())
    assert left_tt.rank == stt.right_rank
    assert right_tt.rank == stt.left_rank


@pytest.mark.parametrize("n_dims", [2, 3, 4])
def test_sketch_dense(n_dims):
    seed = 179
    shape = (10,) * n_dims
    left_rank = 100
    right_rank = 200
    tensor_data = np.random.normal(size=shape)
    tensor = DenseTensor(tensor_data)
    stt = stream_sketch(
        tensor,
        left_rank,
        right_rank,
        seed=seed,
        left_drm_type=DenseGaussianDRM,
        right_drm_type=DenseGaussianDRM,
    )
    assert np.linalg.norm(stt.to_numpy() - tensor.to_numpy()) < 1e-9
    stt = orthogonal_sketch(
        tensor,
        left_rank,
        right_rank,
        seed=seed,
        left_drm_type=DenseGaussianDRM,
        right_drm_type=DenseGaussianDRM,
    )
    assert np.linalg.norm(stt.to_numpy() - tensor.to_numpy()) < 1e-9


@pytest.mark.parametrize("sketch_method", AVAILABLE_SKETCHING_METHODS)
def test_tensor_sum(sketch_method):
    seed = 179
    rank = 4
    n_dims = 4

    X_shape = tuple(range(7, 7 + n_dims))

    X1_tt = TensorTrain.random(X_shape, rank)
    X1 = X1_tt.to_numpy()

    left_rank = tuple(range(rank, rank + n_dims - 1))
    right_rank = tuple(range(rank + 1, rank + n_dims))

    X_sparse = X1_tt.dense().to_sparse()
    X_sparse_sum16 = X_sparse.split(16)
    X_sparse_sum2 = X_sparse.split(2)
    assert isinstance(X_sparse_sum2, TensorSum)
    assert isinstance(X_sparse_sum16, TensorSum)

    left_drm_type = TensorTrainDRM
    right_drm_type = SparseGaussianDRM

    # Recovery should work for TensorSum
    general_exact_recovery(
        X1,
        X_sparse_sum16,
        left_rank,
        right_rank,
        seed,
        left_drm_type,
        right_drm_type,
        sketch_method,
    )
    general_exact_recovery(
        X1,
        X_sparse_sum2,
        left_rank,
        right_rank,
        seed,
        left_drm_type,
        right_drm_type,
        sketch_method,
    )
    # Sketching is linear, so doing it as (parallel) sum should give same result
    stt1 = stream_sketch(X_sparse_sum16, left_rank, right_rank, seed)
    stt3 = stream_sketch(X_sparse_sum2, left_rank, right_rank, seed)
    stt2 = stream_sketch(X_sparse, left_rank, right_rank, seed)
    for Y1, Y2, Y3 in zip(stt1.Psi_cores, stt2.Psi_cores, stt3.Psi_cores):
        assert np.allclose(Y1, Y2)
        assert np.allclose(Y1, Y3)
    for Y1, Y2, Y3 in zip(stt1.Omega_mats, stt2.Omega_mats, stt3.Omega_mats):
        assert np.allclose(Y1, Y2)
        assert np.allclose(Y1, Y3)

    X2_tt = TensorTrain.random(X_shape, rank)
    X2 = X1 + X2_tt.to_numpy()

    X1_plus_X2 = X_sparse + X2_tt
    left_drm_type = TensorTrainDRM
    right_drm_type = TensorTrainDRM
    left_rank_bigger = tuple(r + 10 for r in left_rank)
    right_rank_bigger = tuple(r + 10 for r in right_rank)
    stt4 = stream_sketch(
        X1_plus_X2,
        left_rank_bigger,
        right_rank_bigger,
        seed,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
    )
    assert stt4.to_tt().error(X2) < 1e-8

    # test the user-friendlty way of updating sketch
    stt5 = stream_sketch(
        X_sparse,
        left_rank_bigger,
        right_rank_bigger,
        seed,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
    )
    stt6 = stt5 + X2_tt
    assert stt6.error(X2) < 1e-8


tt_drm_list = [
    drm_type.__name__
    for drm_type in ALL_DRM
    if issubclass(drm_type, CansketchTT)
]
tt_drm_pairs = ["|".join(s) for s in itertools.permutations(tt_drm_list, 2)]
tt_drm_pairs.extend(["|".join([s, s]) for s in tt_drm_list])
assert len(tt_drm_pairs) > 0


@pytest.mark.parametrize("drm_types", tt_drm_pairs)
@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("sketch_method", AVAILABLE_SKETCHING_METHODS)
def test_exact_recovery_tt(n_dims, rank, drm_types, sketch_method):
    seed = 180
    X_shape = tuple(range(10, 10 + n_dims))
    X_tt = TensorTrain.random(X_shape, rank, seed=seed)
    X = X_tt.to_numpy()

    left_rank = tuple(range(rank, rank + n_dims - 1))
    right_rank = tuple(range(rank + 1, rank + n_dims))

    drm_types = drm_types.split("|")
    left_drm_type = DRM_DICT[drm_types[0]]
    right_drm_type = DRM_DICT[drm_types[1]]
    general_exact_recovery(
        X,
        X_tt,
        left_rank,
        right_rank,
        seed,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
        sketch_method=sketch_method,
    )

    if issubclass(left_drm_type, CanSlice) and issubclass(
        right_drm_type, CanSlice
    ):
        general_blocked_sketch(
            X_tt,
            seed,
            left_drm_type=left_drm_type,
            right_drm_type=right_drm_type,
        )

    if issubclass(left_drm_type, CanIncreaseRank) and issubclass(
        right_drm_type, CanIncreaseRank
    ):
        general_rank_increase(
            X,
            X_tt,
            left_rank,
            right_rank,
            seed,
            left_drm_type=left_drm_type,
            right_drm_type=right_drm_type,
        )


cp_drm_list = [
    drm_type.__name__
    for drm_type in ALL_DRM
    if issubclass(drm_type, CansketchCP)
]
cp_drm_pairs = ["|".join(s) for s in itertools.permutations(cp_drm_list, 2)]
cp_drm_pairs.extend(["|".join([s, s]) for s in cp_drm_list])
assert len(cp_drm_pairs) > 0


@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("sketch_method", AVAILABLE_SKETCHING_METHODS)
def test_exact_recovery_tucker(n_dims, rank, sketch_method):
    seed = 180
    X_shape = tuple(range(10, 10 + n_dims))
    X_tucker = TuckerTensor.random(X_shape, rank, seed=seed)
    X = X_tucker.to_numpy()

    left_rank = tuple(range(rank, rank + n_dims - 1))
    right_rank = tuple(range(rank + 1, rank + n_dims))

    general_exact_recovery(
        X,
        X_tucker,
        left_rank,
        right_rank,
        seed,
        left_drm_type=TensorTrainDRM,
        right_drm_type=TensorTrainDRM,
        sketch_method=sketch_method,
    )


@pytest.mark.parametrize("drm_types", cp_drm_pairs)
@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("sketch_method", AVAILABLE_SKETCHING_METHODS)
def test_exact_recovery_cp(n_dims, rank, drm_types, sketch_method):
    seed = 180
    X_shape = tuple(range(10, 10 + n_dims))
    X_cp = CPTensor.random(X_shape, rank, seed=seed)
    X = X_cp.to_numpy()

    left_rank = tuple(range(rank, rank + n_dims - 1))
    right_rank = tuple(range(rank + 1, rank + n_dims))

    drm_types = drm_types.split("|")
    left_drm_type = DRM_DICT[drm_types[0]]
    right_drm_type = DRM_DICT[drm_types[1]]
    general_exact_recovery(
        X,
        X_cp,
        left_rank,
        right_rank,
        seed,
        left_drm_type=left_drm_type,
        right_drm_type=right_drm_type,
        sketch_method=sketch_method,
    )

    if issubclass(left_drm_type, CanSlice) and issubclass(
        right_drm_type, CanSlice
    ):
        general_blocked_sketch(
            X_cp,
            seed,
            left_drm_type=left_drm_type,
            right_drm_type=right_drm_type,
        )

    if issubclass(left_drm_type, CanIncreaseRank) and issubclass(
        right_drm_type, CanIncreaseRank
    ):
        general_rank_increase(
            X,
            X_cp,
            left_rank,
            right_rank,
            seed,
            left_drm_type=left_drm_type,
            right_drm_type=right_drm_type,
        )


# TODO: do this test for all DRM's
@pytest.mark.parametrize("n_dims", [2, 4])
@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize("right_rank_oversample", [1, 2])
def test_sketch_linearity(n_dims, rank, right_rank_oversample):
    N = 1000
    shape = (10,) * n_dims
    left_rank = (rank,) * (n_dims - 1)
    right_rank = (rank + right_rank_oversample,) * (n_dims - 1)
    tot_dim = np.prod(shape)
    N = min(N, tot_dim)

    inds_flat = np.random.choice(tot_dim, size=N)
    inds = np.stack(np.unravel_index(inds_flat, shape))

    entries = np.random.normal(size=N)

    X_sparse = SparseTensor(shape, inds, entries)
    X_sparse1 = SparseTensor(shape, inds[:, : N // 2], entries[: N // 2])
    X_sparse2 = SparseTensor(shape, inds[:, N // 2 :], entries[N // 2 :])
    seed = 179
    tt_sketched = stream_sketch(X_sparse, left_rank, right_rank, seed=seed)
    tt_sketched1 = stream_sketch(X_sparse1, left_rank, right_rank, seed=seed)
    tt_sketched2 = stream_sketch(X_sparse2, left_rank, right_rank, seed=seed)
    for mu in range(n_dims):
        assert np.allclose(
            tt_sketched1.Psi_cores[mu] + tt_sketched2.Psi_cores[mu],
            tt_sketched.Psi_cores[mu],
        )
    for mu in range(n_dims - 1):
        assert np.allclose(
            tt_sketched1.Omega_mats[mu] + tt_sketched2.Omega_mats[mu],
            tt_sketched.Omega_mats[mu],
        )


@pytest.mark.parametrize("n_dims", [3, 4])
@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize("bigger_side", ["left", "right"])
def test_tt_cores_contraction(n_dims, rank, bigger_side):
    seed = 180
    X_shape = tuple(range(5, 5 + n_dims))
    rank = tuple(range(rank, rank + n_dims - 1)[::-1])
    tt = TensorTrain.random(X_shape, rank)

    if bigger_side == "left":
        left_rank = [r + 2 for r in tt.rank]
        right_rank = [r + 1 for r in tt.rank]
    if bigger_side == "right":
        left_rank = [r + 1 for r in tt.rank]
        right_rank = [r + 2 for r in tt.rank]

    stt = stream_sketch(tt, left_rank, right_rank, seed=seed)

    assert (np.linalg.norm(tt.to_numpy() - stt.to_tt().to_numpy())) < 1e-8
    left_tt = TensorTrain(stt.C_cores(direction="left"))
    right_tt = TensorTrain(stt.C_cores(direction="right"))

    assert np.allclose(left_tt.to_numpy(), right_tt.to_numpy())
    assert left_tt.rank == stt.right_rank
    assert right_tt.rank == stt.left_rank


def test_defaults_sketch():
    """Test that the default settings are correct."""
    shape = (8, 9, 10)
    r = 10
    test_tensors = []

    test_tensor_tt = TensorTrain.random(shape, r)
    test_tensors.append(test_tensor_tt)

    test_tensor_sparse = SparseTensor.random(shape, nnz=10)
    test_tensors.append(test_tensor_sparse)

    test_tensor_tucker = TuckerTensor.random(shape, 2)
    test_tensors.append(test_tensor_tucker)

    test_tensor_dense = test_tensor_sparse.dense()
    test_tensors.append(test_tensor_dense)

    test_tensor_sum = test_tensor_tt + test_tensor_sparse
    test_tensors.append(test_tensor_sum)

    test_tensor_cp = CPTensor.random(shape, 5)
    test_tensors.append(test_tensor_cp)

    for test_tensor in test_tensors:
        sketched = hmt_sketch(test_tensor, r + 1)
        assert sketched.error(test_tensor) < 1e-8
        sketched = orthogonal_sketch(
            test_tensor, left_rank=r + 1, right_rank=r + 2
        )
        assert sketched.error(test_tensor) < 1e-8
        sketched = stream_sketch(test_tensor, left_rank=r + 1, right_rank=r + 2)
        assert sketched.error(test_tensor) < 1e-8


def test_sketched_tt_arithmetic():
    n_dims = 3
    rank = tuple(range(3, 3 + n_dims - 1))[::-1]
    right_rank = tuple(r + 1 for r in rank)
    X_shape = tuple(range(9, 9 + n_dims))
    X_tt = TensorTrain.random(X_shape, rank)
    tts1 = stream_sketch(X_tt, left_rank=rank, right_rank=right_rank)
    tt1 = tts1.to_tt()

    tts2 = tts1 * 2
    tt2 = tts2.to_tt()

    assert tt2.error(tt1 * 2) < 1e-10

    tts3 = tts1.T
    tt3 = tts3.to_tt()
    assert tt3.error(tt1.T) < 1e-10