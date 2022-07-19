# %%
from turtle import right
import numpy as np
from tt_sketch.tensor import TuckerTensor, TensorTrain
from tt_sketch.sketch import orthogonal_sketch, stream_sketch
from tt_sketch.sketching_methods.tucker_sketch import (
    sketch_omega_tucker,
    sketch_psi_tucker,
)
from tt_sketch.drm import TensorTrainDRM
from tt_sketch.utils import matricize


def test_basic_tucker():
    shape = (10, 20, 15, 3)
    tuck = TuckerTensor.random(shape, 5)
    for i, U in enumerate(tuck.factors):
        assert U.shape == (tuck.rank[i], tuck.shape[i])
    assert tuck.to_numpy().shape == shape

    assert tuck.T.shape[::-1] == shape

    assert np.linalg.norm((1.5 * tuck - 0.5 * tuck - tuck).to_numpy()) < 1e-12

    permutation = tuple(range(len(shape)))[::-1]
    assert np.allclose(
        tuck.T.to_numpy(), tuck.to_numpy().transpose(permutation)
    )


test_basic_tucker()

# %%
shape = (10, 20, 15, 3)
tuck = TuckerTensor.random(shape, (2, 3, 4, 5))
left_rank = (4, 5, 6)
right_rank = (6, 7, 8)
left_drm = TensorTrainDRM(left_rank, shape, transpose=False)
right_drm = TensorTrainDRM(right_rank, shape, transpose=True)

left_sketches = list(left_drm.sketch_tucker(tuck))

right_sketches = list(right_drm.sketch_tucker(tuck))

mu = 1
left_sketch = left_sketches[mu]
right_sketch = right_sketches[mu]
left_sketch.shape, right_sketch.shape, np.prod(tuck.rank), left_sketch.shape[
    0
] * right_sketch.shape[0]


for mu in range(len(shape)):
    print('--')
    print(f"{mu=}")
    if mu > 0:
        left_sketch = left_sketches[mu-1]
        print(f"{left_sketch.shape=}")
    else:
        left_sketch = None
    if mu < len(shape) - 1:
        right_sketch = right_sketches[mu]
        print(f"{right_sketch.shape=}")
    else:
        right_sketch = None

    psi = sketch_psi_tucker(left_sketch, right_sketch, tensor=tuck, mu=mu)
    left_dim = left_sketch.shape[0] if left_sketch is not None else 1
    right_dim = right_sketch.shape[0] if right_sketch is not None else 1
    print("core shape", (left_dim, tuck.rank[mu], right_dim))
    # core_ord3 = tensor.core.reshape(left_dim, tensor.rank[mu], right_dim)
    print(f"{mu=}, {psi.shape=}")

print("_" * 80)
for mu, (left, right) in enumerate(zip(left_sketches, right_sketches)):
    # mu+=1
    omega = sketch_omega_tucker(left, right, tensor=tuck, mu=mu)
    print(f"{mu=}, {omega.shape=}")
# %%
tensor_sketched = orthogonal_sketch(
    tuck, 10, 15
)
tensor_sketched.relative_error(tuck)

# %%
tensor_sketched = stream_sketch(
    tuck, 15,10
)
tensor_sketched.relative_error(tuck)

# %%
tt = TensorTrain.random(shape, 10)
left_sketches = list(left_drm.sketch_tt(tt))
[S.shape for S in left_sketches]
# %%
