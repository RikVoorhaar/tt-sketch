# %%
from tt_sketch.tt_gmres import tt_sum_gmres
from tt_sketch.tensor import TensorTrain
from cookie_problem import prepare_cookie_problem

map_sum, B, precond_map = prepare_cookie_problem(10, 4)

x0 = TensorTrain.zero(shape=map_sum.in_shape, rank=10)
max_rank = 20
result, history = tt_sum_gmres(
    A=map_sum,
    b=B,
    precond=precond_map,
    x0=x0,
    max_rank=max_rank,
    final_round_rank=100,
    maxiter=50,
    tolerance=0,
    symmetric=True,
    rounding_method="pairwise",
    rounding_method_final="pairwise",
    save_basis=True,
    verbose=True,
)
