# %%
"""I need to figure out how to properly do parallel processing with some kind of
reduction step. Otherwise memory overhead of `parallel_sparse_sketch` is
unacceptable. Not sure we should use joblib parallel for this, better just use
the STL multiprocessing. 

First we need a good test problem..."""
from multiprocessing import Pool
import numpy as np

n, m = (10, 1000)
N = 10


def foo(i, j, d="sdfsdf"):
    np.random.seed(i)
    A = np.random.normal(size=(n, m))
    mean = np.mean(A, axis=1)
    return mean


def do_parallel():
    target = np.zeros(n)

    def my_callback(args):
        nonlocal target
        target += args[0]
    def error_callback(e):
        print(e)
        raise RuntimeError()
    # pool = Pool()
    with Pool() as pool:
        for i in range(N):
            try:
                pool.apply_async(foo, (i), callback=my_callback, error_callback=error_callback)
            except RuntimeError as e:
                print("runtime error")
                raise
        pool.close()
        pool.join()
    # assert np.linalg.norm(target) > 0
    return target


if __name__ == "__main__":
    # for i in range(10):
    target1 = do_parallel()

    # target2 = np.stack([foo(i,"s")[0] for i in range(N)]).sum(axis=0)

    print(np.linalg.norm(target1 - target2), np.linalg.norm(target1))
