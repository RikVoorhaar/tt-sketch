# %%
import numpy as np
from torch import rand

# %%
%timeit np.random.normal(size=(100_000_000))

# %%
bg = np.random.SFC64()
rng = np.random.Generator(bg)
%timeit rng.normal(size=(100_000_000))

# %%
bg = np.random.PCG64()
rng = np.random.Generator(bg)
%timeit rng.normal(size=(100_000_000))

# %%
from numpy.random import default_rng, SeedSequence, Generator, SFC64
import multiprocessing
import concurrent.futures
import numpy as np

from tt_sketch.utils import random_normal


random_normal(shape=(1_000_000,), seed=None)
# %%
seq = SeedSequence()
seq.generate_state(10)