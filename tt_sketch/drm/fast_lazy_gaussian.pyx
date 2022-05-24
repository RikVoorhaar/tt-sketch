# cython: language_level=3
# cython: profile=True

import cython
import numpy as np
from libc.math cimport frexp, floor
cimport numpy as np
cimport scipy.special.cython_special
np.import_array()
ctypedef unsigned long long int uint64 
ctypedef unsigned int uint

@cython.boundscheck(False)
cpdef hash_int_c(uint64[:] vals):
    """Use simple hashing for generating random numbers from indices
    deterministically. See
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    cdef Py_ssize_t N = vals.shape[0]
    cdef uint shift1 = 30
    cdef uint shift2 = 27
    cdef uint shift3 = 31
    cdef uint64 mult1 = 0xBF58476D1CE4E5B9
    cdef uint64 mult2 = 0x94D049BB133111EB
    cdef uint64 add1 = 0X4BE98134A5976FD3

    cdef int i
    cdef uint64 r
    for i in range(N):
        r = vals[i]
        r = r + add1
        r ^= r >> shift1
        r *= mult1
        r ^= r >> shift2
        r *= mult2
        r ^= r >> shift3
        vals[i] = r

@cython.boundscheck(False)
cdef uint_to_normal(double[:] nums):
    """Turn hashed indices into normally distributed numbers"""
    # cdef double[:] nums = <double[:]>inds
    cdef Py_ssize_t N = nums.shape[0]
    cdef Py_ssize_t i
    cdef int e
    cdef double x
    for i in range(N): #prange here messes stuff up when parallelizing
        x = frexp(nums[i], &e) * 2 - 1
        x = scipy.special.cython_special.ndtri(x)
        nums[i] = x

@cython.boundscheck(False)
cpdef double[:] _inds_to_rand_double(
    uint64[:,:] indices, 
    uint64[:] shape,
    int rank_min,
    int rank_max,
    uint64 seed,
):  
    cdef uint64[:] indices_flat = np.copy(indices[0])
    cdef int N = indices.shape[1]
    cdef int n_dim = len(shape)
    cdef int rank = rank_max - rank_min

    cdef int prod = shape[0]
    cdef int i
    cdef int j
    for i in range(1,n_dim):
        for j in range(N):
            indices_flat[j] += indices[i,j] * prod
        prod *= shape[i]
    
    cdef uint64[:] size_salt = np.arange(rank_min, rank_max, dtype=np.uint64)
    hash_int_c(size_salt)
    # print(np.array(size_salt)) (if we increase rank then first r size salts
    # are still the same)
    
    for i in range(rank):
        size_salt[i] += seed

    cdef uint64[:,:] indices_flat_large = np.empty(
        (N, rank), dtype=np.uint64
    )
    
    for i in range(N):
        for j in range(rank):
            indices_flat_large[i,j] = indices_flat[i] + size_salt[j]
    
    cdef uint64[:] hashes =  np.reshape(indices_flat_large, -1)

    cdef uint64 SETBIT3 = 0x2000000000000000
    cdef uint64 SETBITS12 = 0x3FFFFFFFFFFFFFFF
    cdef uint64 h
    hash_int_c(hashes)
    for i in range(N*rank):
        # Set first three bits to 001
        # Avoids NaN values and zero values for float exp
        # Assures mantissa is between 0.5 and 1.0 
        h = hashes[i]
        h = (h | SETBIT3) & SETBITS12 
        hashes[i] = h
    cdef double[:] nums = np.frombuffer(hashes, dtype=np.float64)
    

    return nums


cdef double[:] _inds_to_normal(
    uint64[:,:] indices, 
    uint64[:] shape,
    int rank_min,
    int rank_max,
    uint64 seed,
):  
    cdef double[:] nums = _inds_to_rand_double(
        indices, shape, rank_min, rank_max, seed
    )
    uint_to_normal(nums)
    return nums

@cython.boundscheck(False)
cdef short[:] _inds_to_sparse_sign(
    uint64[:,:] indices, 
    uint64[:] shape,
    int rank,
    int nnz,
    uint64 seed,
):
    cdef Py_ssize_t N = indices.shape[1]
    cdef double[:] uniform_nums = _inds_to_rand_double(
        indices, shape, 0, nnz, seed
    )
    cdef short[:] sparse_sign = np.zeros((N*rank), dtype=np.int16)
    cdef int i
    cdef int j
    cdef int e
    cdef short temp
    cdef int rand_num

    for i in range(N):
        # populate matrix with rand signs
        for j in range(nnz):
            # Replace uniform num with mantissa, store exponent in &e
            uniform_nums[i*nnz+j] = frexp(uniform_nums[i*nnz+j], &e) * 2 - 1
            sparse_sign[i*rank+j] = (e % 2) * 2 - 1  # random sign
        
        # randomly permute
        for j in range(nnz):
            # random int in [j,rank)
            rand_num = <int>(uniform_nums[i*nnz+j]*(rank-j) + j) 
            temp = sparse_sign[i*rank+j]
            sparse_sign[i*rank+j] = sparse_sign[i*rank+rand_num]
            sparse_sign[i*rank+rand_num]=temp
    return sparse_sign

def inds_to_sparse_sign(
    indices,
    shape,
    rank,
    rank_min,
    rank_max,
    non_zero_per_row,
    seed
):
    """
    Converts a list of indices into the non-zero entries of a sparse sign matrix
    """
    indices = indices.astype(np.uint64)
    N = indices.shape[1]
    shape = np.array(shape,dtype=np.uint64)
    rank_min = int(rank_min)
    rank_max = int(rank_max)
    non_zero_per_row = int(non_zero_per_row)
    seed = np.mod(seed,2**63,dtype=np.uint64)
    nums_view = _inds_to_sparse_sign(
        indices, shape, rank, non_zero_per_row, seed
    )
    nums = np.array(nums_view).reshape(N,rank)
    nums = nums[:, rank_min:rank_max]
    return nums


def inds_to_normal(
    indices,
    shape,
    rank_min,
    rank_max,
    seed
):  
    """
    Converts a list of indices into the associated entries of gaussian matrix
    """
    indices = indices.astype(np.uint64)
    N = indices.shape[1]
    shape = np.array(shape,dtype=np.uint64)
    rank_min = int(rank_min)
    rank_max = int(rank_max)
    rank = rank_max - rank_min
    seed = np.mod(seed,2**63,dtype=np.uint64)
    nums_view = _inds_to_normal(indices, shape, rank_min, rank_max, seed)
    nums = np.array(nums_view)
    return nums.reshape(N,rank)