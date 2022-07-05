# %%

import scipy.io

cookies_data = scipy.io.loadmat("data/cookies_matrices_2x2.mat")
cookies_data['b'].reshape(-1)
cookies_data['A'].reshape(-1)