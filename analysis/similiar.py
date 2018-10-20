#!/usr/bin/env python3
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':

    user_rank = sparse.load_npz('src/user.rank.npz')

    sim = cosine_similarity(user_rank)
    print(np.histogram(sim[0], bins=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.7, 0.9, 1.1]))
    print(np.histogram(sim[1], bins=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.7, 0.9, 1.1]))
    print(np.histogram(sim[2], bins=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.7, 0.9, 1.1]))
    print(np.histogram(sim[3], bins=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.7, 0.9, 1.1]))
