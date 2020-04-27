import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile
from util import power_iteration, scale_rows

source = "adjacency_matrix.npy"
n = 15 # number of nodes

if isfile(source):
    adj_matrix = np.load(source).astype(np.float64)
else:
    adj_matrix = np.random.choice([0, 1], size=(n, n), p=[0.8, 0.2]).astype(np.float64)
    np.save(source[:-4], adj_matrix)

print("Adjacency matrix:\n", adj_matrix)

A = scale_rows(adj_matrix)

print("\nA (rounded):\n", np.round(A, 2))

eigenvalue, eigenvector = power_iteration(A)

assert np.allclose(eigenvector, A @ eigenvector / eigenvalue)

print("\nr (rounded):\n", np.round(eigenvector, 4))

print("checking other graphs...")
for i in range(5):
    A = np.random.choice([0, 1], size=(n, n), p=[0.6, 0.4]).astype(np.float64)
    empty = np.sum(A, axis=1) == 0
    A[empty, np.random.randint(0, n, size=np.sum(empty))] = 1
    A = scale_rows(A)
    eigenvalue, eigenvector = power_iteration(A)
    result = np.allclose(eigenvector,  A @ eigenvector / eigenvalue)
    if result:
        print("test passed", result)
    else:
        print(A)
        print("test failed", eigenvector)