import numpy as np
from os.path import isfile
from util import scale_rows, add_random, page_rank, load_graph

source = "adjacency_matrix.npy"
n = 15  # number of nodes

if isfile(source):
    adj_matrix = np.load(source).astype(np.float64)
else:
    adj_matrix = np.random.choice([0, 1], size=(n, n), p=[0.8, 0.2]).astype(np.float64)
    np.save(source[:-4], adj_matrix)

e = np.ones(adj_matrix.shape[0])
adj_matrix = add_random(adj_matrix, e=e)
A = scale_rows(adj_matrix)
r = page_rank(A, e=np.ones(adj_matrix.shape[0]))

print("\nresult for small test. r:\n", np.round(r, 4))

print("checking SNAP...")
for filename in ["p2p-Gnutella08.txt", "Wiki-Vote.txt"]:
    print(filename)
    adjacency_matrix = load_graph(filename)
    n = adjacency_matrix.shape[0]

    for e_val in [1, 0.5, 0]:  # different values of d are equivalent to changing values in e
        e = np.full(n, fill_value=e_val, dtype=np.float32)
        print(f"d*e values = [{e_val}/{adjacency_matrix.shape[0]}, ...]")
        adj_matrix = add_random(adjacency_matrix, e=e)
        A = scale_rows(adj_matrix)
        r = page_rank(A, e=e, delta=1e-8)
        print(f"r min ={np.min(r)}, r max = {np.max(r)}\n")

    print("e[i]=0 except e[0]=1")
    e = np.zeros(n, dtype=np.float32)
    e[0] = 1
    adj_matrix = add_random(adjacency_matrix, e=e)
    A = scale_rows(adj_matrix)
    r = page_rank(A, e=e, delta=1e-8)
    print(f"r min ={np.min(r)}, r max = {np.max(r)}\n")


