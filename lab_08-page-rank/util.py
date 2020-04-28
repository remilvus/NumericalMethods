import numpy as np
from tqdm import tqdm

def load_graph(name):
    with open(name, "r") as file:
        content = [line for line in file.read().split("\n")]

    nodes, _, edges = content[2].split(" ")[2:5]
    print(f"nodes = {nodes} | edges = {edges}")
    nodes = int(nodes)
    adjacency = np.zeros(shape=(nodes, nodes), dtype=np.float32) #  scipy.sparse.csc_matrix
    print(adjacency.shape)
    id_mapper = dict()
    idx = 0
    print("loading...")
    for line in tqdm(content[6:], position=0):
        if len(line) == 0:
            continue
        source, target = [int(x) for x in line.split()]
        for node in [source, target]:
            if node not in id_mapper.keys():
                id_mapper[node] = idx
                idx += 1

        source = id_mapper[source]
        target = id_mapper[target]

        adjacency[target, source] = 1

    return adjacency

def power_iteration(A, max_iter=2000, epsilon=1e-15, log=False):
    x = np.random.random_sample(size=A.shape[0])
    x_prev = 0
    i = 0
    while i < max_iter and np.max(abs(x - x_prev)) > epsilon:
        x_prev = x
        i += 1
        x = A @ x
        eigenvalue = np.linalg.norm(x, ord=np.inf)
        x /= eigenvalue

    eigenvector = x / np.linalg.norm(x)
    if log:
        print(f"iterations = {i}")
    return eigenvalue, eigenvector


def page_rank(A, e, delta=1e-15):
    r = np.random.random_sample(size=A.shape[0])
    r_prev = 0
    while np.max(abs(r - r_prev)) > delta:
        r_prev = r
        r = A @ r
        d = np.max(r_prev) - np.max(r)
        r += d * e

    return r / np.max(r)

def scale_rows(A):
    divider = np.sum(A, axis=1)
    A[:, divider != 0]= A[:, divider != 0] / divider[None, divider != 0]
    return A

def add_random(A, e=None):
    if e is None:
        e = np.ones(A.shape[0])
    return A + e[:, None]