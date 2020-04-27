import numpy as np

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

def scale_rows(A):
    divider = np.sum(A, axis=1)
    A[divider != 0, :] = A[divider != 0, :] / divider[divider != 0, None]
    return A
