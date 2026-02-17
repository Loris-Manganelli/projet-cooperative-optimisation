import numpy as np
from utils import grad_a

############ Gradient tracking algorithm #############
def GT(alpha_0, K_a, K_mm, y_a, W, sigma, nu=1.0, max_iter=1000, lr=0.01):
    alpha = []
    a = len(K_a)
    q = np.array([grad_a(alpha_0[k], k, K_a, K_mm, y_a, sigma, nu) for k in range(a)])
    alpha.append(alpha_0)
    for _ in range(max_iter):
        alpha_temp = alpha[-1]
        grad = np.array([grad_a(alpha_temp[k], k, K_a, K_mm, y_a, sigma, nu) for k in range(a)])
        alpha.append(W@alpha_temp - lr*q)
        alpha_temp = alpha[-1]
        grad_next = np.array([grad_a(alpha_temp[k], k, K_a, K_mm, y_a, sigma, nu) for k in range(a)])
        q=W@q+grad_next-grad
    return alpha
