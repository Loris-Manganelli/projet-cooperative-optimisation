import numpy as np
from utils import grad_a

############ DGD algorithm #############
# DGD algorithm
def DGD(alpha_0, K_a, K_mm, y_a, W, sigma, nu=1.0, max_iter=1000, lr=0.01):
    alpha = []
    alpha.append(alpha_0)
    a = len(K_a)
    for _ in range(max_iter):
        alpha_temp = alpha[-1]
        grad = np.array([grad_a(alpha_temp[k], k, K_a, K_mm, y_a, sigma, nu) for k in range(a)])
        alpha.append(W@alpha_temp - lr*grad)
    return alpha

