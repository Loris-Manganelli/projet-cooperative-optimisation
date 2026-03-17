import numpy as np
from utils import grad_a
############ DGD-DP algorithm #############

# DGD-DP algorithm
def DGD_DP(K_a, K_mm, y_a, W, sigma, gamma_list, nu_list, lr_list, nu=1.0, max_iter=1000):
    np.random.seed(42)
    #no longer need of \alpha_0 because of random initialization
    alpha = []
    a = len(K_a)
    m = len(K_mm)
    alpha_0 = np.zeros((a,m))
    W = W - np.eye(a)/a
    #Random initialization of each component
    for k in range(a):
        alpha_0[k, :] = np.random.randn(m)
    alpha.append(alpha_0)
    for k in range(max_iter):
        noise = np.random.laplace(0, nu_list[k], (a,m))
        alpha_temp = alpha[-1]
        grad = np.array([grad_a(alpha_temp[k], k, K_a, K_mm, y_a, sigma, nu) for k in range(a)])
        alpha_next = np.zeros((a,m))
        alpha_temp+noise
        for i in range(a):
            alpha_next[i] = alpha_temp[i]+(gamma_list[k]*W@(noise))[i]-lr_list[k]*grad[i]
        alpha.append(alpha_next)
    return alpha
