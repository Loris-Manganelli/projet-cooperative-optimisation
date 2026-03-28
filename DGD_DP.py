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
    W_hat = W - np.diag([W[i,i] for i in range(a)])
    #Random initialization of each component
    alpha_0 = np.random.rand(a,m)
    alpha.append(alpha_0)
    for k in range(max_iter):
        noise = np.random.laplace(0, nu_list[k], (a,m))
        alpha_temp = alpha[-1]
        chi = alpha_temp + noise
        grad = np.array([grad_a(alpha_temp[i], i, K_a, K_mm, y_a, sigma, nu) for i in range(a)])
        alpha_next = np.zeros((a,m))
        for i in range(a):
            alpha_next[i] = alpha_temp[i]+(gamma_list[k]*W_hat[i,:]@(chi-alpha_temp[i]))-lr_list[k]*grad[i]
        alpha.append(alpha_next)
    return alpha
