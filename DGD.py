import numpy as np
############ DGD algorithm #############

# Compute the gradient for agent k
def grad_a(alpha, k, K_a, K_mm, y_a, sigma, nu=1.0):
    K_a_k=K_a[k]
    y_a_k=y_a[k]
    grad_k = K_a_k.T @ (K_a_k @ alpha - y_a_k) + sigma**2*K_mm@alpha/5 + nu*alpha/5
    return grad_k

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

