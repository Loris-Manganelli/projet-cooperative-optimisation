# Compute the gradient for agent k
def grad_a(alpha, k, K_a, K_mm, y_a, sigma, nu=1.0):
    K_a_k=K_a[k]
    y_a_k=y_a[k]
    grad_k = K_a_k.T @ (K_a_k @ alpha - y_a_k) + sigma**2*K_mm@alpha/5 + nu*alpha/5
    return grad_k
