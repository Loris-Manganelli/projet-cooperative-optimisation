from utils import grad_a


def dual_decomposition(alpha_0, K_a, K_mm, y_a, A, sigma, nu=1.0, max_iter=1000, lr=0.01):
    alpha = []
    alpha.append(alpha_0)

