import numpy as np

def ADMM_HE(K_a, K_mm, y_a, A, sigma=0.5, nu=1.0, rho=1.0, max_iter=1000):
    """
    Standard Global Consensus ADMM.
    """
    N = len(K_a)
    m = K_mm.shape[0]
    #Initialize local variables z and dual variables lambda
    z = [np.zeros(m) for _ in range(N)]
    lambd = [np.zeros(m) for _ in range(N)]
    #Initialize (only one to ensure consensus) shared global consensus variable
    zeta = np.zeros(m)
    #Use the form of the article with H_i
    H_inv = []
    for i in range(N):
        # Local Hessian share of the global objective:
        #Le H_i est celui de l'article que l'on a identifié à notre problème
        Hi = (sigma**2 / N) * K_mm + K_a[i].T @ K_a[i] + (nu / N) * np.eye(m)
        Hi_inv = np.linalg.inv(Hi + rho * np.eye(m))
        H_inv.append(Hi_inv)
    alpha_history = []
    for iter in range(max_iter):
        for i in range(N):
            rhs = K_a[i].T @ y_a[i] + rho * zeta - lambd[i]
            z[i] = H_inv[i] @ rhs
        #zeta update (consensus, just average everything)
        val_sum = np.zeros(m)
        for i in range(N):
            val_sum += z[i]
        zeta = (val_sum / N)
        #dual variable update
        for i in range(N):
            lambd[i] += (rho * (z[i] - zeta))
        # Record the local estimates for the current iteration
        alpha_history.append(np.array([z[i].copy() for i in range(N)]))

    return alpha_history