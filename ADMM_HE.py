import numpy as np

def ADMM_HE(K_a, K_mm, y_a, A, sigma=0.5, nu=1.0, rho=1.0, max_iter=1000, encryption=False):
    """
    Standard Global Consensus ADMM.
    """
    if encryption:
        #using article notation for the mapping
        s=10**9 #scaling (pour augmenter le précision)
        q =10**20 + 7 #modulo (s'il n'est pas énorme, on ne peut pas faire le unmapping (comment retrouver un nombre juste avec un rest modulo q)), en fait ici le rest modulo q = le nombre (puisque q énorme)
        def mapping(x):
            return (s*x)%q
        def unmapping(x):
            #Gere les alpha_i négatifs 
            return np.where(x > q/2, x - q, x) / s
        # On simule les clés par des vecteurs de taille m (masques additifs modulo q)
        sk_0 = np.random.rand(K_mm.shape[0]) * q # Clé de l'opérateur
        sk_agents = [np.random.rand(K_mm.shape[0]) * q for _ in range(len(K_a))] # Clés des agents
        
        # L'opérateur crée publiquement les Switching Keys : (sk_0 - sk_i)
        SWK = [(sk_0 - sk_agents[i]) % q for i in range(len(K_a))]
        #Dans la vraie vie, les voisins de i reçoivent SWK[i] et font le calcul, mais i ne reçoit pas SWK[i].
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
        Hi = (sigma**2 / N) * K_mm + K_a[i].T @ K_a[i] + (nu / N) * np.eye(m)
        Hi_inv = np.linalg.inv(Hi + rho * np.eye(m))
        H_inv.append(Hi_inv)
    alpha_history = []
    for _ in range(max_iter):
        #z udpdate (which is in the case just alpha_i)
        for i in range(N):
            rhs = K_a[i].T @ y_a[i] + rho * zeta - lambd[i]
            z[i] = H_inv[i] @ rhs
        #zeta update (consensus, just average everything)
        val_sum = np.zeros(m)
        for i in range(N):
                val_sum += z[i]
        zeta = val_sum / N
        #dual variable update
        for i in range(N):
            lambd[i] += rho * (z[i] - zeta)
        # Record the local estimates for the current iteration
        if encryption:
            alpha_iter = []
            for i in range(N):
                z_mapped = mapping(z[i])
                z_mapped_0 = (z_mapped + sk_0) % q
                # Ici on imagine que le voisin j reçoit z_mapped_0 et la Switching Key destinée à i (SWK[i]) (i ne peut pas avoir accès à SWK[i]).
                # Le voisin j calcule le changement de clé, sans jamais lire le message en clair !
                # Mathématiquement : (M + sk_0) - (sk_0 - sk_i) = M + sk_i
                z_mapped_i = (z_mapped_0 - SWK[i]) % q
                z_decrypted_mapped = (z_mapped_i - sk_agents[i]) % q
                z_final = unmapping(z_decrypted_mapped)
                
                alpha_iter.append(z_final)
                
            alpha_history.append(np.array(alpha_iter))
        else:
            alpha_history.append(np.array([z[i].copy() for i in range(N)]))

    return alpha_history