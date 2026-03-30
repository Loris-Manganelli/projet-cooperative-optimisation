import numpy as np

def ADMM_Cloud(K_a, K_mm, y_a, A, sigma=0.5, nu=1.0, rho=1.0, max_iter=1000):
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

import tenseal as ts #alternative à OpenFHE

def ADMM_HE(K_a, K_mm, y_a, sigma=0.5, nu=1.0, rho=1.0, max_iter=10):
    N, m = len(K_a), K_mm.shape[0]

    #Définition des paramètres de chiffrement homomorphe
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
    )
    context.global_scale = 2**40 #facteur s de l'article
    context.generate_galois_keys()
    context.auto_relin = True
    context.auto_rescale = True

    #Matrices utiles par la suite
    H_inv = []
    KTy_enc = []
    for i in range(N):
        Hi = (sigma**2 / N) * K_mm + K_a[i].T @ K_a[i] + (nu / N) * np.eye(m)
        H_inv.append(np.linalg.inv(Hi + rho * np.eye(m)).tolist())
        KTy_enc.append(ts.ckks_vector(context, K_a[i].T @ y_a[i]))

    zeta_enc = ts.ckks_vector(context, [0.0] * m)
    lambd_enc = [ts.ckks_vector(context, [0.0] * m) for _ in range(N)]
    
    alpha_history = []

    for it in range(max_iter):
        z_enc = []
        for i in range(N):
            # rho * zeta
            # On utilise une multiplication scalaire simple
            term_rho_zeta = zeta_enc * rho
            
            # rhs = KTy + term_rho_zeta - lambd
            rhs = KTy_enc[i] + term_rho_zeta
            rhs = rhs - lambd_enc[i]
            
            # z_i = rhs @ H_inv
            z_i = rhs.matmul(H_inv[i])
            z_enc.append(z_i)

        # Consensus (Moyenne)
        zeta_val = np.mean([np.array(z.decrypt()) for z in z_enc], axis=0)
        zeta_enc = ts.ckks_vector(context, zeta_val.tolist())

        # Dual Update
        for i in range(N):
            z_val = np.array(z_enc[i].decrypt())
            #Dans l'article c'est marqué que l'agent à accès au zeta_i le concernant, évidemment c'est complètement con, parce que ça fait juste ADMM classique,
            #le cryptage ne sert à rien, le seul intérêt les itérations autre que lambda se font en crypté...
            l_val = np.array(lambd_enc[i].decrypt())
            l_val = l_val + rho * (z_val - zeta_val)
            lambd_enc[i] = ts.ckks_vector(context, l_val.tolist())

        # Historique
        alpha_history.append(np.array([np.array(z.decrypt()) for z in z_enc]))
        print(f"Iteration {it+1}/{max_iter} OK")

    return alpha_history