import numpy as np

# Compute the gradient for agent k
def grad_a(alpha, k, K_a, K_mm, y_a, sigma, nu=1.0,N=5):
    K_a_k=K_a[k]
    y_a_k=y_a[k]
    grad_k = K_a_k.T @ (K_a_k @ alpha - y_a_k) + sigma**2*K_mm@alpha/N + nu*alpha/N
    return grad_k

# Compute the value of the objective function for agent k
def objective_a(alpha, k, K_a, K_mm, y_a, sigma, nu=1.0,N=5):
    K_a_k=K_a[k]
    y_a_k=y_a[k]
    obj_k = 1/2 * np.linalg.norm(K_a_k @ alpha - y_a_k)**2 + sigma**2/2*alpha.T @ K_mm @ alpha/N + nu/2*alpha.T @ alpha/N
    return obj_k

def incidence_oriented(adj):
    """
    adj : matrice d'adjacence numpy (N x N)

    Retour :
        Am : matrice d'"anti-incidence" orientée de taille (E, N)
        """
    
    N = adj.shape[0]
    
    # Extraire les liens (i < j)
    edges = [(i, j) for i in range(N) 
                      for j in range(i+1, N) 
                      if adj[i, j] != 0]
    
    E = len(edges)
    
    # Initialisation : N matrices nulles de taille (E*m, m)
    Am = np.zeros((E, N))    
    # Remplissage
    for e, (i, j) in enumerate(edges):
                
        # bloc +I sur le plus petit indice
        Am[e][i] = 1
        Am[e][j] = -1
    return Am

def incidence(adj):
    """
    adj : matrice d'adjacence numpy (N x N)

    Retour :
        Am : matrice d'incidence de taille (E, N)
        """
    
    N = adj.shape[0]
    
    # Extraire les liens (i < j)
    edges = [(i, j) for i in range(N) 
                      for j in range(i+1, N) 
                      if adj[i, j] != 0]
    
    E = len(edges)
    
    # Initialisation : N matrices nulles de taille (E*m, m)
    Am = np.zeros((E, N))    
    # Remplissage
    for e, (i, j) in enumerate(edges):
                
        # bloc +I sur le plus petit indice
        Am[e][i] = 1
        
        # bloc -I sur l'autre
        Am[e][j] = 1
    
    return Am
