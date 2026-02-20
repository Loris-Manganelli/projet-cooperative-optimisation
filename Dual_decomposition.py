from utils import grad_a
import numpy as np

def Solve_Lagrane(K_a,K_mm,y_a,sigma,nu,k,multiplier,Am):  
    """
    Compute the dual function for agent k in multplier
    
    :param K_a: list of kernel matrices for each agent N elements of size (n_a, m) where n_a is the number of data points for agent a and m is the number of training points
    :param K_mm: kernel matrix of size (m,m) between the training points Cov(training points)
    :param y_a: list of labels for each agent N elements of size (n_a, 1)
    :param sigma: global parameter of the problem
    :param nu: global parameter of the problem
    :param k: index of the agent for which we compute the dual function
    :param multiplier: list of multipliers for each agent N elements of size (m*N,1)
    :param Am: list of columns of constraints matrices Am[k] for each agent k of size (N*m,m)
    """
    N=len(K_a)
    #print("Shape of multiplier.T@Am[k]:", multiplier.T.shape)

    return np.linalg.solve(K_a[k].T @ K_a[k] + (1/N)*sigma**2*K_mm + nu/(N)*np.eye(K_mm.shape[0]), K_a[k].T @ y_a[k]-multiplier.T@Am[k])

import numpy as np

def build_constraint_matrices(adj, m):
    """
    adj : matrice d'adjacence numpy (N x N)
    m   : dimension de chaque variable xi
    
    Retour :
        Am : liste de N matrices numpy
            chaque matrice est de taille (E*m, m)
            où E = nombre de liens !!! Attention au graphes complet !!! (E = N*(N-1)/2)
    """
    
    N = adj.shape[0]
    
    # Extraire les liens (i < j)
    edges = [(i, j) for i in range(N) 
                      for j in range(i+1, N) 
                      if adj[i, j] != 0]
    
    E = len(edges)
    
    # Initialisation : N matrices nulles de taille (E*m, m)
    Am = [np.zeros((E*m, m)) for _ in range(N)]
    
    Id = np.eye(m)
    
    # Remplissage
    for e, (i, j) in enumerate(edges):
        
        row_start = e * m
        row_end   = (e + 1) * m
        
        # bloc +I sur le plus petit indice
        Am[i][row_start:row_end, :] = Id
        
        # bloc -I sur l'autre
        Am[j][row_start:row_end, :] = -Id
    
    return Am

def dual_decomposition(multiplier_0,K_a, K_mm, y_a, A, sigma, nu=1.0, max_iter=1000, lr=0.01):
    alpha = []
    multiplier = [multiplier_0]
    N=len(K_a)
    Am=build_constraint_matrices(A, K_mm.shape[0])


    for _ in range(max_iter):
        alpha_temp=[Solve_Lagrane(K_a,K_mm,y_a,sigma,nu,k,multiplier[-1],Am) for k in range(N)]
        alpha.append(np.array(alpha_temp))
        multiplier.append(multiplier[-1]+lr*(np.sum([Am[k]@alpha_temp[k] for k in range(N)],axis=0)))
    return alpha, multiplier

# print(build_constraint_matrices(np.array([[0,1,1],[1,0,1],[1,1,0]]), 2)[1]) #test

