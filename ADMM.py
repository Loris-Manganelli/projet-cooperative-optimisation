from utils import grad_a
import numpy as np

def Solve_Augmented_Lagrangian(K_a,K_mm,y_a,sigma,nu,k,Edge_variable,Incid,beta,dout):  
    """
    Compute the solution of the augmented Lagrangian for agent k in multplier
    :param K_a: list of kernel matrices for each agent N elements of size (n_a, m) where n_a is the number of data points for agent a and m is the number of training points
    :param K_mm: kernel matrix of size (m,m) between the training points Cov(training points)
    :param y_a: list of labels for each agent N elements of size (n_a, 1)
    :param sigma: global parameter of the problem 
    :param nu: global parameter of the problem
    :param k: index of the agent for which we compute the dual function
    :param Edge_variable: list of variables for each edge E elements of size (m,1)
    :param Incid: incidence matrix of the graph of size (E,N)
    :param beta: parameter of the augmented Lagrangian
    :param dout: degree out of each node of the graph
    """
    N=len(K_a)
    #print("Shape of multiplier.T@Am[k]:", multiplier.T.shape)
    return np.linalg.solve(K_a[k].T @ K_a[k] + (1/N)*sigma**2*K_mm + (beta*dout+nu/(N))*np.eye(K_mm.shape[0]), K_a[k].T @ y_a[k]-Incid[:,k].T@Edge_variable)


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

def ADMM(multiplier_0,egalizer_0,beta,K_a, K_mm, y_a, A, sigma, nu=1.0, max_iter=1000):
    alpha = []
    multiplier = [multiplier_0]
    egalizer = [egalizer_0]
    N=len(K_a)
    I=incidence(A)
    # degree out:
    dout = np.sum(A, axis=1)
    print([dout[k] for k in range(N)])
    
    for _ in range(max_iter):        
        alpha_temp=np.array([Solve_Augmented_Lagrangian(K_a,K_mm,y_a,sigma,nu,k,multiplier[-1]-beta*egalizer[-1],I,beta,dout[k]) for k in range(N)])
        alpha.append(alpha_temp)
        egalizer.append(I@alpha_temp/2)
        multiplier.append(multiplier[-1]+beta*(I@alpha_temp-egalizer[-1]))
    return alpha, multiplier

