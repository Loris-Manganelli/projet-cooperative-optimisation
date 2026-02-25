from utils import grad_a
import numpy as np
import utils as ut
def Solve_Augmented_Lagrangian(K_a,K_mm,y_a,sigma,nu,k,multiplier,egalizer,Incid,Anti_Incid,beta,dout):  
    """
    Compute the solution of the augmented Lagrangian for agent k in multplier
    :param K_a: list of kernel matrices for each agent N elements of size (n_a, m) where n_a is the number of data points for agent a and m is the number of training points
    :param K_mm: kernel matrix of size (m,m) between the training points Cov(training points)
    :param y_a: list of labels for each agent N elements of size (n_a, 1)
    :param sigma: global parameter of the problem 
    :param nu: global parameter of the problem
    :param k: index of the agent for which we compute the dual function
    :param multiplier: list of multipliers for each agent N elements of size (E,m)
    :param egalizer: list of egalizers for each agent N elements of size (E,m)
    :param Incid: incidence matrix of the graph of size (E,N)
    :param Anti_Incid: anti-incidence matrix of the graph of size (E,N)
    :param beta: parameter of the augmented Lagrangian
    :param dout: degree out of each node of the graph
    """
    N=len(K_a)
    #print("Shape of multiplier.T@Am[k]:", multiplier.T.shape)
    return np.linalg.solve(K_a[k].T @ K_a[k] + (1/N)*sigma**2*K_mm + (beta*dout+nu/(N))*np.eye(K_mm.shape[0]), K_a[k].T @ y_a[k]+beta*Incid[:,k].T @ egalizer-Anti_Incid[:,k].T @ multiplier)





def ADMM(multiplier_0,egalizer_0,beta,K_a, K_mm, y_a, A, sigma, nu=1.0, max_iter=1000):
    alpha = []
    multiplier = [multiplier_0]
    egalizer = [egalizer_0]
    N=len(K_a)
    Incid=ut.incidence(A)
    Anti_Incid=ut.incidence_oriented(A)
    # degree out:
    dout = np.sum(A, axis=1)
    
    for _ in range(max_iter):        
        alpha_temp=np.array([Solve_Augmented_Lagrangian(K_a,K_mm,y_a,sigma,nu,k,multiplier[-1],egalizer[-1],Incid,Anti_Incid,beta,dout[k]) for k in range(N)])
        alpha.append(alpha_temp)
        egalizer.append(Incid@alpha_temp/2)
        multiplier.append(multiplier[-1]+beta*(Anti_Incid@alpha_temp)/2)
    return alpha, multiplier

A=np.ones([4,4])-np.eye(4)
