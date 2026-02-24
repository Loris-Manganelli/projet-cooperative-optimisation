import pickle
from ADMM import ADMM
from Centralized_solution import solve, plot_me, Cov2, Cov
import numpy as np
from DGD import DGD
from GT import GT
from Dual_decomposition import dual_decomposition
from visualisation import make_gap_graph, make_reconstruction_graph
import matplotlib.pyplot as plt
with open('data/first_database.pkl', 'rb') as f:
   x,y = pickle.load(f)


### PARAMETERS TO BE MODIFIED
num_points = 100
n_iter = 50000
step_size = 0.001

# Graph topology :
a = 5 # number of agents
W = np.ones([a,a])/a # consensus matrix (fully connected graph) TO BE MODIFIED for other topologies
#adjacency matrix of the graph (circular graph)
A = np.ones([a,a])- np.eye(a) 
# A = np.zeros([a,a])
# for i in range(a):
#     A[i,(i+1)%a] = 1
#     A[(i+1)%a,i] = 1

## DATA PREPARATION
x = x[:num_points]
y = y[:num_points]
alpha, ind = solve(x[:num_points],y[:num_points], selection=True)
n = len(x)
m = int(np.sqrt(n))
x2 = [x[i] for i in ind] # subset M of the data points
K_nm = Cov2(x, x2) # kernel matrix between the data points and the subset M
K_mm = Cov(x2) # kernel matrix between the subset M
points_per_agent = n // a # number of data points per agent
indices = np.random.permutation(n) # random permutation of the data points to be distributed among the agents
K_a = [K_nm[indices[i*points_per_agent:(i+1)*points_per_agent], :] for i in range(a)]
y_a = [y[indices[i*points_per_agent:(i+1)*points_per_agent]] for i in range(a)]


alpha_0 = np.zeros((a,m)) # Initialization of the local variables for each agent
multipliers_0 = np.zeros([int(np.sum(A)/2),m]) # Initialization of the multipliers for dual decomposition
egalizers = np.zeros((int(np.sum(A)/2),m)) # Initialization of the egalizers for dual decomposition


## DGD SOLVE 
alpha_dgd = DGD(alpha_0, K_a, K_mm, y_a, W, sigma=0.5, nu=1.0, max_iter=n_iter, lr=step_size)
## GT SOLVE
alpha_gt = GT(alpha_0, K_a, K_mm, y_a, W, sigma=0.5, nu=1.0, max_iter=n_iter, lr=step_size)
## DUAL DECOMPOSITION SOLVE
alpha_dualdecomp, multipliers = dual_decomposition(multipliers_0, K_a, K_mm, y_a, np.ones([a, a]), sigma=0.5, nu=1.0, max_iter=n_iter, lr=10*step_size)   
### ADMM SOLVE
alpha_admm, multipliers_admm = ADMM(multipliers_0, egalizers, beta=10.0, K_a=K_a, K_mm=K_mm, y_a=y_a, A=np.ones([a,a])-np.eye(a), sigma=0.5, nu=1.0, max_iter=n_iter)


# test de forat : 
# print ("alpha_dgd shape : ", alpha_dgd[-1].shape)
# print ("alpha_gt shape : ", alpha_gt[-1].shape)
# print ("alpha_dualdecomp shape : ", alpha_dualdecomp[-1].shape)

# PLOTS
# plot_me(x[:num_points],y[:num_points], alpha, ind, selection=True)


alphaDict = {'DGD': alpha_dgd, 'GT': alpha_gt, 'Dual Decomposition': alpha_dualdecomp, 'ADMM': alpha_admm}

make_reconstruction_graph(x[:num_points],y[:num_points], alpha, alpha_dgd[-1], ind, selection=True, n_iter=n_iter, method_name="DGD", nt=250, agent_index=0)
make_gap_graph(alpha, alphaDict)


