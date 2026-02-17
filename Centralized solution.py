import numpy as np
import matplotlib.pyplot as plt
import pickle
from DGD import DGD

def Cov(x):
    m = len(x)
    Kmm = np.eye(m)
    for ii in range(m):
        for jj in range(ii+1,m):
            Kmm[ii,jj] = np.exp(-(x[ii]-x[jj])**2 )
            Kmm[jj,ii] = Kmm[ii,jj]

    return Kmm

def Cov2(x1,x2):
    m = len(x2)
    n = len(x1)
    Knm = np.zeros([n,m])
    for ii in range(n):
        for jj in range(m):
            Knm[ii, jj] = np.exp(-(x1[ii] - x2[jj]) ** 2 )
    return Knm

def solve(x,y, selection=True):
    n = len(x)

    # you can either select the points among the ones you have:
    if selection:
        sel = [i for i in range(n)]
        ind = np.random.choice(sel, int(np.sqrt(n)), replace=False)
        x2 = [x[i] for i in ind]

    # or take them uniformly distributed
    else:
        x2 = np.linspace(-1, 1, 10)
        ind = []

    M = Cov2(x, x2)
    A = (0.5**2)*Cov(x2) + M.T @ M
    b = M.T @ y

    # here the regularization parameter nu is 1.0
    A = A + 1.*np.eye(int(np.sqrt(n)))

    # it is good to compute the max/min eigenvalues of A for later, but only for small-size matrices
    if n<101:
        ei, EI =np.linalg.eig(A)
        vv = [min(ei), max(ei)]
        print('Min and max eigenvalues of A : ', print(vv))

    alpha = np.linalg.solve(A,b)

    return alpha, ind

def plot_me(x,y, alpha, ind, selection=True):

    plt.plot(x,y,'o')

    xo = np.linspace(-1,1,100)
    if selection:
        x2 = [x[i] for i in ind]
    else:
        x2 = np.linspace(-1, 1, 10)


    yo = Cov2(xo, x2) @ alpha
    plt.plot(xo, yo, '-')
    plt.xlabel(r'$x$ feature')
    plt.ylabel(r'$y$ label')
    plt.grid()

    plt.show()


"""
Main follows
"""

with open('first_database.pkl', 'rb') as f:
   x,y = pickle.load(f)


num_points = 100
alpha, ind = solve(x[:num_points],y[:num_points], selection=True)

print('Result summary -----------------')
print('Optimal centralised alpha = ', alpha)

plot_me(x[:num_points],y[:num_points], alpha, ind, selection=True)


# matplotlib setup
import matplotlib
# adjust the font size accordingly
font = {'family' : 'sans',
        'size'   : 12}

matplotlib.rc('font', **font)

############ DGD algorithm #############
# Parameters for DGD (TO BE DISCUSSED and justified)
n_iter = 200000
step_size = 0.001
a = 5 # number of agents
x = x[:num_points]
y = y[:num_points]
n = len(x)
m = int(np.sqrt(n))
x2 = [x[i] for i in ind] # subset M of the data points
K_nm = Cov2(x, x2) # kernel matrix between the data points and the subset M
K_mm = Cov(x2) # kernel matrix between the subset M
points_per_agent = n // a # number of data points per agent
indices = np.random.permutation(n) # random permutation of the data points to be distributed among the agents
# Split the data among the agents for the sum over Agents in the DGD algorithm
K_a = [K_nm[indices[i*points_per_agent:(i+1)*points_per_agent], :] for i in range(a)]
y_a = [y[indices[i*points_per_agent:(i+1)*points_per_agent]] for i in range(a)]
W = np.ones([a,a])/a # consensus matrix (fully connected graph) TO BE MODIFIED for other topologies
alpha_0 = np.zeros((a,m)) # Initialization of the local variables for each agent
alpha_dgd = DGD(alpha_0, K_a, K_mm, y_a, W, sigma=0.5, nu=1.0, max_iter=n_iter, lr=step_size)
print('DGD alpha = ', alpha_dgd[-1])
# Plot Optimal gap
gap = []
opt_gap = [np.linalg.norm(alpha_i - alpha) for alpha_i in alpha_dgd]
plt.loglog(np.arange(1,len(opt_gap)+1), opt_gap)
plt.plot(gap, label='DGD')
plt.xlabel('Number of iterations')
plt.ylabel(r'Optimal gap $|\alpha_i - \alpha^*|$')
plt.title(r'Convergence in terms of $|\alpha_i - \alpha^*|$ for DGD on the Kernel ridge regression example')
plt.grid()
# Save your figure in .pdf
plt.tight_layout()
plt.savefig('DGD.pdf')