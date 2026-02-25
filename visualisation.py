import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from Centralized_solution import Cov2

def make_gap_graph(alpha, alphaDict,pdf=True,precisionlimit=1e-12,title="files/GAP.pdf"):
    plt.figure() # Create a new figure for the gap graph
    font = {'family' : 'sans',
        'size'   : 12}

    matplotlib.rc('font', **font)

    for (methodName, alphaList) in alphaDict.items():
        opt_gap = [np.linalg.norm(alpha_i - alpha) for alpha_i in alphaList]
        plt.loglog(np.arange(1,len(opt_gap)+1), opt_gap, label=methodName)
    plt.ylim(bottom=precisionlimit)
    plt.xlabel('Number of iterations')
    plt.ylabel(r'Optimality gap $|\alpha_i - \alpha^*|$')
    plt.title('Convergence of optimality gap on Kernel ridge regression')
    plt.grid()
    plt.legend()
    # plt.tight_layout()
    if pdf:
        plt.savefig(title)
    else:
        plt.show()

def make_reconstruction_graph(x, y,alpha, alpha_method, ind, n_iter, agent_index, method_name, nt, selection=True,):
    plt.figure() # Create a new figure for the reconstruction graph
    plt.plot(x,y,'o', label="Data")
    xo = np.linspace(-1,1,nt)
    if selection:
        x2 = [x[i] for i in ind]
    else:
        x2 = np.linspace(-1, 1, 10)

    y_exact = Cov2(xo, x2) @ alpha
    plt.plot(xo, y_exact, color='orange', linestyle='-', label="Optimal reconstruction")
    yo = Cov2(xo, x2) @ alpha_method[agent_index]
    plt.plot(xo, yo, color='red', linestyle='--', label=f"Reconstruction Agent {agent_index} after {n_iter} iterations")
    plt.xlabel(r'$x$ feature')
    plt.ylabel(r'$y$ label')
    plt.title(f'Reconstruction of the function with {method_name} compared to the exact solution')
    plt.grid()
    plt.legend()
    plt.savefig("files/RECONSTRUCTION.pdf")

    
