import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def make_gap_graph(alpha, alphaDict):

    font = {'family' : 'sans',
        'size'   : 12}

    matplotlib.rc('font', **font)

    for (methodName, alphaList) in alphaDict.items():
        opt_gap = [np.linalg.norm(alpha_i - alpha) for alpha_i in alphaList]
        plt.loglog(np.arange(1,len(opt_gap)+1), opt_gap, label=methodName)


    plt.xlabel('Number of iterations')
    plt.ylabel(r'Optimality gap $|\alpha_i - \alpha^*|$')
    plt.title('Convergence of optimality gap on Kernel ridge regression')
    plt.grid()
    plt.legend()
    # plt.tight_layout()
    plt.savefig('files/DGD.pdf')