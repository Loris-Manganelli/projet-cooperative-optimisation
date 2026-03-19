import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from Centralized_solution import Cov2
from utils import objective_a

def make_gap_graph(alpha, alphaDict,pdf=True,precisionlimit=None,title="files/GAP.pdf"):
    plt.figure() # Create a new figure for the gap graph
    font = {'family' : 'sans',
        'size'   : 12}

    matplotlib.rc('font', **font)

    

    for (methodName, alphaList) in alphaDict.items():
        n_agents = len(alphaList[0])
        iterations = np.arange(1, len(alphaList) + 1)
        line, = plt.plot([], []) # On crée une ligne vide juste pour piocher la couleur suivante
        color = line.get_color()
        for j in range(n_agents):
            # Calcul du gap pour l'agent j à chaque itération
            agent_gap = [np.linalg.norm(alpha_it[j, :] - alpha) for alpha_it in alphaList]
            
            # On trace avec une ligne fine et une transparence (alpha) pour ne pas surcharger
            # On ne met le label que pour le premier agent pour éviter de polluer la légende
            plt.loglog(iterations, agent_gap,
                    color = color,
                    label=methodName if j == 0 else "")
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

def make_reconstruction_graph(x, y,alpha, alpha_method, ind, n_iter, agent_index, method_name, nt, selection=True):
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

    

def make_FedAVG_graph(alpha, alpha_fedavg, K_a, K_mm, y_a, sigma, nu=1.0, a=5, pdf=True, title="files/FedAVG.pdf"):

    alphaValue = sum([objective_a(alpha, k, K_a, K_mm, y_a, sigma, nu=nu) for k in range(a)])
    alpha_fedavgValues = {E: [] for E in alpha_fedavg.keys()}

    for E in alpha_fedavg.keys():
        for t in range(len(alpha_fedavg[E])):
            alpha_fedavgValues[E].append(sum([objective_a(alpha_fedavg[E][t], k, K_a, K_mm, y_a, sigma, nu=nu) for k in range(a)]))
    
    

    plt.figure() # Create a new figure for the gap graph
    font = {'family' : 'sans',
        'size'   : 12}

    matplotlib.rc('font', **font)

    for (E, valueList) in alpha_fedavgValues.items():
        opt_gap = [np.linalg.norm(alphaValueE - alphaValue) for alphaValueE in valueList]
        plt.loglog(np.arange(1,len(opt_gap)+1), opt_gap, label=f"E={E}")
    plt.xlabel('Number of iterations')
    plt.ylabel(r'Objective gap $|F(\alpha_i) - F(\alpha^*)|$')
    plt.title('FedAVG : Objective gap convergence on Kernel regression')
    plt.grid()
    plt.legend()
    # plt.tight_layout()
    if pdf:
        plt.savefig(title)
    else:
        plt.show()

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

def make_gap_graph_with_tikz(alpha, alphaDict, A, W, lr, precisionlimit=None):
    plt.figure() # Create a new figure for the gap graph
    font = {'family' : 'sans',
        'size'   : 12}

    matplotlib.rc('font', **font)

    

    for (methodName, alphaList) in alphaDict.items():
        n_agents = len(alphaList[0])
        iterations = np.arange(1, len(alphaList) + 1)
        line, = plt.plot([], []) # On crée une ligne vide juste pour piocher la couleur suivante
        color = line.get_color()
        for j in range(n_agents):
            # Calcul du gap pour l'agent j à chaque itération
            agent_gap = [np.linalg.norm(alpha_it[j, :] - alpha) for alpha_it in alphaList]
            
            # On trace avec une ligne fine et une transparence (alpha) pour ne pas surcharger
            # On ne met le label que pour le premier agent pour éviter de polluer la légende
            plt.loglog(iterations, agent_gap,
                    color = color,
                    label=methodName if j == 0 else "")
    plt.ylim(bottom=precisionlimit)
    plt.xlabel('Number of iterations')
    plt.ylabel(r'Optimality gap $|\alpha_i - \alpha^*|$')
    plt.title('Convergence of optimality gap on Kernel ridge regression')
    plt.grid()
    plt.legend()

    # Sauvegarde du fond
    plt.savefig("files/plot_background_DGD_GT.pdf", bbox_inches='tight')
    plt.close()

    #Conversion de la matrice W en chaîne LaTeX (Fractions dynamiques) ---
    matrix_rows = []
    for row in W:
        items = []
        for val in row:
            # On vérifie si la valeur est strictement positive (seuil epsilon)
            if val > 1e-5:
                # Détection des fractions spécifiques
                if abs(val - 1/3) < 1e-3:
                    items.append(r"\frac{1}{3}")
                elif abs(val - 2/3) < 1e-3:
                    items.append(r"\frac{2}{3}")
                elif abs(val - 1/5) < 1e-3:
                    items.append(r"\frac{1}{5}")
                else:
                    # Pour les autres valeurs, on affiche 2 décimales proprement
                    items.append(f"{val:.2f}".rstrip('0').rstrip('.'))
            else:
                # Cellule vide pour W = 0
                items.append(" ") 
        
        matrix_rows.append(" & ".join(items))
    
    latex_matrix = " \\\\\n            ".join(matrix_rows)

    #Génération du fichier .tex dynamique ---
    tex_content = r"""\documentclass{standalone}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{amsmath}
\begin{document}
\begin{tikzpicture}
    \node[anchor=south west, inner sep=0] (image) at (0,0) {\includegraphics[width=12cm]{plot_background_DGD_GT.pdf}};
    \begin{scope}[x={(image.south east)}, y={(image.north west)}]
        
        % Graphe d'adjacence A
        \node[anchor=south west] at (0.15, 0.18) {
            \begin{tikzpicture}[scale=0.8, every node/.style={circle, fill=black, inner sep=1.2pt}]
                % Calcul automatique des positions des noeuds en cercle
                \foreach \i in {1,...,""" + str(len(A)) + r"""}{
                    \node (n\i) at ({360/""" + str(len(A)) + r""" * \i}:1cm) {};
                }
                % Dessin des arretes si A[i,j] > 0
                """ + "".join([f"\\draw (n{i+1}) -- (n{j+1}); " 
                               for i in range(len(A)) 
                               for j in range(i+1, len(A)) if A[i,j] > 0]) + r"""
            \end{tikzpicture}
        };

        % Matrice W dynamique
        \node[anchor=south] at (0.52, 0.2) {
            $W = \begin{bmatrix} 
            """ + latex_matrix + r"""
            \end{bmatrix}$
        };

        % Step-size lr dynamique
        \node[anchor=south west] at (0.75, 0.25) {
            $s = """ + str(lr) + r"""$
        };
    \end{scope}
\end{tikzpicture}
\end{document}
"""
    with open("files/DGD_GT.tex", "w") as f:
        f.write(tex_content)
    
    print("Fichiers 'plot_background_DGD_GT.pdf' et 'DGD_GT.tex' générés.")