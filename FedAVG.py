import numpy as np
from utils import grad_a_batch

def FedAVG(alpha_0, K_a, K_mm, y_a, sigma, nu, max_iter, lr):

    E = 10
    a = len(K_a) # number of agents
    num_points_per_agent = len(K_a[0]) # number of data points per agent
    # the data of each agent is divided in n_batches batches
    n_batches = 1
    batch_size = num_points_per_agent // n_batches
    alpha = []
    alpha.append(alpha_0)

    # starting from alpha_0, each agent performs mini-batch gradient descent
    BATCHES = {i: None for i in range(1, a+1)}
    for k in range(a):
        indices = np.random.permutation(num_points_per_agent) # random permutation of the data points for agent k
        BATCHES[k] = [indices[i*batch_size:(i+1)*batch_size] for i in range(n_batches)] # batches of data points for agent k

    current_alpha = alpha_0
    alpha_list = [current_alpha]


    for _ in range(max_iter):

        ## CLIENT UPDATES
        alphas = [current_alpha for _ in range(a)] # create a copy of the current alpha for each agent
        for _ in range(E): # perform E iterations of mini-batch gradient descent for each agent
            for k in range(a):
                batch = BATCHES[k][np.random.randint(0, n_batches)] # select a random batch of data points for agent k
                # select the corresponding kernel matrices and labels for the selected batch
                K_a_batch = np.array([K_a[k][i, :] for i in batch])
                y_a_batch = np.array([y_a[k][i] for i in batch])
                grad = grad_a_batch(alphas[k], K_a_batch, K_mm, y_a_batch, sigma, a, nu) # gradient of the objective function of agent k using the data of agent k
                alphas[k] = alphas[k] - lr*grad # update of the local variable of agent k using the gradient and the learning rate
            
        ## SERVER UPDATE
        current_alpha = np.mean(alphas, axis=0) #average of the local variables of all agents
        alpha_list.append(current_alpha)

    return alpha_list
