import numpy as np

(S, P, R, T) = (0, 1, 3, 5)

def prisoner_dilemma(c1, c2):
    r1 = c1 * c2 * R + (1 - c1) * (1 - c2) * P + c1 * (1 - c2) * S + c2 * (1 - c1) * T
    r2 = c1 * c2 * R + (1 - c1) * (1 - c2) * P + c1 * (1 - c2) * T + c2 * (1 - c1) * S
    return (r1, r2)

def N_player_prisoner_dilemma(coop_matrix):
    (n_agents, _) = np.shape(coop_matrix)
    rewards = np.zeros(n_agents)
    for i in range(n_agents):
        for j in range(n_agents):
            if i > j:
                (c_i, c_j) = coop_matrix[i,j], coop_matrix[j,i]
                #(c_i, c_j) = accurate_coop_vectors_matrix[i ,j], accurate_coop_vectors_matrix[j ,i]
                (r_i, r_j) = prisoner_dilemma(c_i, c_j)
                rewards[i] += r_i  # * self.max_coop[j,i]
                rewards[j] += r_j  # * self.max_coop[i,j]
    return rewards


ex1 = np.array([[0,1,1,0],[0,0,1,1],[1,0,0,1],[1,1,0,0]])*1.0
ex2 = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])*1.0
ex3 = np.array([[0,0,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])*1.0
