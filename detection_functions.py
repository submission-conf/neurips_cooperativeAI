import numpy as np
from utils import *
from agents_RL import *
from graphbasedTFT import *



def dectect_potential_cooperation_graph(detected_potential, V_i_j_values, V_i_ij_values, tau = 0.2):
    (n,p) = np.shape(V_i_j_values)
    graph = V_i_j_values
    arg_max = np.argmax(graph, axis=1)
    current_potential = np.zeros([n,n])*1.0
    for i in range(n):
        delta_max = np.max(V_i_j_values[i,:]) - V_i_j_values[i,i]
        if abs(delta_max) < 0.0001:
            current_potential[i,i] = 1.0
            break
        else:
            for j in range(n):
                if j != i:
                    delta = V_i_j_values[i,j] -  V_i_j_values[i,i]
                    current_potential[i,j] = np.clip(delta/delta_max, 0.0, 1.0 )

    new_detected_potential = (1-tau)*detected_potential + tau*current_potential
    return np.round(new_detected_potential, decimals=2)



def dectect_potential_cooperation_graph_bis(detected_potential, V_i_j_values, V_i_ij_values, tau = 0.2):
    (n,p) = np.shape(V_i_j_values)
    graph = V_i_j_values
    arg_max = np.argmax(graph, axis=1)
    current_potential = np.zeros([n,n])*1.0
    for i, j in enumerate(arg_max):
        current_potential[i,j] = 1.0
    new_detected_potential = (1-tau)*detected_potential + tau*current_potential
    return np.round(new_detected_potential, decimals=2)


def detect_current_cooperation_graph_bis(current_cooperation, actions, Q_value_list_list, tau = 0.5):
    (n,_) = np.shape(current_cooperation)
    detection = np.zeros([n,n])*1.0
    for i in range(n):
        for j in range(n):
            delta_a_max = Q_value_list_list[i][j][0][actions[i]] - Q_value_list_list[i][i][0][actions[i]]
            delta_pi_max = np.max(Q_value_list_list[i][j]) - np.max(Q_value_list_list[i][i])

            if abs(delta_pi_max) < 0.0001:
                detection[i,j] = 0.0
            else:
                detection[i, j] = np.clip(delta_a_max/delta_pi_max,0,1)

    new_detection = (1-tau)*current_cooperation + tau*detection
    return new_detection



def detect_current_cooperation_graph(current_cooperation, actions, Q_value_list_list, tau = 0.2):
    (n,_) = np.shape(current_cooperation)
    detection = np.zeros([n,n])*1.0
    for i in range(n):
        for j in range(n):
            Q_values = Q_value_list_list[i][j] # values of Q for the current state of agent i which cooperates with j
            a_true = np.argmax(Q_values)
            if actions[i] == a_true:
                detection[i,j] = 1.0
    new_detection = (1-tau)*current_cooperation + tau*detection
    return new_detection

a = np.array([[4.2534235, 3.848113,  3.3238988, 3.2148008],
              [5.2534235, 3.848113,  3.3238988, 3.2148008],
              [3.2534235, 3.848113,  6.3238988, 3.2148008],
              [3.2534235, 7.848113,  3.3238988, 3.2148008]])

current_potential = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
    ])

