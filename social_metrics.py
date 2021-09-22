import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from run_game import *
import statistics as st
import pickle
from scipy.spatial.distance import euclidean
from graphbasedTFT import *

# CREATE a default max coop matrix
max_coop_graph_env = np.array([[0,1,1,0],[0,0,1,1],[1,0,0,1],[1,1,0,0]])*1.0

# CREATE n_agent Nice "TFT agents"
nices_algos = [Agent_Nice(i, 'N', n_agents=4, max_coop_matrix=max_coop_graph_env) for i in range(4)]

# CREATE n_agent Egoist "TFT agents"
egoists_algos = [Agent_Egoist(i, 'D', n_agents=4, max_coop_matrix=max_coop_graph_env) for i in range(4)]



def mean_std(values, rd = 3):
    m = st.mean(values)
    std = st.stdev(values)
    return (np.round(m, rd) , np.round(std, rd))


def efficiency(curves, rd = 3):
    n = len(curves)
    t_max = len(curves[0])
    u = 0
    for c in curves:
        u += c[-1]
    output = u/t_max
    return np.round(output, decimals=rd)

def safety(curves_agent_vs_allegoists, curves_all_egoists, rd = 3):
    n = len(curves_agent_vs_allegoists)
    t_max = len(curves_agent_vs_allegoists[0])
    r1 = curves_agent_vs_allegoists[0][-1] # agent face to all egoists
    r2 = curves_all_egoists[0][-1]  # all egoist
    for i in range(0):
        pass
        #u1 += curves_agent_vs_allegoists[i][-1]
        #u2 += curves_all_egoists[i][-1]
    #print("safety ", r1, r2)
    output = (r1 - r2)/t_max
    return np.round(output, decimals=rd)


def incentive_compatibility(curves_1cooperator_vs_agents, curves_1defector_vs_agents, rd = 3):
    n = len(curves_1cooperator_vs_agents)
    t_max = len(curves_1cooperator_vs_agents[0])
    r1 = curves_1cooperator_vs_agents[0][-1] # payoff if cooperation
    r2 = curves_1defector_vs_agents[0][-1] # payoff if defection
    #print(r1, r2)
    output = (r1 - r2)/t_max
    return np.round(output, decimals=rd)



def compute_mean_social_metrics(env, list_agents,
                                list_TFT, egoists_algos = egoists_algos, nices_algos = nices_algos,
                                n_runs = 5 , t_max = 200, choose_agent_for_incent = False,
                                given_potential = True, given_detection = True,
                                change_coop_graph = [],
                                change_t_moments = [],
                                ):

    n_agents = len(list_agents)

    for i in range(n_agents):
        list_agents[i].agent_grTFT = list_TFT[i]

    efficiencies = []
    incentives = []
    safeties = []

    print("BEGIN Computation of Social metrics")
    # for efficiency:
    for k in range(n_runs):
        env.reset(seed=100 + k)
        render=False
        if k<0:
            render = True
        curves_all_agents = run_game(env, list_agents=list_agents, t_max=t_max, render=render, render_env=False, given_detection=given_detection,
                                        given_potential=given_potential, max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)


        #curves_all_agents = run_game(env, list_agents=l_agents, t_max=t_max, render=True, render_env=True, name_expe = '0_TFT_6', limits_y=(-10,100), steps_change_max_coop=[10], max_coop_matrix_change=[CIRC_PROBAS_4P_B])
        ef = efficiency(curves_all_agents)
        print('efficiency - run ' +str(k) + ': ' + str(ef))
        efficiencies.append(ef)

    # for incentive-compatibility
    for k in range(n_runs):
        if choose_agent_for_incent:
            list_agents[0].agent_grTFT = list_TFT[0]
        else:
            list_agents[0].agent_grTFT = nices_algos[0]
        env.reset(seed=200 + k)
        #curves_1cooperator_vs_agents = run_game(env, list_agents=l_agents, t_max=t_max, render=True, render_env=False, name_expe = '0_incentive1', limits_y=(-10,100))
        curves_1cooperator_vs_agents = run_game(env, list_agents=list_agents, t_max=t_max, render=False, given_detection=given_detection, given_potential=given_potential,
                                                                                    max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)
        list_agents[0].agent_grTFT = egoists_algos[0]
        env.reset(seed=300 + k)
        curves_1defector_vs_agents = run_game(env, list_agents=list_agents, t_max=t_max, render=False, given_detection=given_detection, given_potential=given_potential,
                                                                                    max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)
        ic = incentive_compatibility(curves_1cooperator_vs_agents, curves_1defector_vs_agents)
        print('incentive - run ' +str(k) + ': ' + str(ic))
        incentives.append(ic)

    # for safety:
    for k in range(n_runs):

        # all egoists
        for i in range(n_agents):
            list_agents[i].agent_grTFT = egoists_algos[i]
        env.reset(seed=400 + k)
        curves_all_egoists = run_game(env, list_agents=list_agents, t_max=t_max, render=False, given_detection=given_detection, given_potential=given_potential,
                                                                                    max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)


        # 1 agent vs (n-1) egoist
        list_agents[0].agent_grTFT = list_TFT[0]
        env.reset(seed=500 + k)
        curves_agent_vs_allegoists = run_game(env, list_agents=list_agents, t_max=t_max, render=False, given_detection=given_detection, given_potential=given_potential,
                                                                                    max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)
        #curves_agent_vs_allegoists = run_game(env, list_agents=l_agents, t_max=t_max, render=True, render_env=False, name_expe='0_safety_4', limits_y=(-10, 100))

        sf = safety(curves_agent_vs_allegoists, curves_all_egoists)
        print('safety - run ' +str(k) + ': ' + str(sf))
        safeties.append(sf)

    output = [mean_std(efficiencies), mean_std(incentives), mean_std(safeties)]
    print()
    print("Mean and standard deviation of metrics:")
    print(output)


    return output




# metrics to compute distance between true graph and detected graph - Non used ###
def compute_distance_graph(env, list_agents,
                                list_TFT,
                                n_runs = 5 ,
                                t_max = 200,
                                change_t_moments = None,
                                change_coop_graph = None
                           ):

    n_agents = len(list_agents)

    for i in range(n_agents):
        list_agents[i].agent_grTFT = list_TFT[i]

    curves_distances_current_coop = []
    curves_distances_potential = []


    for k in range(n_runs):
        env.reset(seed=100 + k)
        curves_payoffs, distances_current_coop, distances_potential_coop = run_game(env, list_agents=list_agents, t_max=t_max,
                                                                                    render=False, render_env=False, name_expe = '012_A',
                                                                                    dist_potential = True, dist_current_coop = False,
                                                                                    max_coop_matrix_change=change_coop_graph, steps_change_max_coop=change_t_moments)

        curves_distances_potential.append(distances_potential_coop)
        curves_distances_current_coop.append(distances_current_coop)

    return curves_distances_current_coop, curves_distances_potential