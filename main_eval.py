import gym
import circular_collect

import numpy as np
import os
from display_functions import *
from agent_rlTFT import *
from run_game import run_game
from social_metrics import *
from agents_RL import *
import statistics as st
import pickle
from scipy.spatial.distance import euclidean
from graph_examples import *

""" 
--->>> BEGIN CONFIGURATION <<<---

choice_env:
    'C' (circular)
    'B' (bilateral)
choice_tft:
    'gr' (graph-based Tit-for-Tat)
    'va' (vanilla Tit-for-Tat)
    'eg' (egoist)
    'ni' (nice)
parameters_TFT:
    alpha: inertia
    r0: coefficient of incentive (initial)
    beta: coefficient to adapt the incentive r
    gamma: coefficient of stochasticity (incentive with proba gamma)
"""
choice_env = 'C' # C: circular, B: bilateral
choice_tft = 'gr' # gr: graphTFT, va: vanilla TFT, eg: egoist, ni: nice
parameters_TFT = [0.6, 0.3, 0.6, 0.1] # alpha, r0, beta, gamma


""" 
Parameters of simulation for the computation of metrics
"""
change_t_moments = [] # for dynamic maximal cooperation of the game
change_coop_graph = [] # for dynamic maximal cooperation of the game
given_potential = True
given_detection = True
n_runs = 3
t_max = 500
name_expe = 'EXPE'
save_results = False
""" 
--->>> BEGIN CONFIGURATION <<<---
"""


# CHOICE OF ENV
if choice_env == 'C': #Circular env
    name_env = "circular_collect_special_4x4-v0"
else: #Bilateral env
    name_env = "circular_collect_special_4x4-v1"

# Parameters of Tit-for-Tat
alpha = parameters_TFT[0]
r0 = parameters_TFT[1]
beta = parameters_TFT[2]
gamma = parameters_TFT[3]

# CREATION of environment
env = gym.make(name_env)
env.reset()

# DEFAULT maximal cooperation graph, just for initialisation (will be modified by agent rlTFT)
max_coop_graph_env = None

##### CREATION of Negociation agents (TFT, graphTFT, Egoist, Nices or Traitor) #############################

# Graph-based Tit-for-Tat ######
agent_grTFT0 = Agent_TFT_Gamma(0, 'agent0', n_agents=4, max_coop_matrix=max_coop_graph_env, alpha_inertia=alpha, r_incentive=r0, beta_adaptive=beta, gamma_proba=gamma)
agent_grTFT1 = Agent_TFT_Gamma(1, 'agent1', n_agents=4, max_coop_matrix=max_coop_graph_env, alpha_inertia=alpha, r_incentive=r0, beta_adaptive=beta, gamma_proba=gamma)
agent_grTFT2 = Agent_TFT_Gamma(2, 'agent2', n_agents=4, max_coop_matrix=max_coop_graph_env, alpha_inertia=alpha, r_incentive=r0, beta_adaptive=beta, gamma_proba=gamma)
agent_grTFT3 = Agent_TFT_Gamma(3, 'agent3', n_agents=4, max_coop_matrix=max_coop_graph_env, alpha_inertia=alpha, r_incentive=r0, beta_adaptive=beta, gamma_proba=gamma)
list_algos_tft = [agent_grTFT0, agent_grTFT1, agent_grTFT2, agent_grTFT3]

# Vanilla Tit-for-Tat ######
agent_TFT0 = Agent_TFT_NoGraph_Gamma(0, 'agent0', n_agents=4, max_coop_matrix=max_coop_graph_env, alpha_inertia=alpha, r_incentive=r0, beta_adaptive=beta, gamma_proba=gamma)
agent_TFT1 = Agent_TFT_NoGraph_Gamma(1, 'agent1', n_agents=4, max_coop_matrix=max_coop_graph_env, alpha_inertia=alpha, r_incentive=r0, beta_adaptive=beta, gamma_proba=gamma)
agent_TFT2 = Agent_TFT_NoGraph_Gamma(2, 'agent2', n_agents=4, max_coop_matrix=max_coop_graph_env, alpha_inertia=alpha, r_incentive=r0, beta_adaptive=beta, gamma_proba=gamma)
agent_TFT3 = Agent_TFT_NoGraph_Gamma(3, 'agent3', n_agents=4, max_coop_matrix=max_coop_graph_env, alpha_inertia=alpha, r_incentive=r0, beta_adaptive=beta, gamma_proba=gamma)
list_algos_tft_classic = [agent_TFT0, agent_TFT1, agent_TFT2, agent_TFT3]


# One Traitor (defects between t=50 and t=60) ######
traitor = Agent_Traitor(3,'traitor', max_coop_matrix=max_coop_graph_env, t_traitor=[50,60])

# Nice agents (with graph structure, naive cooperator)
nices_algos = [Agent_Nice(i, 'N', n_agents=4, max_coop_matrix=max_coop_graph_env) for i in range(4)]

# Egoist agents (with graph structure, pure defector)
egoists_algos = [Agent_Egoist(i, 'D', n_agents=4, max_coop_matrix=max_coop_graph_env) for i in range(4)]

##### END of CREATION of Negociation agents (TFT, graphTFT, Egoist, Nices or Traitor) ########################


state_size = 80
action_size = 5
# CHOICE of RL agent
agentRL = Agent_DQN(state_size=state_size, action_size=action_size)

# CREATION of rlTFT agents

agentRLTFT0 = Agent_RLTFT(0, agent_grTFT0, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT1 = Agent_RLTFT(1, agent_grTFT1, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT2 = Agent_RLTFT(2, agent_grTFT2, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT3 = Agent_RLTFT(3, agent_grTFT3, agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))

# INSTANCES of agents rlTFT
l_agents = [agentRLTFT0, agentRLTFT1, agentRLTFT2, agentRLTFT3]
list_TFT = list_algos_tft_classic


# CHOICE OF NEGOCIATION agents
if choice_tft == "gr":
    list_TFT = list_algos_tft
elif choice_tft == "va":
    list_TFT = list_algos_tft_classic
elif choice_tft == "eg":
    list_TFT = egoists_algos
else:
    list_TFT = nices_algos



# RUNNING GAMES for computation of social metrics #
output = compute_mean_social_metrics(env,
                                     list_agents = l_agents,
                                     list_TFT = list_TFT,
                                     n_runs = n_runs, t_max = t_max,
                                     given_potential=given_potential, given_detection=given_detection,
                                     change_coop_graph=change_coop_graph, change_t_moments=change_t_moments)



if save_results:
    f = open(name_expe + str(n_runs)+"runs_"+str(t_max)+"steps.obj","wb")
    pickle.dump(output,f)
    f.close()

