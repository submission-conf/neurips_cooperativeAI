import gym
import circular_collect

import numpy as np
import os
from display_functions import *
from agent_rlTFT import *
from run_game import run_game
from social_metrics import *
from agents_RL import *
from graph_examples import *

""" 
--->>> BEGIN CONFIGURATION <<<---

choice_demo:
    1: graph-based TFT in Circular Game   
    2: graph-based TFT in Bilateral Game 
    3: vanilla TFT in Circular Game   
    4: vanilla TFT in Bilateral Game 
    5: graphTFT + 1 Traitor (defects for t in [30,50] in Circular Game   
    6: graphTFT + 1 Traitor (defects for t in [30,50] in Bilateral Game
    7: graphTFT in a Dynamic Game 
"""
choice_demo = 1  # see above
name_expe = 'DEMOO_'+str(choice_demo) # name of the expe
render = True  # to render graph/payoffs information & environment
render_env = True  # there is an issue if it is impossible to open a display window: very long -> put to False
t_max = 200
limits_y = (-10, 80)
seed = 12345

""" 
--->>> END CONFIGURATION <<<---
"""


if choice_demo in [1,3,5]:
    choice_env = 'C' #circular
else:
    choice_env = 'B'


# CHOICE OF ENV
if choice_env == 'C': #Circular env
    name_env = "circular_collect_special_4x4-v0"
else: #Bilateral env
    name_env = "circular_collect_special_4x4-v1"

# Parameters of Tit-for-Tat
parameters_TFT = [0.6, 0.3, 0.6, 0.1] # alpha, r0, beta, gamma
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
traitor = Agent_Traitor(3,'traitor', max_coop_matrix=max_coop_graph_env, t_traitor=[30,50])

# Nice agents (with graph structure, naive cooperator)
nices_algos = [Agent_Nice(i, 'N', n_agents=4, max_coop_matrix=max_coop_graph_env) for i in range(4)]

# Egoist agents (with graph structure, pure defector)
egoists_algos = [Agent_Egoist(i, 'D', n_agents=4, max_coop_matrix=max_coop_graph_env) for i in range(4)]

##### END of CREATION of Negociation agents (TFT, graphTFT, Egoist, Nices or Traitor) ########################


state_size = 80
action_size = 5
# CHOICE of RL agent
agentRL = Agent_DQN(state_size=state_size, action_size=action_size)

if choice_demo in [1,2]: # only graphTFT
    list_TFT = list_algos_tft
elif choice_demo in [3,4]: # only vanilla TFT
    list_TFT = list_algos_tft_classic
else: # with a Traitor
    list_TFT = list_algos_tft[:3]+[traitor]

# CREATION of rlTFT agents

agentRLTFT0 = Agent_RLTFT(0, list_TFT[0], agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT1 = Agent_RLTFT(1, list_TFT[1], agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT2 = Agent_RLTFT(2, list_TFT[2], agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))
agentRLTFT3 = Agent_RLTFT(3, list_TFT[3], agentRL, state_size, action_size, n_agents=4, env_state_size = (11,11))

# INSTANCES of agents rlTFT
l_agents = [agentRLTFT0, agentRLTFT1, agentRLTFT2, agentRLTFT3]


if choice_demo in []:
    given_potential = False
else:
    given_potential = True

if choice_demo in []:
    given_detection = False
else:
    given_detection = True



if choice_demo in [7]:
    given_potential = False
    steps_change_max_coop = [40,70]
    max_coop_matrix_change = [CIRC_PROBAS_4P_2C, CIRC_PROBAS_4P_B]
else:
    steps_change_max_coop = []
    max_coop_matrix_change = []



env.reset()
seed_game = 54321


curves = run_game(env, list_agents = l_agents, t_max = t_max, seed_game = seed_game,
                  render = render, render_env = render_env, name_expe = name_expe,
                  k_smooth = 10, limits_y = limits_y,
                  given_potential = given_potential, given_detection = given_detection,
                  steps_change_max_coop = steps_change_max_coop, max_coop_matrix_change = max_coop_matrix_change,
                  print_step=True)

