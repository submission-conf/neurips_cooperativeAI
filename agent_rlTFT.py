import numpy as np
from utils import *
from agents_RL import *
from graphbasedTFT import *
from detection_functions import *


IDENT_TO_COORD = [(0,0),(1,0),(1,1),(0,1)]

class Agent_RLTFT:
    """
    Agent mixing Reinforcement Learning policies and graph-based Tit-for-Tat (TFT) strategies
    Only for execution phase
    """

    def __init__(self, ident, agent_grTFT, agent_RL,
                 state_size, action_size, n_agents=4, env_state_size = (11,11),
                 path_model = './models/DQN_simple/'):

        self.ident = ident
        self.n_agents = n_agents
        self.agent_grTFT = agent_grTFT # agent dealing with graph-based cooperation (Tit-for-Tat)
        self.agent_RL = agent_RL # generic RL agent whose weights can be modified to define other cooperation policy
        self.inner_coop_graph = np.eye(self.n_agents) # cooperation graph
        self.detection_coop_graph = np.eye(self.n_agents) # detection of cooperation graph
        self.discrete_coop = True
        self.RL_policies_path = './'
        self.RL_policies = [[]]*self.n_agents # size = 2^(n_agents-1) if discrete_coop, size = 1 otherwise
        self.RL_generic_policies = [[]]*4
        self.max_coop_graph = np.eye(self.n_agents) # maximal possible cooperation graph (dynamic or fixed)
        self.state_size = state_size
        self.action_size = action_size
        self.env_state_size = env_state_size
        self.set_partition = create_partition_set(self.n_agents)
        self.path_model = path_model

        self.prev_state = []

        self.k_steps_act_coop_max = 5  # act a fixed cooperation policy during k_steps_act_coop steps
        self.k_steps_act_coop = 0

        self.k_steps_dect_coop_max = 5  # for the (discrete) detection of cooperation
        self.k_steps_dect_coop_max = 0  # for the (discrete) detection of cooperation

        self.coop_receivers = [] # if discrete cooperation: fixed set of agents receiving cooperation (to choose the proper policy)
        self.coop_degrees_fixed = [0] * self.n_agents # if continuous cooperation
        self.current_reaction_coop = np.ones(self.n_agents)

        self.fct_detection_potential = dectect_potential_cooperation_graph
        self.fct_detection_cooperation = detect_current_cooperation_graph

        self.step_t = 0


    def reset(self):
        # reset the Tit-for-Tat algorithm
        self.agent_grTFT.reset()

        self.step_t = 0
        self.k_steps_act_coop_max = 5  # act a fixed cooperation policy during k_steps_act_coop steps
        self.k_steps_act_coop = 0

        self.k_steps_dect_coop_max = 5  # for the (discrete) detection of cooperation
        self.k_steps_dect_coop_max = 0  # for the (discrete) detection of cooperation

    def set_parameters_TFT(self, parameters_TFT):
        (alpha, r0, beta, gamma) = parameters_TFT
        #alpha_inertia = 0.6, r_incentive = 0.3, beta_adaptive = 0.6, gamma_proba = 0.1
        if hasattr(self.agent_grTFT, 'alpha_inertia'):
            print('has alpha')
            self.agent_grTFT.alpha_inertia = alpha

    def preprocess_state_fct(self, ident, list_agents):
        def fct(state):
            state = self.preprocess_state(state, ident)
            state = preprocess_state_layers_for_agent(state, ident, list_agents).reshape(1, -1)
            return state
        return fct

    def set_cooperation_policy(self):
        # temporary
        if len(self.coop_receivers) == 0:
            return self.set_policy_i_for_j(self.ident, [self.ident])
        elif len(self.coop_receivers) >= 1:
            return self.set_policy_i_for_j(self.ident, self.coop_receivers)

    def set_policy_i_for_j(self, i, l):
        policy_weights = self.path_model + 'Q_gen_P_a_for_x.pth'
        self.agent_RL.model.load_state_dict(torch.load(policy_weights))
        return self.preprocess_state_fct(i, l)


    def set_policy_i_for_list(self, i, list_agents):
        if len(list_agents) == 0:
            policy_weights = self.path_model + 'Q_gen_P_a_for_x.pth'
            self.agent_RL.model.load_state_dict(torch.load(policy_weights))
            return self.preprocess_state_fct(i, [i])
        elif len(list_agents) == 1:
            policy_weights = self.path_model + 'Q_gen_P_a_for_b.pth'
            self.agent_RL.model.load_state_dict(torch.load(policy_weights))
            return self.preprocess_state_fct(i, list_agents)
        elif len(list_agents) == 2:
            policy_weights = self.path_model + 'Q_gen_P_a_for_bc.pth'
            self.agent_RL.model.load_state_dict(torch.load(policy_weights))
            return self.preprocess_state_fct(i, list_agents)
        elif len(list_agents) == 3:
            policy_weights = self.path_model + 'Q_gen_P_a_for_bcd.pth'
            self.agent_RL.model.load_state_dict(torch.load(policy_weights))
            return self.preprocess_state_fct(i, list_agents)


    def preprocess_state(self, state, ident=None):
        if ident is not None:
            (k,l) = IDENT_TO_COORD[ident]
        else:
            (k,l) = IDENT_TO_COORD[self.ident]
        return preprocess_state(state, self.env_state_size[0], self.env_state_size[1], k, l)


    def get_Q_values(self, state, ident, list_agents):
        # get values of Q[state] with Q the model from ident cooperating with agents of list_agents
        fct_obs_preprocess = self.set_policy_i_for_list(ident, list_agents)
        state = fct_obs_preprocess(state)
        return self.agent_RL.q_values(state)

    def compute_Q_values_list(self, state):
        list_list_q_values = []
        for i in range(self.n_agents):
            list_q_values = []
            for j in range(self.n_agents):
                q_values = self.get_Q_values(state, i, [j])
                list_q_values.append(q_values)
            list_list_q_values.append(list_q_values)
        return list_list_q_values

    def compute_V_values(self, state):
        # for the state, compute the values V[s] for each agent i cooperating with j
        V = np.zeros([self.n_agents, self.n_agents])
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                V[i,j] = np.max(self.get_Q_values(state, i, [j]))
        return V


    def dectect_potential(self, state, max_coop_potential = None):
        # updates the estimation of the maximal cooperation graph (potential cooperation between agents)
        if max_coop_potential is None:
            V_values = self.compute_V_values(state)
            self.max_coop_graph = self.fct_detection_potential(self.max_coop_graph, V_values, V_values)
        else:
            self.max_coop_graph = max_coop_potential

        self.agent_grTFT.max_coop_matrix = self.max_coop_graph

    def detect_cooperation(self, prev_state, prev_actions, prev_coop_mat=None):
        # updates the estimation of the current cooperation graph (previous cooperation between agents)
        if prev_coop_mat is None:
            q_values_list_list = self.compute_Q_values_list(prev_state)
            self.detection_coop_graph = self.fct_detection_cooperation(self.detection_coop_graph, prev_actions, q_values_list_list, tau = 0.1)
        else:
            self.detection_coop_graph = prev_coop_mat


    def change_coop_choice(self, tmp_cont_degrees=None):
        # modify the choice of partners every k_steps_act_coop_max steps
        self.k_steps_act_coop = (self.k_steps_act_coop+1)%self.k_steps_act_coop_max
        if self.k_steps_act_coop == 0:
            #continuous_coop_degree = self.agent_grTFT.act(self.detection_coop_graph, self.step_t)
            #continuous_coop_degree = tmp_cont_degrees
            #print(continuous_coop_degree)
            continuous_coop_degree = self.current_reaction_coop
            chosen_agents = np.random.binomial(1, continuous_coop_degree)
            self.coop_receivers = [i for i in range(self.n_agents) if (chosen_agents[i] == 1)]
        #print("t = ", self.step_t, " receivers for agent ", self.ident, self.coop_receivers)

    def reaction_cooperation(self):
        # Tit-for-Tat algorithm
        # according to a cooperation detection, modify the inner graph-cooperation graph
        self.inner_coop_graph = self.inner_coop_graph
        #print("agent ", self.ident)
        #print("name_algo", self.agent_grTFT.name)
        #print('self.dectection', self.detection_coop_graph)
        max_coop = self.agent_grTFT.max_coop_matrix[self.ident, :]
        #print("my max coop", max_coop)
        self.current_reaction_coop = self.agent_grTFT.act(self.detection_coop_graph, self.step_t)
        #print('self.current_reaction_coop', self.current_reaction_coop)

    def act(self, state, prev_state, prev_actions, prev_coop_mat = None, max_coop_potential = None):
        self.step_t += 1
        self.detect_cooperation(prev_state, prev_actions, prev_coop_mat)
        self.dectect_potential(state, max_coop_potential)
        self.reaction_cooperation()
        self.change_coop_choice()

        fct_obs_preprocess = self.set_cooperation_policy()
        state = fct_obs_preprocess(state)
        return self.agent_RL.act(state)

