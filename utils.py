import numpy as np
import string
import gym
#import circular_collect

partition_set1 = ['x','0']
partition_set2 = ['x','0','1','01']
partition_set2 = ['x','0','1','2','01','02','12','012']
partition_set3 = ['x','0','1','2','3','01','02','03','12','13','23','012','013','023','123','0123']


def partition_indiv(p, ident_agent):
    p2 = []
    for c in p:
        if str(ident_agent) not in c:
            p2.append(c)
    return p2

def create_partition(liste):
    if liste == []:
        return ['x']
    else:
        a = liste[-1]
        l = liste[:-1]
        partition = create_partition(l)
        new_partition = []
        for c in partition:
            new_partition.append(c)
            if c == 'x':
                new_partition.append(a)
            else:
                new_partition.append(c+a)
        return new_partition

def create_partition_set(n_agents):
    if n_agents>10:
        print("WARNING: with more than 10 players, the name of models is not adapted")
    list_chars = [str(i) for i in range(n_agents)]
    output = create_partition(list_chars)
    output.sort()
    return output

def create_partition_set_agent(n_agents, ident_agent):
    if n_agents>10:
        print("WARNING: with more than 10 players, the name of models is not adapted")
    list_chars = [str(i) for i in range(n_agents)]
    partition = create_partition(list_chars)
    output = partition_indiv(partition, ident_agent)
    output.sort()
    return output


def create_name_model(ident_player, discrete_coop = True, partition = None, ext = '.pth'):
    if discrete_coop:
        assert partition is not None
        name = 'Q_p' + str(ident_player) + '_for_' + str(partition) + ext
    else:
        name = 'Q_p' + str(ident_player) + ext

    return name


def preprocess_state(state, width, height, k,l):
    # extract the grid of coordinates (k,l)
    if k == 0:
        a=1
        b=int(height/2)
    else:
        a=1+int(height/2)
        b=height-1

    if l == 0:
        c=1
        d=int(width/2)
    else:
        c=1+int(width/2)
        d=width-1

    return state[a:b,c:d,:]



def preprocess_state_layers(state, ident_agent, order_agents):
    (w,h,l) = np.shape(state)
    n_agents = int(l/2)
    rest_agents = list(set(range(n_agents))-set([ident_agent]+order_agents))


    new_state = np.zeros([w,h,n_agents+1], dtype=int)
    new_state[:,:,0] = state[:,:,ident_agent + n_agents]
    new_state[:,:,1] = state[:,:,ident_agent]
    for i,x in enumerate(order_agents):
        new_state[:, :, i + 2] = state[:,:,x]

    for i,x in enumerate(rest_agents):
        new_state[:, :, len(order_agents) + i + 2] = state[:, :, x]

    return new_state


def preprocess_state_layers_for_agent(state, ident_agent, order_agents):
    (w,h,l) = np.shape(state)
    n_agents = int(l/2)
    rest_agents = list(set(range(n_agents))-set(order_agents))

    new_state = np.zeros([w,h,n_agents+1], dtype=int)
    new_state[:,:,0] = state[:,:,ident_agent + n_agents]
    new_state[:,:,1] = state[:,:,order_agents[0]]
    for i,x in enumerate(order_agents[1:]):
        new_state[:, :, i + 2] = state[:,:,x]

    for i,x in enumerate(rest_agents):
        new_state[:, :, len(order_agents) + i + 1] = state[:, :, x]

    return new_state


def transform_reward(ident_agent, list_rewards, coeff_agents = None, index_agents=None, normalise = 1.0):
    n_agents = len(list_rewards)
    if index_agents is not None:
        assert coeff_agents is None
        coeff_agents = [1 if (i in index_agents or i==ident_agent) else 0 for i in range(n_agents)]


    return np.dot(list_rewards, coeff_agents)*normalise


def name_generic_model_list(n_agents, ext = '.pth'):
    alphabet = list(string.ascii_lowercase)
    list_output = []

    model_name = 'Q_gen_P_a_for_'
    list_output.append(model_name+'x'+ext)
    for i in range(n_agents-1):
        model_name += alphabet[i+1]
        list_output.append(model_name+ext)

    return list_output


def prob_matrix_to_coop_matrix(matrix):
    return np.array(matrix)


def replicate_env(name_env, env, seed):
    new_env = gym.make(name_env)
    new_env.reset()
    new_env.replicate(env, seed)
    return new_env
