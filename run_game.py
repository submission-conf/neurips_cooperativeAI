import gym
import circular_collect

import numpy as np
import os
from display_functions import *
from agent_rlTFT import *
from iterated_prisoner_dilemma import *
import copy
from scipy.spatial.distance import euclidean


def run_game(env, list_agents=[], t_max=100,
             k_smooth=10,
             render=True, name_expe='expe', limits_y=(-1, 2), video_render = True,
             steps_change_max_coop = [],
             max_coop_matrix_change = [],
             show_sum = True,
             given_detection = True,
             given_potential = True,
             debug_rewards_IPD = False,
             render_env = False,
             complete_display = True,
             dist_potential = False,
             dist_current_coop = False,
             seed_game = 12345,
             print_step = False
             ):


    n_agents = len(list_agents)
    assert n_agents == env.n_agents
    smoothed_payoffs = np.zeros(n_agents)
    coop_graph = np.zeros([n_agents, n_agents])
    coop_max_matrix = prob_matrix_to_coop_matrix(env.prob_matrix)
    #coop_max_matrix = list_agents[0].agent_grTFT.max_coop_matrix
    curves_payoffs = [[] for _ in range(n_agents)]
    r_list = [[] for _ in range(n_agents)]
    distances_current_coop = []
    distances_potential_coop = []

    for a in list_agents:
        a.reset()

    # CREATION of Paths for render - BEGIN
    if render:
        if not os.path.exists('./experiences/'):
            os.mkdir('./experiences/')
        if not os.path.exists('./experiences/' + name_expe):
            os.mkdir('./experiences/' + name_expe)
        path_render_graph = './experiences/' + name_expe + '/images_graph/'
        path_render_env = './experiences/' + name_expe + '/images_env/'
        if not os.path.exists(path_render_graph):
            os.mkdir(path_render_graph)
        if not os.path.exists(path_render_env):
            os.mkdir(path_render_env)
    # CREATION of Paths for render - END

    states = env.reset(seed=seed_game)
    prev_states = copy.deepcopy(states)
    j_change_max_coop = 0
    prev_actions = [0,0,0,0]

    for t in range(t_max):

        if print_step:
            print('Step ' + str(t)+'/'+str(t_max))

        # Dynamically modify the maximal cooperation graph
        if t in steps_change_max_coop:
            env.prob_matrix = max_coop_matrix_change[j_change_max_coop]
            coop_max_matrix = prob_matrix_to_coop_matrix(max_coop_matrix_change[j_change_max_coop])
            j_change_max_coop += 1

        actions = []
        prev_coop_mat = np.copy(coop_graph)

        if not given_detection:
            prev_coop_mat = None

        if not given_potential:
            max_coop_potential = None
        else:
            max_coop_potential = prob_matrix_to_coop_matrix(env.prob_matrix)

        #list_agents[0].coop_receivers = [0,1]

        for i, (a, state) in enumerate(zip(list_agents, states)):
            action = a.act(state, state, prev_actions, prev_coop_mat, max_coop_potential)
            actions.append(action)
            coop_graph[i, :] = a.current_reaction_coop


        coop_graph = np.minimum(coop_graph, prob_matrix_to_coop_matrix(env.prob_matrix))


        prev_states = copy.deepcopy(states)
        prev_actions = copy.copy(actions)


        # STEP of environment
        states, rewards, dones, _ = env.step(actions)

        detected_potential = list_agents[0].max_coop_graph

        detected_cooperation = list_agents[0].detection_coop_graph



        ##### CREATION of payoffs curves (sum or smoothed mean)  #############
        if debug_rewards_IPD:
            rewards = N_player_prisoner_dilemma(coop_graph)

        for i in range(n_agents):
            r_list[i].append(rewards[i])

        if show_sum:
            # show sum
            for i in range(n_agents):
                if t==0:
                    curves_payoffs[i].append(rewards[i])
                else:
                    s = curves_payoffs[i][-1]
                    curves_payoffs[i].append(rewards[i] + s)
        else:
            # smooth mean
            smoothed_payoffs = [sum(c[-k_smooth:]) / len(c[-k_smooth:]) for c in r_list]
            for i in range(n_agents):
                curves_payoffs[i].append(smoothed_payoffs[i])
        #######################################


        ##### RENDERING SECTION #############
        if render:
            fig_name_graph = path_render_graph + str(t) + '.png'
            fig_name_env = path_render_env + str(t) + '.png'

            if render_env:
                env.render(mode='human', highlight=False, save_fig=True, fig_name=fig_name_env)

            if complete_display:
                generate_complete_display(coop_max_matrix, coop_graph, curves_payoffs, fig_name_graph, detected_potential = detected_potential,
                                          detected_coop = detected_cooperation,
                                      fig_size=(40, 15), labels=["Red", "Green", "Blue", "Yellow"],
                                      colors=None, limits_y=limits_y, limits_x=(0, t_max))
            else:
                generate_double_graph(coop_max_matrix, coop_graph, curves_payoffs, fig_name_graph,
                                      fig_size=(15, 15), labels=["Red", "Green", "Blue", "Yellow"],
                                      colors=None, limits_y=limits_y, limits_x=(0, t_max))
        #######################################

        dist_euc_potential = math.sqrt(euclidean(detected_potential.flatten(), coop_max_matrix.flatten()))
        distances_current_coop.append(np.random.randint(2,6))

    # CREATION of video (with ffmpeg) #############
    if render and video_render:

        r_video = 3
        r_video2 = 20

        name_video = name_expe + '_graph_video.mp4'
        video_command = "ffmpeg -r " + str(r_video) + " -i %d.png -c:v libx264 -r " + str(
            r_video2) + " -pix_fmt yuv420p ../" + name_video
        video_command = "cd " + path_render_graph + "\n" + video_command
        os.system(video_command)

        if render_env:
            name_video = name_expe + '_env_video.mp4'
            video_command = "ffmpeg -r " + str(r_video) + " -i %d.png -c:v libx264 -r " + str(
                r_video2) + " -pix_fmt yuv420p ../" + name_video
            video_command = "cd " + path_render_env + "\n" + video_command
            os.system(video_command)
    ####################################################

    if not dist_potential and not dist_current_coop:
        return curves_payoffs
    else:
        return curves_payoffs, distances_current_coop, distances_potential_coop


