# DISPLAY Tit-for-Tat : cooperation graphs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib import cm

# plt.style.use('dark_background')
ext = "png"


def draw_arrow_pair(ax, x1, y1, x2, y2, labelcolor1, labelcolor2, alpha1, alpha2, width, head_width,
                    head_length, between_arrow=0.2, around_dots=0.8):
    arrow_h_offset = 0.25  # data coordinates, empirically determined
    # max_arrow_length = 1 - 2 * arrow_h_offset
    # max_head_width = 2.5 * max_arrow_width
    # max_head_length = 2 * max_arrow_width
    arrow_params = {'length_includes_head': False, 'shape': 'left',
                    'head_starts_at_zero': True}

    dx, dy = (x2 - x1), (y2 - y1)
    d_scale = math.sqrt(dx ** 2 + dy ** 2)
    dx_s, dy_s = dx / d_scale, dy / d_scale
    distance_points = math.sqrt((x1 - x2) ** 2 + (y2 - y1) ** 2)

    x_pos, y_pos = x1 + dx_s * around_dots - between_arrow * dy_s, y1 + dy_s * around_dots + between_arrow * dx_s
    delta_x, delta_y = (distance_points - 2 * around_dots) * dx_s, (distance_points - 2 * around_dots) * dy_s
    ax.arrow(x_pos, y_pos, delta_x, delta_y, alpha=alpha1, width=width,
             head_width=head_width, head_length=head_length, shape='right', color=labelcolor1,
             length_includes_head=True, head_starts_at_zero=False)

    x_pos, y_pos = x2 - dx_s * around_dots + between_arrow * dy_s, y2 - dy_s * around_dots - between_arrow * dx_s
    delta_x, delta_y = -(distance_points - 2 * around_dots) * dx_s, -(distance_points - 2 * around_dots) * dy_s
    ax.arrow(x_pos, y_pos, delta_x, delta_y, alpha=alpha2, width=width,
             head_width=head_width, head_length=head_length, shape='right', color=labelcolor2,
             length_includes_head=True, head_starts_at_zero=False)


def show_text(ax, x1, y1, text, labelcolor1):
    text_gap = 1.5
    fts = 15
    d_scale = math.sqrt(x1 ** 2 + y1 ** 2)
    dx, dy = x1 / d_scale, y1 / d_scale
    x_pos, y_pos = x1 + dx * text_gap, y1 + dy * text_gap
    if dx > 0:
        if dy > 0:  # dx>0
            ax.text(x_pos, y_pos, text, horizontalalignment='left',
                    verticalalignment='bottom', fontsize=fts)
        else:  # dx>0 dy<0
            ax.text(x_pos, y_pos, text, horizontalalignment='left',
                    verticalalignment='top', fontsize=fts)
    else:
        if dy > 0:  # dx<0
            ax.text(x_pos, y_pos, text, horizontalalignment='right',
                    verticalalignment='bottom', fontsize=fts)
        else:  # dx<0 dy<0
            ax.text(x_pos, y_pos, text, horizontalalignment='right',
                    verticalalignment='top', fontsize=fts)


class Point:  pass


cmap = plt.cm.get_cmap('RdYlGn')


def show_cooperation_graph(ax, id_agents_list, mean_coop_rates, matrix_coop_rates,
                           max_rates=None, cmap='RdYlGn', scale=12, max_t=150, side_scale='right'):
    cmap = plt.cm.get_cmap(cmap)
    n = len(id_agents_list)
    scale = scale
    scale_n = n ** (-0.5)
    alpha = 1

    mean_coop_rates = np.array(
        [(n * np.mean(matrix_coop_rates[i, :]) - matrix_coop_rates[i, i]) / (n - 1) for i in range(n)])

    width_arrow = 0.8 * scale_n
    width_head_arrow = 2.5 * scale_n
    between_arrow = 0.5 * scale_n
    around_dots = 1.3 + 0.8 * scale_n
    size_dot = 2000 * scale_n ** 2
    pL = []

    for i in range(n):
        p = Point()

        if n ==4:
            p.x, p.y = scale * math.cos(3 * math.pi / 4 - i * 2 * math.pi / n), scale * math.sin(3 * math.pi / 4 - i * 2 * math.pi / n)
        else:
            p.x, p.y = scale * math.cos(i * 2 * math.pi / n), scale * math.sin(i * 2 * math.pi / n)

        pL.append(p)

    # ax = plt.axes()
    nPoints = len(pL)

    for i in range(nPoints):
        labercolor = cmap(mean_coop_rates[i])
        ax.scatter(pL[i].x, pL[i].y, s=size_dot, color=labercolor, alpha=alpha, edgecolors='black', linewidth=1)
        show_text(ax, pL[i].x, pL[i].y, id_agents_list[i], 'r')

        for j in range(i + 1, nPoints):
            labercolor1 = cmap(matrix_coop_rates[i, j])
            labercolor2 = cmap(matrix_coop_rates[j, i])
            alpha1 = max_rates[i, j]
            alpha2 = max_rates[j, i]
            draw_arrow_pair(ax, pL[i].x, pL[i].y, pL[j].x, pL[j].y, labercolor1, labercolor2,
                            alpha1, alpha2, width_arrow, width_head_arrow,
                            width_head_arrow, between_arrow=between_arrow, around_dots=around_dots)

    ax.axis('equal')
    ax.axis('off')

    ax.set_xlim(-1.2 * scale, 1.2 * scale)
    ax.set_ylim(-1.1 * scale, 1.1 * scale)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    plt.colorbar(sm, ax=ax, pad=0.2, shrink=0.6, aspect=10)

    # plt.colorbar.ColorbarBase(ax=ax, cmap=coolwarm, values=sorted(v),
    # orientation="horizontal")

    # plt.savefig(output_fig+'.svg', transparent=True)
    # plt.savefig(output_fig+'.png', transparent=True)

    # plt.show()


nA = 45
mean_coop_rates = 0.8 * np.ones(nA)
mean_coop_rates[3] = 0.1
mean_coop_rates[0] = 0.85
matrix_coop_rates = 0.1 * np.ones([nA, nA])
matrix_coop_rates[0, 1] = 0.9
matrix_coop_rates[0, 2] = 0.9
matrix_coop_rates[0, 3] = 0.5
matrix_coop_rates[0, 4] = 0.5

matrix_coop_rates[4, 1] = 0.5
matrix_coop_rates[4, 2] = 0.5
matrix_coop_rates[4, 3] = 0.9
matrix_coop_rates[4, 0] = 0.9


def show_histo(ax, values, labels, colors, initial_baselines=None, optimal_baselines=None, max_gain=None):
    n_agents = len(labels)
    #y_pos = np.arange(n_agents + 1)
    y_pos = np.arange(n_agents)
    max_bar = max_gain
    #ax.bar(y_pos, values + [max_bar], color=colors + ['white'], tick_label=labels + [''])
    ax.bar(y_pos, values, color=colors, tick_label=labels)
    # x = np.linspace(-0.5, n_agents, n_agents)

    if initial_baselines and optimal_baselines:
        x_indiv_baselines = [np.linspace(-0.5 + i, 0.5 + i, 2) for i in range(n_agents)]
        y_opt_baselines = [opt_value * np.ones(2) for opt_value in optimal_baselines]
        y_init_baselines = [init_value * np.ones(2) for init_value in initial_baselines]
        for i in range(n_agents):
            if i == 0:
                ax.plot(x_indiv_baselines[i], y_init_baselines[i], 'k-', label='initial')
                ax.plot(x_indiv_baselines[i], y_opt_baselines[i], 'k--', label='optimal')
            else:
                ax.plot(x_indiv_baselines[i], y_init_baselines[i], 'k-')
                ax.plot(x_indiv_baselines[i], y_opt_baselines[i], 'k--')
    # ax.legend()

    # ax.yticks(np.arange(0, max_bar, step=10))


def show_multiple_curves(ax, curves, labels, colors, xlim, ylim):
    for i, c in enumerate(curves):
        # print(len(c))
        ax.plot(range(len(c)), c, color=colors[i], label=labels[i],linewidth = 4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc=2)

    # ax.title(titl)  # Problemes avec accents (plot_directive) !


# plt.legend([p1, p2], ["Sinus", "Cosinus"])


def show_multiple_curves_sum_gain(ax, curves, optimal_gain, labels, label_other, colors, color_other, xlim, ylim,
                                  ylim_other):
    n_agents = len(labels)

    for i, c in enumerate(curves):
        # print(len(c))
        ax.plot(range(len(c)), c, color=colors[i], label=labels[i])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend(loc=2)

    t = range(len(c))
    n_agents = len(labels)
    s1 = range(len(c))
    x = np.arange(len(c))
    y_opti_sum = n_agents * (n_agents - 1) * optimal_gain * x

    curve_sum = []
    curves_np = np.array(curves)

    for k in range(len(c)):
        # print(curves_np[:,:k])
        # print(np.sum(curves_np[:,:k]))
        curve_sum.append(np.sum(curves_np[:, :k]))

    ax2 = ax.twinx()
    ax2.plot(x, y_opti_sum, 'w--', label="optimal sum")
    ax2.plot(t, curve_sum, 'r--', label="sum")
    ax2.set_ylim(ylim_other)

    ax2.legend(loc=4)
    # ax2.set_ylabel('sin', color='r')
    # ax2.tick_params('y', colors='r')

    # fig.tight_layout()


def show_global(mean_coop_rates,
                matrix_coop_rates,
                curves_coop_rates_agents,
                curves_gain_agents,
                labels_agents,
                colors_agents,
                max_t,
                max_gain,
                optimal_single_gain_R=1,
                output_fig='essaiFINAL'):
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))

    t = len(curves_gain_agents[0])
    n_agents = len(labels_agents)

    current_values_gains = [c[-1] for c in curves_gain_agents]
    current_coop_rates_gains = [c[-1] for c in curves_coop_rates_agents]
    current_mean_gains = sum(current_values_gains) / n_agents
    current_optimal_mean_gains = optimal_single_gain_R * (n_agents - 1)

    label_sum = "sum gains"
    color_sum = 'black'

    max_sum = optimal_single_gain_R * (n_agents - 1) * n_agents * max_t

    show_histo(axs[1, 1], current_values_gains, labels_agents, colors_agents, current_mean_gains,
               current_optimal_mean_gains, max_gain, max_t=max_t)
    show_multiple_curves(axs[0, 0], curves_coop_rates_agents, labels_agents, colors_agents, [0, max_t], [-0.1, 1.1])
    show_cooperation_graph(axs[0, 1], labels_agents, current_coop_rates_gains, matrix_coop_rates, cmap='RdYlGn')
    show_multiple_curves_sum_gain(axs[1, 0], curves_gain_agents, optimal_single_gain_R, labels_agents, label_sum,
                                  colors_agents, color_sum, [0, max_t], [3, 20], [0, max_sum])

    # axs[0, 0].set_title('Mean Cooperation Rates')
    # axs[0, 0].plot(x, y, 'tab:orange')
    # axs[0, 1].set_title('Cooperation Rates')
    # axs[1, 0].plot(x, -y, 'tab:green')
    # axs[1, 0].set_title('Axis [1,0]')
    # axs[1, 1].plot(x, -y, 'tab:red')
    # axs[1, 1].set_title('Axis [1,1]')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # fig.savefig('essaiFINAL_trans.png', transparent=True)
    fig.savefig(output_fig + '.' + ext, transparent=False)


#### RADAR CHARTS ####
# from https://www.python-graph-gallery.com

import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
df = pd.DataFrame({
    'group': ['A', 'B', 'C', 'D'],
    'var1': [38, 1.5, 30, 4],
    'var2': [29, 10, 9, 34],
    'var3': [8, 39, 23, 24],
    'var4': [7, 31, 33, 14],
    'var5': [28, 15, 32, 14]
})


def radar_chart(list_metrics, labels_metrics, labels_algos, output_fig):
    # ------- PART 1: Create background

    # number of variable
    categories = labels_metrics
    N = len(list_metrics[0])
    K = len(list_metrics)
    assert N == len(categories)
    assert K == len(labels_algos)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)
    # plt.set_varlabels(categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.50, 0.75, 1.0], ["", "0.5", "", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)

    labels_list = labels_algos

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

    colors = ['r', 'b', 'g'] + ['b'] * 10

    for i in range(K):
        values = list_metrics[i]
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=1, linestyle='solid', label=labels_list[i])
        ax.fill(angles, values, colors[i], alpha=0.05)

    plt.legend(loc=2, bbox_to_anchor=(-0.12, 1.06))

    plt.savefig(output_fig)




def generate_double_graph(coop_max_matrix, coop_rates_matrix, curves, output_name,
                          fig_size = (10, 10), labels = ["A", "B", "C", "D", "E", "F", "G"],
                          colors = None, limits_y = (-5,10), limits_x = (0,100)):
    yellow = True
    if colors is None:
        if yellow:
            colors = ['#dc6446', '#7daa55', '#326eb9', '#f5c341', '#966ec8']
        else:
            colors = ['#dc6446', '#7daa55', '#326eb9', '#966ec8', '#f5c341']


    current_payoffs = [c[-1] for c in curves]
    fig, axs = plt.subplots(2, 2, figsize=fig_size)
    (n,_) = np.shape(coop_rates_matrix)
    list_rates = [1.0]*n
    show_cooperation_graph(axs[0,0], labels, list_rates, coop_max_matrix, max_rates=np.ones([n,n]), cmap='Blues')
    axs[0,0].set_title('Maximum Cooperation Graph', fontsize=20)
    show_cooperation_graph(axs[0,1], labels, list_rates, coop_rates_matrix, max_rates=coop_max_matrix, cmap='Greens')
    axs[0,1].set_title('Current Cooperation Graph', fontsize=20)


    show_histo(axs[1,0], current_payoffs, labels, colors)
    axs[1,0].set_title('Current payoffs', fontsize=20)
    axs[1,0].set_ylim(limits_y)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    show_multiple_curves(axs[1,1], curves, labels, colors, xlim=limits_x, ylim=limits_y)
    axs[1,1].set_title('Current payoffs', fontsize=20)
    #axs[1,1].set_ylim(limits_y)


    fig.savefig(output_name, transparent=False)
    plt.clf()
    plt.close()





def generate_complete_display(coop_max_matrix, coop_rates_matrix, curves, output_name, detected_coop = None, detected_potential=None,
                          fig_size = (8, 20), labels = ["A", "B", "C", "D", "E", "F", "G"],
                          colors = None, limits_y = (-5,10), limits_x = (0,100)):
    yellow = True
    if colors is None:
        if yellow:
            colors = ['#dc6446', '#7daa55', '#326eb9', '#f5c341', '#966ec8']
        else:
            colors = ['#dc6446', '#7daa55', '#326eb9', '#966ec8', '#f5c341']

    if detected_coop is None:
        detected_coop = coop_rates_matrix
    if detected_potential is None:
        detected_potential = coop_max_matrix

    current_payoffs = [c[-1] for c in curves]
    fig, axs = plt.subplots(2, 3, figsize=(25,10))
    (n,_) = np.shape(coop_rates_matrix)
    list_rates = [1.0]*n
    show_cooperation_graph(axs[0,1], labels, list_rates, coop_max_matrix, max_rates=np.ones([n,n]), cmap='Blues', scale=8)
    axs[0,1].set_title('Potential Cooperation Graph', fontsize=15)
    show_cooperation_graph(axs[0,2], labels, list_rates, coop_rates_matrix, max_rates=coop_max_matrix, cmap='Greens', scale=8)
    axs[0,2].set_title('Current Cooperation Graph', fontsize=15)


    show_cooperation_graph(axs[1,1], labels, list_rates, detected_potential, max_rates=np.ones([n,n]), cmap='Blues', scale=8)
    axs[1,1].set_title('Detected Potential Graph', fontsize=15)
    show_cooperation_graph(axs[1,2], labels, list_rates, detected_coop, max_rates=coop_max_matrix, cmap='Greens', scale=8)
    axs[1,2].set_title('Detected Current Graph', fontsize=15)

    show_histo(axs[0,0], current_payoffs, labels, colors)
    axs[0,0].set_title('Current payoffs sum', fontsize=15)
    axs[0,0].set_ylim(limits_y)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    show_multiple_curves(axs[1,0], curves, labels, colors, xlim=limits_x, ylim=limits_y)
    axs[1,0].set_title('Evolution payoffs sum', fontsize=15)
    axs[1,0].set_ylim(limits_y)

    #plt.show()
    fig.savefig(output_name, transparent=False)
    plt.clf()
    plt.close()