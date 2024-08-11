from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx
import networks
import algorithms as alg
import experiments as exp
import pandas as pd
import probability as prob
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.ticker import NullLocator
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
import itertools

##############################
# Code for the paper figures #
# some code remains unused   #
# in the final paper. Method #
# names generally do not     #
# correspond to the actual   #
# plots in the paper.        #
##############################

# COLOR SCHEME
# https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=6
COLORS = [
    "#d73027",
    "#fc8d59",
    "#fee090",
    "#e0f3f8",
    "#91bfdb",
    "#4575b4"
]

# plot defaults
LINEWIDTH = 3
FONT_SIZE = 14
FONT = {'fontname':'Times New Roman'}
PATH = './'
FIG_SIZE_LARGE = (12, 6)
FIG_SIZE_SMALL = (6, 3)

# needed for some overrides
algo_dict = {
        "random": False,
        "myopic": True,
        "naive_myopic": False,
        "gonzales": False,
        "furthest_non_seed_0": False,
        "furthest_non_seed_1": False,
        "bfs_myopic": False,
        "naive_bfs_myopic": False,
        "ppr_myopic": False,
        "naive_ppr_myopic": False,
        "degree_lowest_centrality_0": False,
        "degree_lowest_centrality_1": False,
        "degree_highest_degree_neighbor_0": False,
        "degree_highest_degree_neighbor_1": False,
    }


ALGO_RENAME = {
    'random': 'Random',
    'myopic': 'Myopic',
    'naive_myopic': 'Naive Myopic',
    'gonzales': 'Gonzales',
    'furthest_non_seed_0': 'LeastCentral',
    'furthest_non_seed_1': 'LeastCentral_n',
    'bfs_myopic': 'Myopic BFS',
    'naive_bfs_myopic': 'Naive Myopic BFS',
    'ppr_myopic': 'Myopic PPR',
    'naive_ppr_myopic': 'Naive Myopic PPR',
    'degree_lowest_centrality_0': 'MinDegree_hc',
    'degree_lowest_centrality_1': 'MinDegree_hcn',
    'degree_highest_degree_neighbor_0': 'MinDegree_nd',
    'degree_highest_degree_neighbor_1': 'MinDegree_ndn',
    'inconclusive' : 'Inconclusive'
}


def fit_line(x, y, intercept=None):
    # fit a line to the data
    a, b = np.polyfit(x, y, 1)

    # return the slope and the fitted y values
    return [a], a * x + b, b

def run_fig1():
    # <k> vs ln(n) with points representing encoded domains

    # load corpus dataframe
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # drop last row
    df = df.drop(df.tail(1).index)

    # start a plot, wide figure
    plt.figure(figsize=(6,3))

    # get unique domains
    domains = df['networkDomain'].unique()

    # map domains to special symbols
    domain_map = {
        'Social': 'o',
        'Economic': 's',
        'Biological': 'v',
        'Informational': '^',
        'Technological': 'D',
        'Transportation': 'P'
    }

    domain_colors = {
        'Social': COLORS[0],
        'Economic': COLORS[1],
        'Biological': COLORS[2],
        'Informational': COLORS[3],
        'Technological': COLORS[4],
        'Transportation': COLORS[5]
    }

    for i, row in df.iterrows():
        # create a graph from the edge list
        nodes = df.iloc[i]['nodes_id']
        edges = df.iloc[i]['edges_id']
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # compute average degree
        avg_degree = np.mean([G.degree(node) for node in G.nodes()])


        # plot with a marker and color, opacity 0.5
        plt.scatter(len(G.nodes()), avg_degree, marker=domain_map[df.iloc[i]['networkDomain']], color=domain_colors[df.iloc[i]['networkDomain']], alpha=0.5)

    # plot legend outside of the plot to the right
    plt.tick_params(axis='both',labelsize=FONT_SIZE)

    # plot legend inside the plot
    plt.legend([plt.scatter([], [], marker=domain_map[domain], color=domain_colors[domain]) for domain in domains], domains, loc='lower right', fontsize=FONT_SIZE, borderpad=0.2)

    # add labels
    plt.ylabel(r'average degree, $\langle k \rangle$', fontsize = FONT_SIZE)
    plt.xlabel(r'number of nodes, $n$', fontsize = FONT_SIZE)

    # override xticks to be in powers of 10 with 5 ticks
    plt.xscale('log')
    plt.yscale('log')

    # add more ticks
    plt.xticks([10**3, 10**4, 10**5])

    plt.tight_layout()

    # save the plot
    plt.savefig(PATH + '/previews/fig1.png', bbox_inches='tight', dpi=300)
    plt.savefig(PATH + 'fig1.pdf', bbox_inches='tight', dpi=300)

def run_fig2():
    # generate a plot with 7 nodes

    G = nx.Graph()
    G.add_nodes_from(range(7))
      # add edges
    G.add_edges_from([(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (1,2), (1,3), (1,4)])

    # run myopic algorithm on this network
    a = alg.get_algorithm('myopic')
    algo = a(G, k=2, p=0.5, ic_trials=1000, use_cache=False)
    algo.override_seeds([0])

    plt.figure(figsize=(6,6))

    # plot the graph circular
    nx.draw_circular(G, with_labels=False, node_color=COLORS[-3])

    plt.margins(x=0)
    plt.margins(y=0)

    # highlight node 6 in green
    nx.draw_networkx_nodes(G, pos=nx.circular_layout(G), nodelist=[6], node_color=COLORS[-1])
    # add label

    # highlight node 0 in red
    nx.draw_networkx_nodes(G, pos=nx.circular_layout(G), nodelist=[0], node_color=COLORS[0])

    # highlight node 5 in cyan
    nx.draw_networkx_nodes(G, pos=nx.circular_layout(G), nodelist=[5], node_color=COLORS[-2])

    # legend
    plt.legend([plt.scatter([], [], marker='o', color=COLORS[-3]), plt.scatter([], [], marker='o', color=COLORS[-1]), plt.scatter([], [], marker='o', color=COLORS[0]), plt.scatter([], [], marker='o', color=COLORS[-2])], ['Unselected nodes', 'Myopic selection', 'Initial seed', 'Lowest degree,\nlowest harmonic centrality'], bbox_to_anchor=(1.05, 1), loc='best', fontsize=FONT_SIZE*2)

    # save the figure
    plt.savefig(PATH + 'fig2.png', bbox_inches='tight', dpi=300)

def run_fig3a():
    k = 10

    # color dictionary
    color_dict = {
        'myopic': COLORS[0],
        'bfs_myopic' : COLORS[-1],
        'gonzales' : COLORS[-2],
    }

    symbol_dict = {
        'myopic': 'o',
        'bfs_myopic' : 's',
        'gonzales' : 'X',
    }

    # load evaluations one by one and plot
    
    # start plt figure
    plt.figure()

    # load in the corpus
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # get list of files in cache/evaluations
    files = os.listdir('./cache/evaluations')

    for filename in files:
        # parse out the name
        hashed_network_name = filename.split('_')[0]

        # get the index of the network in the corpus
        index = df[df['hashed_network_name'] == hashed_network_name].index[0]

        if hashed_network_name == '1697dbe2c9c52899aab4584aa3fe4f65':
            # get some stats on the one network we care about
            net = df[df['hashed_network_name'] == hashed_network_name]

        with open(os.path.join("./cache/evaluations/", filename), 'rb') as f:

            if filename != "1697dbe2c9c52899aab4584aa3fe4f65_04_0.4.npy":
                continue

            # load file as a dictionary
            d = np.load(f, allow_pickle=True)

            # convert to a regular dictionary
            d = d.item()

            # clear plt
            plt.clf()
            plt.figure(figsize=(6,3))
            plt.margins(x=0)
            plt.margins(y=0)
            algos = ['myopic','bfs_myopic', 'gonzales']
            betas = [[],[],[],[]]
            intercepts = [[],[],[],[]]

            for i, algo in enumerate(algos):
                vals = d[algo]

                for v in vals:
                # compute the line of best fit
                    a, yfit, b = fit_line(np.array(range(0, len(v))), np.array(v), intercept=v[0])

                    # store slope (henceforth beta)
                    betas[i].append(a[0])
                    intercepts[i].append(b)

                # average over vals
                avg = np.mean(vals, axis=0)
    
                # plot the curve
                plt.plot(avg, color=color_dict[algo], label=f'{ALGO_RENAME[algo]}', alpha=1, linewidth=2, marker=symbol_dict[algo], markersize=10, markevery=[4,8])

                slope = np.mean(betas[i])
                intercept = np.mean(intercepts[i])

                # Generate x values from 0 to 10
                x_values = np.linspace(0, 10, 100)

                # Calculate corresponding y values using the linear equation: y = ax + b
                y_values = slope * x_values + intercept

                # Plot the linear regression line
                plt.plot(x_values, y_values, color=color_dict[algo], linewidth = 2, alpha = 0.5, linestyle='dashed')
                
            # legend in bottom right, a few pixels up
            plt.legend(fontsize=FONT_SIZE, loc='lower right', bbox_to_anchor=(1, 0.13))

            # pi in latex on y-axis
            plt.tick_params(axis='both',labelsize=FONT_SIZE)

            plt.ylabel(r'min access prob., $\pi_{min}$',fontsize=FONT_SIZE)

            plt.xlabel('seed set size, k', fontsize=FONT_SIZE)

            # xticks
            plt.xticks(list(range(0,11)))

            # yticks
            plt.yticks(list(np.arange(0.30, 1.05, 0.1)))

            # set tight layout
            plt.tight_layout()

            # png preview for editing
            plt.savefig(f'{PATH}/previews/fig3a.png', bbox_inches='tight', dpi=300)

            # save as eps
            plt.savefig(f'{PATH}/fig3a.eps', bbox_inches='tight', dpi=300)

            if filename == "1697dbe2c9c52899aab4584aa3fe4f65_03_0.3.npy":
                break

    
def run_fig3b():
    k = 10

    y_algo_pos = {
        'random': 0,
        'myopic': 1,
        'naive_myopic': 2,
        'gonzales': 3,
        'furthest_non_seed_0': 4,
        'furthest_non_seed_1': 5,
        'bfs_myopic': 6,
        'naive_bfs_myopic': 7,
        'ppr_myopic': 8,
        'naive_ppr_myopic': 9,
        'degree_lowest_centrality_0': 10,
        'degree_lowest_centrality_1': 11,
        'degree_highest_degree_neighbor_0': 12,
        'degree_highest_degree_neighbor_1': 13,
    }

    title_dict = {
        '03': 'p=0.3',
        '04': 'p=0.4',
        '05': 'p=0.5',
        'low': 'p_low',
        'med': 'p_med',
        'high': 'p_high',
    }

    #p_tags = ['03', '04', '05', 'low', 'med', 'high']
    p_tag = 'high'

    # read in corpus dataframe
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # get the list of all files in ./cache/evaluations
    files = os.listdir('./cache/evaluations')
    file_p_tags = []
    file_net_names = []

    # parse out the p value tags and names from filenames
    for f in files:
        file_net_names.append(f.split('_')[0])
        file_p_tags.append(f.split('_')[1])

    # get the indices of the files with the p tag
    indices = [i for i, x in enumerate(file_p_tags) if x == p_tag]

    skips = 0 # counter for how many networks were skipped, helps adjust the figure's x-axis

    domains = ['Biological', 'Social', 'Economic', 'Technological', 'Transportation', 'Informational']

    errs_temp = []

    # for the degree v 'best' plot
    degrees = []
    sizes = []
    diameters = []
    
    best_algos = []
    best_algos_0 = []

    result_colors = {}
    result_myopic_best = {}
    for domain in domains:
        result_colors[domain] = {}
        result_myopic_best[domain] = []
        for algo in algo_dict.keys():
            result_colors[domain][algo] = []

    for i, domain in enumerate(domains):
        plotted = 0

        # iterate over the dataframe
        for j, row in df.iterrows():
            ready = False
            file_index = -1

            # is this net in the domain we want?
            if row['networkDomain'] != domain:
                continue

            # geytickst the hashed name of the network
            hashed_network_name = row['hashed_network_name']

            # is this network ready to go?
            for k in indices:
                if hashed_network_name in file_net_names[k]:

                    ready = True
                    file_index = k

            if not ready:
                skips += 1
                continue
            else:

                # read in the file
                with open(os.path.join("./cache/evaluations/", files[file_index]), 'rb') as f:
                    # load file as a dictionary
                    d = np.load(f, allow_pickle=True)

                    # convert to a regular dictionary
                    d = d.item()

                    algo_performance = {}

                    # for the avg degree vs best figure
                    best_algo = 'random'
                    best_value = 0
                
                    for algo in algo_dict.keys():
                        evals = d[algo]

                        for e in evals:
                            vals = e[:k]

                            # compute line of best fit
                            a, yfits, _ = fit_line(np.array(range(0, len(vals))), np.array(vals))

                            # store the slope
                            if algo not in algo_performance.keys():
                                algo_performance[algo] = []

                            algo_performance[algo].append(a[0])

                        algo_performance[algo] = np.mean(algo_performance[algo])

                        # is this better than the best performance so far?
                        # or if this matches and isn't myopic, also set as best
                        if algo_performance[algo] > best_value or (algo_performance[algo] == best_value and algo != 'myopic'):
                            best_value = algo_performance[algo]
                            best_algo = algo

                    if best_value > 0:
                        best_algos_0.append(best_algo)
                    else:
                        best_algos_0.append('inconclusive')

                    myopic_perf = algo_performance['myopic']

                    error = np.sqrt(myopic_perf * (1 - myopic_perf) / 20000)

                    # normalize
                    for algo in algo_performance.keys():
                        if myopic_perf > 0:
                            # are we within the error?
                            if np.abs(algo_performance[algo] - myopic_perf) > error:
                                algo_performance[algo] = algo_performance[algo] / myopic_perf
                            else:
                                # too close to tell / inconclusive
                                algo_performance[algo] = -1

                        elif algo_performance[algo] > 0:
                            algo_performance[algo] = 1.5 # set to some value above 1 for later, since this outperforms myopic
                        
                        else:
                            algo_performance[algo] = -1 # inconclusive


                myopic_best = 1

                good_for_avg_deg_plot = []

                for algo in algo_performance.keys():
                    color = ''
                    # compute the color
                    if algo == 'myopic':
                        color = '#808080' # grey for myopic
                    elif algo_performance[algo] == -1:
                        color = COLORS[-2] # inconclusive
                    elif algo_performance[algo] > 0.8 and algo_performance[algo] < 1.0:
                        color = COLORS[-3] # slightly worse than Myopic
                    elif algo_performance[algo] < 1.0:
                        color = COLORS[1] # worse than myopic
                    else:
                        color = COLORS[-1] # better than myopic

                    # if worse than myopic, not even within 80%
                    # we want to track it
                    if color != COLORS[1] and color != '#808080':
                        myopic_best = 0
                    
                    # if the algo is better or ties with myopic, track it for the avg degree vs best plot
                    if color == COLORS[-1] or color == COLORS[-2]:
                        good_for_avg_deg_plot.append(algo)
                    
                    # save color to global
                    result_colors[domain][algo].append(color)

                result_myopic_best[domain].append(myopic_best)

                # for the avg degree vs best plot
                # (pretty sure I am not using this anymore)
                if good_for_avg_deg_plot == []:
                    good_for_avg_deg_plot.append('myopic')
                best_algos.append(good_for_avg_deg_plot)

    print(best_algos_0)

    # what fraction of the networks sees myopic perform best?
    temp_best = 0
    temp_total = 0
    for d in domains:
        temp_best += sum(result_myopic_best[d])
        temp_total += len(result_myopic_best[d])
    
    print("Fraction of the corpus dominated by Myopic:", temp_best / temp_total)

    # total number of networks processed across all domains
    n_nets = len(df) - skips
    widths = []

    for domain in domains:
        # compute width of each column based on number of networks in domain
        widths.append(len(result_colors[domain]['random']) / n_nets)

    # setup
    plt.margins(x=0)
    plt.margins(y=0)

    # start plt figure with 1 row, one column for each domain
    if p_tag != 'low':
        fig, axs = plt.subplots(1, len(domains), sharey=True, gridspec_kw={'width_ratios':widths}, figsize=(20,5.1))
    else:
        fig, axs = plt.subplots(1, len(domains), sharey=True, gridspec_kw={'width_ratios':widths}, figsize=(20,4.8))

    #axs[0].plot([-40], [0], alpha=0)  # Invisible point for padding

    for i, domain in enumerate(domains):

        # set title, offset slightly upward
        if p_tag in ['low', 'med']:
            axs[i].set_title(domain, fontsize=FONT_SIZE*1.5, loc='center', pad=15, rotation=-10, ha='right')

        # set xticks to domain length
        axs[i].set_xticks(np.arange(0, len(result_colors[domain]['random']), 1))

        for algo in algo_dict.keys():
            for x, color in enumerate(result_colors[domain][algo]):
                # plot filled rectangle
                #axs[i].fill([plotted, plotted, plotted+1, plotted+1], [y_algo_pos[algo], y_algo_pos[algo]+1, y_algo_pos[algo]+1, y_algo_pos[algo]], color=color)

                rect = mpatches.Rectangle((x, y_algo_pos[algo]), 1, 1, color=color, clip_on=False)

                axs[i].add_patch(rect)

        # more setup
        axs[i].set_xlabel(' ')
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        
        axs[i].set_xticks([])

        axs[i].yaxis.set_tick_params(width=2, length=7)
        

    # equal axis scale
    # plt.axis('equal')

    # set y-ticks to be the algo names
    # put together renamed algo list
    renamed_algos = [ALGO_RENAME[algo] for algo in algo_dict.keys()]

    # set y-ticks

    # draw y ticks manually
    axs[0].set_yticks(np.arange(0.5, len(y_algo_pos)+0.5, 1), renamed_algos, fontsize=FONT_SIZE*1.5)

    # classes:
    # myopic: grey
    # better than myopic: blue (COLORS[-1])
    # worse than myopic: orange (COLORS[1])
    # inconclusive: skyblue (COLORS[-2])
    # within 80% of myopic: white (COLORS[-3])

    if p_tag in ['high', 'med']:
        plt.legend([mpatches.Patch(color='#808080'), mpatches.Patch(color=COLORS[-1]), mpatches.Patch(color=COLORS[1]), mpatches.Patch(color=COLORS[-2]), mpatches.Patch(color=COLORS[-3])], ['Myopic', 'Better than Myo.', 'Worse than Myo.', 'Equivalent', 'Within 80% of Myo.'], bbox_to_anchor=(-0.7, -0.01), ncol=5, fontsize=FONT_SIZE*1.5)

    # saving is disabled for now
    # save at path
    plt.savefig(f'{PATH}/previews/fig3b_{p_tag}.png', bbox_inches='tight')
    # save as pdf
    plt.savefig(f'{PATH}/fig3b_{p_tag}.pdf', bbox_inches='tight')
        
#########################
def run_fig3c():
    # color dictionary for each algorithm with 14 easily distinguishable colors
    color_dict = {
        'random': 'red',
        'myopic': '#6495ED',
        'naive_myopic': '#87CEEB',
        'gonzales': 'orange',
        'furthest_non_seed_0': '#FFA500',
        'furthest_non_seed_1': '#FF8C00',
        'bfs_myopic': '#8A2BE2',
        'naive_bfs_myopic': '#9370DB',
        'ppr_myopic': '#FF69B4',
        'naive_ppr_myopic': '#FF1493',
        'degree_lowest_centrality_0': '#006400',  # Dark Green
        'degree_lowest_centrality_1': '#32CD32',  # Lime Green
        'degree_highest_degree_neighbor_0': '#CD5C5C',
        'degree_highest_degree_neighbor_1': '#B22222',
    }


    symbol_dict = {
        'random' : 'o',
        'myopic' : 's',
        'naive_myopic' : 'v',
        'gonzales' : '^',
        'furthest_non_seed_0' : 'D',
        'furthest_non_seed_1' : 'P',
        'bfs_myopic' : 'o',
        'naive_bfs_myopic' : 's',
        'ppr_myopic' : 'v',
        'naive_ppr_myopic' : '^',
        'degree_lowest_centrality_0' : 'D',
        'degree_lowest_centrality_1' : 'P',
        'degree_highest_degree_neighbor_0' : 'o',
        'degree_highest_degree_neighbor_1' : 's',
    }

    k = 10

    # compute an offset for each algo's symbol
    symbol_offset_dict = {}

    # compute spacing between symbols
    spacing = (k - 1) / len(symbol_dict.keys())

    for i, algo in enumerate(symbol_dict.keys()):
        symbol_offset_dict[algo] = i * spacing


    # p_tags = ['03', '04', '05', 'low', 'med', 'high']
    p_tags = ['low', 'med']
    #p_tags = ['high']

    # read in corpus dataframe
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # get the list of all files in ./cache/evaluations
    files = os.listdir('./cache/evaluations')
    file_p_tags = []
    file_net_names = []
    domains = ['Biological', 'Social', 'Economic', 'Technological', 'Transportation', 'Informational']

    # parse out the p value tags and names from filenames
    for f in files:
        file_net_names.append(f.split('_')[0])
        file_p_tags.append(f.split('_')[1])

    for fig_ix, p in enumerate(p_tags):
        domain_algo_performance_dict = {}
        for d in domains:
            domain_algo_performance_dict[d] = {}

        # get the indices of the files with the p tag
        indices = [i for i, x in enumerate(file_p_tags) if x == p]

        # go over every file with the p tag
        for i in indices:
            # get the name of the network
            name = file_net_names[i]

            # get the domain of the network
            domain = df[df['hashed_network_name'] == name]['networkDomain'].values[0]

            # open the file
            with open(os.path.join("./cache/evaluations/", files[i]), 'rb') as f:

                # load file as a dictionary
                d = np.load(f, allow_pickle=True)

                # convert to a regular dictionary
                d = d.item()

                for algo in algo_dict.keys():
                    evals =  d[algo]

                    for e in evals:
                        vals = e[:k]

                        # compute line of best fit
                        a, _, _ = fit_line(np.array(range(0, len(vals))), np.array(vals), intercept=vals[0])

                        # store the slope
                        if algo not in domain_algo_performance_dict[domain].keys():
                            domain_algo_performance_dict[domain][algo] = []

                        domain_algo_performance_dict[domain][algo].append(a[0])

        # start plt figure with 1 row, one column for each domain
        fig, axs = plt.subplots(1, len(domain_algo_performance_dict.keys()), sharey=True, figsize=(20, 5))


        for i, domain in enumerate(domain_algo_performance_dict.keys()):
            for key in domain_algo_performance_dict[domain].keys():
                a_mean = np.mean(np.array(domain_algo_performance_dict[domain][key]))

                # plot the line
                axs[i].plot(range(0, k), [a_mean] * k, label=key, color=color_dict[key], alpha=0.7, linewidth=3, zorder=1)

                # plot the label on each line
                axs[i].scatter(symbol_offset_dict[key], a_mean, marker=symbol_dict[key], color=color_dict[key], alpha=1, zorder=10, s = 75) # zorder to plot on top of the line


            # set title if it's the first figure
            if fig_ix == 0:
                axs[i].set_title(domain, fontsize=FONT_SIZE*1.5)

            # set x-axis label
            axs[i].set_xlabel(' ')

        # tight layout
        fig.tight_layout()

        # create renaming for the legend with ALGO_RENAME
        renamed = [ALGO_RENAME[algo] for algo in symbol_dict.keys()]

        # plot legend below the plot if it's the last figure
        if fig_ix == len(p_tags) - 1:
            fig.legend([plt.scatter([], [], marker=symbol_dict[algo], color=color_dict[algo], s = 75) for algo in symbol_dict.keys()], renamed, bbox_to_anchor=(0.48, 0.1), loc='upper center', ncol=7, fontsize=FONT_SIZE*1.5, columnspacing=0.5, handletextpad=0)

        # set yticks size
        plt.tick_params(axis='both',labelsize=FONT_SIZE*1.5)

        # remove x ticks
        # set y ticks
        for ax in list(axs):
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE*1.5)
            ax.set_xticks([])

            if p == 'low':
                ax.set_yticks(np.arange(0.000, 0.007, 0.001))

            if p == 'med':
                ax.set_yticks(np.arange(0.00, 0.021, 0.002))

            if p == 'high':
                ax.set_yticks(np.arange(0.00, 0.035, 0.005))

        # plotting <beta>
        fig.text(-0.04, 0.5, r'average slope, $\langle \beta \rangle$', va='center', rotation='vertical', fontsize=FONT_SIZE*1.5)

        # save the figure
        plt.savefig(f'{PATH}/previews/fig3c_{p}.png', bbox_inches='tight', dpi=300)

        # save as eps
        plt.savefig(f'{PATH}/fig3c_{p}.pdf', bbox_inches='tight', dpi=300)

        plt.clf() # clear plt figure
        plt.close()


def run_fig3d():

    algo_performance_dict = {}

    p = 0.5

    k = 10

    for algo in algo_dict.keys():
        algo_performance_dict[algo] = []

    # read every file in the cache/evaluations directory
    for filename in os.listdir('./cache/evaluations'):

        # load only if filename ends with p
        if filename.endswith(str(p) + '.npy'):
            # load file
            with open(os.path.join("./cache/evaluations/", filename), 'rb') as f:
                # load file as a dictionary
                d = np.load(f, allow_pickle=True)

                # convert to a regular dictionary
                d = d.item()

                for key in d.keys():
                    # compute line of best fit
                    
                    vals = d[key][:k]

                    b = np.polyfit(range(0, len(vals)), vals, 1)


                    algo_performance_dict[key].append(b)            

    # start plt figure
    plt.figure()

    for key in algo_performance_dict.keys():
        a = np.mean(np.array(algo_performance_dict[key])[:,0])
        b = np.mean(np.array(algo_performance_dict[key])[:,1])

        # plot line of best fit
        plt.plot(range(0, k), a * range(0, k) + b, label=key)


    # add legend
    plt.legend()

    plt.title(f'p = {p}')

    # pi in latex on y-axis
    plt.ylabel(r'$\pi$ (min access probability)')
    plt.xlabel('k (seed set size)')

    plt.show()



def run_fig_spreadability():
    LOW = 0.2
    MED = 0.5
    HIGH = 0.8

    # precomputed run results
    spreadability = [0.0022606000900268555, 0.0047021999359130855, 0.012935999870300294, 0.03092679977416992, 0.04279119873046875, 0.05466400146484375, 0.1026897964477539, 0.09930760192871094, 0.09864920043945312, 0.11987999725341797, 0.1656342010498047, 0.13339340209960937, 0.1553354034423828, 0.1644739990234375, 0.1949857940673828, 0.23505499267578125, 0.21395460510253905, 0.2069246063232422, 0.18147000122070311, 0.24406979370117188, 0.2685675964355469, 0.34333319091796877, 0.30625320434570313, 0.29159298706054687, 0.3336925964355469, 0.327755615234375, 0.3564092102050781, 0.3824216003417969, 0.362622802734375, 0.360422607421875, 0.4028869934082031, 0.3761261901855469, 0.4710419921875, 0.4274303894042969, 0.4696409912109375, 0.47800259399414063, 0.5072438049316407, 0.4663971862792969, 0.5497991943359375, 0.45967620849609375, 0.4826588134765625, 0.5024992065429688, 0.53472021484375, 0.65612060546875, 0.5135023803710937, 0.5938109741210937, 0.5952562255859375, 0.6057122192382812, 0.6170239868164062, 0.6700513916015625, 0.6653060302734375, 0.672041015625, 0.6957982177734375, 0.6405737915039063, 0.662593994140625, 0.7233989868164062, 0.7217335815429687, 0.7069407958984375, 0.6155568237304687, 0.6387296142578125, 0.7212996215820312, 0.7365811767578125, 0.7853201904296875, 0.79927978515625, 0.7771519775390625, 0.824763427734375, 0.8254788208007813, 0.8248576049804688, 0.8464022216796875, 0.8516276245117187, 0.879239013671875, 0.805905029296875, 0.8478671875, 0.8628547973632813, 0.8337918090820312, 0.8541511840820313, 0.8349251708984375, 0.8549307861328125, 0.8805897827148438, 0.913234375, 0.89942138671875, 0.9410136108398437, 0.9269509887695313, 0.9545859985351562, 0.931893798828125, 0.9226671752929687, 0.93978857421875, 0.9491771850585937, 0.9492415771484375, 0.953644775390625, 0.97038720703125, 0.9357999877929688, 0.978100830078125, 0.9727523803710938, 0.975649169921875, 0.985892578125, 0.9740195922851562, 0.996150390625, 0.998152587890625, 1.0]

    p_vals = list(np.arange(0.01, 1.01, 0.01))

    # fit a curve to the spreadability values
    # what we really care about for that figure is an easy to understand visualization
    curve = np.polyfit(p_vals, spreadability, 3)

    # replace the spreadability values with the curve
    spreadability = np.polyval(curve, p_vals)

    # get p_low, p_med, p_high from the curve
    p_low = p_vals[np.argmin(np.abs(spreadability - LOW))]
    p_med = p_vals[np.argmin(np.abs(spreadability - MED))]
    p_high = p_vals[np.argmin(np.abs(spreadability - HIGH))]

    print(f'p_low = {p_low}, spread = {spreadability[p_vals.index(p_low)]}')
    print(f'p_med = {p_med}, spread = {spreadability[p_vals.index(p_med)]}')
    print(f'p_high = {p_high}, spread = {spreadability[p_vals.index(p_high)]}')

    # plot the spreadability values vs  p

    # lines to mark the spreadability values
    low_line = [LOW] * len(p_vals[:p_vals.index(p_low)+1])
    mid_line = [MED] * len(p_vals[:p_vals.index(p_med)+1])
    high_line = [HIGH] * len(p_vals[:p_vals.index(p_high)+1])

    plt.figure(figsize=(6,3))
    plt.margins(x=0)
    plt.margins(y=0)

    plt.plot(p_vals, spreadability, label='spreadability', linewidth = LINEWIDTH, color=COLORS[0])

    # dashed lines till p_low, p_med, p_high
    plt.plot(p_vals[:p_vals.index(p_low)+1], low_line, '--', color=COLORS[-1], linewidth = LINEWIDTH,label='low spreading')
    plt.plot(p_vals[:p_vals.index(p_med)+1], mid_line, '--', color=COLORS[-2], linewidth = LINEWIDTH, label='med. spreading')
    plt.plot(p_vals[:p_vals.index(p_high)+1], high_line, '--', color=COLORS[1], linewidth = LINEWIDTH, label='high spreading')

    # dashed lines at the p values

    plt.plot([p_low, p_low], [0, 0.2], '--', color=COLORS[-1], linewidth = LINEWIDTH)
    
    plt.plot([p_med, p_med], [0, 0.5], '--', color=COLORS[-2], linewidth = LINEWIDTH)

    plt.plot([p_high, p_high], [0, 0.8], '--', color=COLORS[1], linewidth = LINEWIDTH)

    # dots at the ends
    plt.plot(p_low, 0.2, 'o', color=COLORS[-1], linewidth = LINEWIDTH)
    plt.plot(p_med, 0.5, 'o', color=COLORS[-2], linewidth = LINEWIDTH)
    plt.plot(p_high, 0.8, 'o', color=COLORS[1], linewidth = LINEWIDTH)

    # tick p_low, p_med, p_high
    #plt.xticks(list(np.arange(0, 1.1, 0.2)) + [p_low, p_med, p_high])
    plt.tick_params(axis='both',labelsize=FONT_SIZE)

    # include p_low, p_med, p_high on the xticks
    plt.xticks([0.0, p_low, p_med, p_high, 1])
    # plt.xticks(list(np.arange(0, 1.1, 0.2)))

    plt.yticks(list(np.arange(0, 1.1, 0.1)))
    # plt.yticks([0.0, LOW, MED, HIGH, 1])

    plt.xlabel(r'activation probability, $\alpha$', size=FONT_SIZE, labelpad=15)

    # latex <|T|>/n
    plt.ylabel(r'spreadability, $\frac{\langle |T| \rangle}{n}$', rotation=90, size=FONT_SIZE, labelpad=15)

    # legend in bottom right, a few pixels up
    plt.legend(fontsize=FONT_SIZE, loc='lower right', bbox_to_anchor=(1, 0))

    plt.tight_layout()

    # save figure at path
    plt.savefig(f'{PATH}/previews/fig_spreadability.png', bbox_inches='tight')
    # save as pdf
    plt.savefig(f'{PATH}/fig_spreadability.pdf', bbox_inches='tight')

def run_ml_1():
    net_features_dict = {}
    algo_map = {}
    p_tag = 'med'
    k = 10

    # load npz file
    with np.load('./cache/features.npz') as data:
        net_features_dict = dict(data.items())

    # read in corpus dataframe
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # get the list of all files in ./cache/evaluations
    files = os.listdir('./cache/evaluations')
    file_p_tags = []
    file_net_names = []
    best_performances = []
    best_algos = []

    # parse out the p value tags and names from filenames
    for f in files:
        file_net_names.append(f.split('_')[0])
        file_p_tags.append(f.split('_')[1])

    # get the indices of the files with the p tag
    indices = [i for i, x in enumerate(file_p_tags) if x == p_tag]

    skips = []

    for i, row in df.iterrows():
        ready = False

        file_index = -1

        # get the hashed name of the network
        hashed_network_name = row['hashed_network_name']

        # is this network ready to go?
        for j in indices:
            if hashed_network_name in file_net_names[j]:
                ready = True
                file_index = j

        if not ready:
            skips.append(i)
            continue
        else:
            # read in the file
            with open(os.path.join("./cache/evaluations/", files[file_index]), 'rb') as f:
                # load file as a dictionary
                d = np.load(f, allow_pickle=True)

                # convert to a regular dictionary
                d = d.item()

                algo_performance = {}
                best_performance = 0
                best_algo = 'random'

                # parts 1 and 2: compute the best possible performance and the best algorithm
                for algo in algo_dict.keys():
                    if algo in ['myopic', 'naive_myopic']:
                        # discard myopic
                        continue

                    evals = d[algo]

                    for e in evals:
                        vals = e[:k]

                        # compute line of best fit
                        a, _,_ = fit_line(np.array(range(0, len(vals))), np.array(vals), intercept=vals[0])

                        # store the slope
                        if algo not in algo_performance.keys():
                            algo_performance[algo] = []

                        algo_performance[algo].append(a[0])

                    algo_performance[algo] = np.mean(algo_performance[algo])

                    if algo_performance[algo] > best_performance:
                        best_performance = algo_performance[algo]
                        best_algo = algo
                
                best_performances.append(best_performance)
                best_algos.append(best_algo)

    # remove skipped rows from net_features_dict
    for key in net_features_dict.keys():
        net_features_dict[key] = [i for j, i in enumerate(net_features_dict[key]) if j not in skips]

    # create a map for all algos to integers
    for i,key in enumerate(algo_dict.keys()):
        algo_map[key] = i

    # # map the algos in y to integers
    y_classification = [algo_map[i] for i in best_algos]

    # construct a dataframe
    df = pd.DataFrame(net_features_dict)

    # # drop name column
    df = df.drop(columns=['name'])

    # encode domain by mapping to integers
    df['domain'] = df['domain'].map({'Social': 0, 'Economic': 1, 'Biological': 2, 'Informational': 3, 'Technological': 4, 'Transportation': 5})

    # drop eccentricity
    df = df.drop(columns=['number_edges', 'mean_10_highest_deg_nodes_eccentricity', 'highest_deg_node_eccentricity'])

    # classification 
    X_train, X_test, y_train, y_test_classification = train_test_split(df, y_classification, test_size=0.3, random_state=42)

    print('starting random search for hyperparameters')
    rf = RandomForestClassifier(max_features=1, criterion='gini', min_samples_leaf=1, min_samples_split=8, max_depth=10, n_estimators=70)

    model_random = rf.fit(X_train,y_train)

    print('Best hyperparameters are: '+str(model_random.best_params_))
    print('Best score is: '+str(model_random.best_score_))
    # predict
    y_pred_classification = model_random.predict(X_test)
    print('Accuracy:\t\t', accuracy_score(y_test_classification, y_pred_classification))


    clf_classification = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_classification.fit(X_train, y_train)
    y_pred_classification = clf_classification.predict(X_test)

    # classification
    print('Accuracy:\t\t', accuracy_score(y_test_classification, y_pred_classification))

    # get confusion matrix
    cm = confusion_matrix(y_test_classification, y_pred_classification)

    # labels
    labels = [list(algo_dict.keys())[i] for i in sorted(set(list(y_pred_classification) + y_test_classification))]

    # relabel with ALGO_RENAME
    labels = [ALGO_RENAME[i] for i in labels]

    # plot as a heatmap and show numerical values
    plt.figure(figsize=(6, 6))

    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"fontsize":FONT_SIZE}, square=True, cbar=False)

    # Draw horizontal and vertical lines between cells
    for i in range(len(labels) + 1):
        plt.hlines(i, xmin=0, xmax=len(labels), colors='gray', linewidth=0.5,clip_on=False)
        plt.vlines(i, ymin=0, ymax=len(labels), colors='gray', linewidth=0.5,clip_on=False)

    # rotate x-ticks
    plt.xticks(rotation=45, ha='right',rotation_mode='anchor')
    plt.yticks(rotation=45, ha='right',rotation_mode='anchor')
    plt.xlabel('Predicted', fontsize=FONT_SIZE)
    plt.ylabel('True', fontsize=FONT_SIZE)

    # font sizes
    plt.tick_params(axis='both',labelsize=FONT_SIZE)

    # right layout and save
    plt.tight_layout()
    
    # preview
    plt.savefig(f'{PATH}/previews/fig_ml_classification.png', bbox_inches='tight')

    # save as eps
    plt.savefig(f'{PATH}/fig_ml_classification.eps', bbox_inches='tight')

    plt.clf()

    y_regression = best_performances

    # plot distribution of y
    plt.figure(figsize=(6, 3))
    sns.histplot(y_regression, bins=20, kde=True, color=COLORS[-1])
    plt.xlabel(r'best performance, $\beta$', fontsize=FONT_SIZE)
    plt.ylabel('frequency', fontsize=FONT_SIZE)
    plt.tick_params(axis='both',labelsize=FONT_SIZE)
    plt.tight_layout()

    # y-ticks
    plt.yticks(np.arange(0, 31, 5))

    # save as eps
    plt.savefig(f'{PATH}/fig_ml_regression_y_dist.eps', bbox_inches='tight')

    print('distribution fig saved')

    # turn off manually if not needed
    if True:
        # train 100 models for regression
        MSEs = []
        MAEs = []
        feature_importances = {}

        # array for each feature
        for i in range(len(df.columns)):
            feature_importances[df.columns[i]] = []

        for i in range(100):
            X_train, X_test, y_train, y_test_regression = train_test_split(df, y_regression, test_size=0.3, random_state=i)
            model_regression = RandomForestRegressor(n_estimators=100, random_state=i)
            model_regression.fit(X_train, y_train)
            y_pred_regression = model_regression.predict(X_test)

            # compute MSE
            MSE = np.mean((y_pred_regression - y_test_regression)**2)
            MAE = np.mean(np.abs(y_pred_regression - y_test_regression))

            # get feature importances
            fi = {}
            for j in range(len(df.columns)):
                fi[df.columns[j]] = model_regression.feature_importances_[j]

            # append to list
            MSEs.append(MSE)
            MAEs.append(MAE)
            
            for key in fi.keys():
                feature_importances[key].append(fi[key])


        # averages
        print('Average MSE:\t\t', np.mean(MSEs))
        print('Average MAE:\t\t', np.mean(MAEs))

        df_fi = pd.DataFrame(feature_importances)

        # plot
        plt.figure(figsize=(6, 4.4))
        sns.boxplot(data=df_fi, orient='v', palette='colorblind')
        plt.ylabel('mean decr. in impurity', fontsize=FONT_SIZE*1.2)
        plt.xlabel('feature', fontsize=FONT_SIZE*1.2)
        plt.xticks(rotation=25, ha='right',rotation_mode='anchor')

        # y-ticks, two decimal places
        plt.yticks(np.arange(0, 0.56, 0.05))

        plt.tick_params(axis='both',labelsize=FONT_SIZE*1.2)
        plt.tight_layout()

        # save
        plt.savefig(f'{PATH}/previews/fig_ml_importances.png', bbox_inches='tight')

        # save as pdf
        plt.savefig(f'{PATH}/fig_ml_importances.pdf', bbox_inches='tight')

        accuracies = []
        feature_importances = {}
        # array for each feature
        for i in range(len(df.columns)):
            feature_importances[df.columns[i]] = []

        for i in range(100):
            X_train, X_test, y_train, y_test_classification = train_test_split(df, y_classification, test_size=0.3, random_state=i)
            clf_classification = RandomForestClassifier(n_estimators=100, random_state=i)
            clf_classification.fit(X_train, y_train)
            y_pred_classification = clf_classification.predict(X_test)

            # compute accuracy
            accuracy = accuracy_score(y_test_classification, y_pred_classification)
            accuracies.append(accuracy)


            # get feature importances
            fi = {}
            for j in range(len(df.columns)):
                fi[df.columns[j]] = clf_classification.feature_importances_[j]

            # append to list
            MSEs.append(MSE)
            MAEs.append(MAE)
            
            for key in fi.keys():
                feature_importances[key].append(fi[key])

        print('Average accuracy:\t', np.mean(accuracies))
        
        # for feature importances' we'll do a box and whiskers plot
        # create a dataframe
        df_fi = pd.DataFrame(feature_importances)

        # plot
        plt.figure(figsize=(6, 4.4))
        sns.boxplot(data=df_fi, orient='v', palette='colorblind')
        plt.ylabel('mean decr. in impurity', fontsize=FONT_SIZE*1.2)
        plt.xlabel('feature', fontsize=FONT_SIZE*1.2)
        plt.xticks(rotation=25, ha='right',rotation_mode='anchor')

        # y-ticks, two decimal places
        plt.yticks(np.arange(0, 0.20, 0.02))

        plt.tick_params(axis='both',labelsize=FONT_SIZE*1.2)
        plt.tight_layout()

        # save preview
        plt.savefig(f'{PATH}/previews/fig_ml_importances_classifier.png', bbox_inches='tight')

        # save as pdf
        plt.savefig(f'{PATH}/fig_ml_importances_classifier.pdf', bbox_inches='tight')



def run_corpus_stats():
    # load the corpus
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # drop last row
    df = df.iloc[:-1]

    # get unique domains
    domains = df['networkDomain'].unique()

    corpus_mean_nodes = df['number_nodes'].mean()
    corpus_mean_edges = df['number_edges'].mean()
    corpus_degrees = []
    corpus_Cs = []
    corpus_diameters = []
    corpus_variances = []

    # corpus number of rows
    n = len(df)
    print(n)

    for domain in domains:
        # get all network indices in the domain
        indices = df[df['networkDomain'] == domain].index


        # number of networks in the domain
        n = len(indices)

        # mean number of nodes
        mean_nodes = df.iloc[indices]['number_nodes'].mean()

        # mean number of edges
        mean_edges = df.iloc[indices]['number_edges'].mean()

        degrees = []
        Cs = []
        diameters = []
        variances = []

        for i in indices:
            G = networks.get_corpus_graph(i)

            degrees.append(np.mean(list(dict(G.degree()).values())))
            Cs.append(nx.transitivity(G))
            diameters.append(nx.diameter(G))
            variances.append(np.var(list(dict(G.degree()).values())))

        corpus_degrees.extend(degrees)
        corpus_Cs.extend(Cs)
        corpus_diameters.extend(diameters)
        corpus_variances.extend(variances)

        # mean degree
        mean_degree = np.mean(degrees)

        # mean clustering coefficient
        mean_C = np.mean(Cs)

        # mean diameter
        mean_diameter = np.mean(diameters)

        # mean variance of degree
        mean_variance = np.mean(variances)

        print(f'Domain: {domain}')
        print(f'Number of networks: {n}')
        print(f'Mean number of nodes: {mean_nodes}')
        print(f'Mean number of edges: {mean_edges}')
        print(f'Mean degree: {mean_degree}')
        print(f'Mean clustering coefficient: {mean_C}')
        print(f'Mean diameter: {mean_diameter}')
        print(f'Mean variance of degree: {mean_variance}')
        print('\n')


    print(corpus_degrees)

    # corpus
    print(f'Corpus mean number of nodes: {corpus_mean_nodes}')
    print(f'Corpus mean number of edges: {corpus_mean_edges}')
    print(f'Corpus mean degree: {np.mean(corpus_degrees)}')
    print(f'Corpus mean clustering coefficient: {np.mean(corpus_Cs)}')
    print(f'Corpus mean diameter: {np.mean(corpus_diameters)}')
    print(f'Corpus mean variance of degree: {np.mean(corpus_variances)}')



def run_fig_avg_deg_vs_best():
    # values for this follow the order from 3b since that's where I saw a convenient way to precompute this

    avg_degrees = [3.83, 4.72, 4.77, 4.74, 5.11, 4.85, 4.86, 4.86, 4.85, 92.88, 4.52, 3.53, 4.57, 4.67, 4.83, 4.7, 5.07, 5.21, 5.17, 4.85, 4.67, 5.22, 2.99, 4.88, 3.09, 4.88, 4.72, 5.31, 2.96, 2.71, 4.21, 3.06, 3.2, 3.58, 6.79, 7.17, 7.07, 6.89, 43.7, 6.51, 6.48, 6.46, 6.43, 6.49, 16.05, 38.04, 6.83, 6.2, 6.38, 6.27, 6.22, 6.83, 6.29, 6.43, 6.6, 6.24, 6.42, 24.46, 39.11, 9.62, 19.43, 82.42, 5.8, 43.69, 73.69, 81.39, 75.01, 83.38, 85.72, 48.84, 81.05, 61.58, 63.91, 65.41, 7.48, 43.69, 6.46, 5.74, 21.62, 11.97, 8.26, 4.6, 5.85, 70.85, 47.14, 18.85, 16.97, 32.32, 34.72, 16.53, 54.83, 73.41, 55.01, 56.61, 69.37, 68.66, 64.48, 61.93, 62.42, 62.97, 64.03, 57.63, 64.8, 69.2, 60.05, 71.46, 69.1, 68.34, 68.19, 65.45, 58.46, 50.37, 20.54, 67.2, 20.48, 31.23, 43.98, 112.55, 102.66, 69.63, 79.9, 4.52, 4.61, 4.59, 4.55, 4.52, 4.55, 4.56, 4.56, 3.01, 4.57, 4.55, 4.57, 4.59, 4.56, 3.77, 3.53, 4.13, 4.08, 3.64, 3.68, 3.52, 3.02, 3.71, 3.86, 8.55, 3.32, 3.9, 2.23, 2.55, 4.08, 4.08, 11.92, 3.38, 2.0, 3.87, 2.92, 3.31, 3.19, 2.51, 3.07, 5.43, 3.9, 2.13, 3.93, 4.63, 2.89, 2.68, 2.88, 7.32, 9.6, 6.01, 2.7]

    sizes = [537, 546, 598, 600, 601, 609, 610, 610, 619, 677, 683, 705, 725, 759, 769, 774, 783, 808, 813, 816, 861, 880, 890, 896, 964, 965, 974, 1042, 1084, 1108, 1213, 1647, 2214, 3155, 542, 592, 646, 682, 762, 776, 777, 786, 789, 794, 800, 803, 808, 809, 812, 814, 822, 825, 837, 840, 841, 842, 848, 849, 962, 1133, 1294, 1446, 1463, 1510, 1657, 2235, 2250, 2312, 2613, 2672, 2788, 2920, 2970, 2970, 3775, 4039, 4158, 8638, 571, 608, 679, 774, 866, 1076, 1105, 1137, 1552, 1976, 1993, 2025, 2113, 2114, 2114, 2114, 2114, 2114, 2115, 2115, 2115, 2115, 2115, 2115, 2115, 2115, 2115, 2115, 2115, 2115, 2115, 2115, 2115, 2351, 2454, 2461, 2601, 2733, 2917, 3180, 5620, 6547, 12164, 504, 512, 522, 527, 530, 538, 541, 544, 545, 545, 549, 553, 557, 561, 567, 588, 686, 692, 737, 770, 771, 819, 971, 998, 1031, 1035, 1036, 1786, 1893, 2067, 2132, 500, 519, 823, 930, 948, 974, 1009, 1039, 1040, 1087, 1103, 1168, 1226, 1554, 1603, 2266, 3353, 994, 1024, 1350, 2879]

    diameters = [12, 8, 9, 9, 8, 9, 10, 9, 8, 3, 13, 25, 10, 9, 9, 10, 11, 11, 8, 11, 14, 10, 16, 11, 14, 11, 13, 9, 14, 16, 19, 14, 14, 17, 16, 18, 18, 19, 6, 15, 17, 14, 15, 13, 7, 5, 14, 13, 14, 15, 13, 14, 14, 13, 15, 13, 17, 8, 6, 8, 7, 6, 6, 7, 6, 7, 6, 6, 6, 7, 6, 7, 8, 7, 10, 8, 17, 18, 9, 10, 8, 27, 21, 7, 7, 10, 7, 7, 8, 11, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 9, 11, 6, 11, 9, 8, 12, 10, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 17, 8, 8, 8, 8, 8, 10, 11, 13, 13, 12, 11, 11, 18, 17, 16, 9, 12, 16, 69, 54, 10, 9, 7, 16, 9, 27, 36, 46, 27, 62, 37, 11, 16, 62, 17, 29, 54, 98, 54, 10, 6, 12, 4]

    print(len(diameters))

    best_algos_low = ['myopic', 'myopic', 'ppr_myopic', 'gonzales', 'myopic', 'ppr_myopic', 'ppr_myopic', 'ppr_myopic', 'gonzales', 'furthest_non_seed_1', 'myopic', 'myopic', 'myopic', 'gonzales', 'bfs_myopic', 'myopic', 'myopic', 'myopic', 'gonzales', 'myopic', 'myopic', 'gonzales', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'furthest_non_seed_1', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'degree_highest_degree_neighbor_1', 'myopic', 'myopic', 'bfs_myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'myopic', 'degree_lowest_centrality_1', 'inconclusive', 'inconclusive', 'inconclusive', 'inconclusive', 'inconclusive', 'degree_lowest_centrality_1', 'inconclusive', 'inconclusive', 'inconclusive', 'inconclusive', 'bfs_myopic', 'degree_highest_degree_neighbor_1', 'myopic', 'myopic', 'myopic', 'myopic', 'furthest_non_seed_1', 'myopic', 'myopic', 'inconclusive', 'inconclusive', 'bfs_myopic', 'bfs_myopic', 'ppr_myopic', 'myopic', 'gonzales', 'furthest_non_seed_1', 'degree_lowest_centrality_1', 'furthest_non_seed_1', 'furthest_non_seed_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_highest_degree_neighbor_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_highest_degree_neighbor_1', 'degree_lowest_centrality_1', 'myopic', 'degree_lowest_centrality_1', 'degree_highest_degree_neighbor_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_highest_degree_neighbor_1', 'degree_lowest_centrality_1', 'degree_highest_degree_neighbor_1', 'degree_lowest_centrality_1', 'myopic', 'inconclusive', 'naive_myopic', 'inconclusive', 'inconclusive', 'inconclusive', 'inconclusive', 'inconclusive', 'inconclusive', 'inconclusive', 'inconclusive', 'myopic', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'random', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'myopic', 'random', 'random', 'myopic', 'degree_lowest_centrality_1', 'myopic', 'degree_lowest_centrality_1', 'myopic', 'bfs_myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'bfs_myopic', 'myopic', 'degree_highest_degree_neighbor_1', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'myopic', 'myopic', 'random']
    
    best_algos_med = ['myopic', 'myopic', 'gonzales', 'gonzales', 'myopic', 'ppr_myopic', 'ppr_myopic', 'ppr_myopic', 'ppr_myopic', 'myopic', 'myopic', 'myopic', 'naive_myopic', 'gonzales', 'gonzales', 'gonzales', 'gonzales', 'gonzales', 'myopic', 'myopic', 'myopic', 'gonzales', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'bfs_myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'naive_myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'furthest_non_seed_1', 'myopic', 'myopic', 'degree_highest_degree_neighbor_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'myopic', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'myopic', 'degree_lowest_centrality_1', 'myopic', 'degree_highest_degree_neighbor_1', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'furthest_non_seed_1', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_highest_degree_neighbor_0', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_0', 'myopic', 'myopic', 'degree_highest_degree_neighbor_0', 'degree_lowest_centrality_0', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'bfs_myopic', 'bfs_myopic', 'degree_lowest_centrality_1', 'myopic', 'bfs_myopic', 'myopic', 'myopic', 'gonzales', 'myopic', 'gonzales', 'myopic', 'myopic', 'gonzales', 'myopic', 'gonzales', 'myopic', 'gonzales', 'myopic', 'gonzales', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'furthest_non_seed_0', 'myopic', 'myopic', 'bfs_myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'ppr_myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'bfs_myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'myopic', 'myopic', 'random']

    best_algos_high = ['myopic', 'myopic', 'ppr_myopic', 'ppr_myopic', 'myopic', 'ppr_myopic', 'ppr_myopic', 'ppr_myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'naive_myopic', 'random', 'myopic', 'gonzales', 'naive_myopic', 'furthest_non_seed_0', 'naive_myopic', 'naive_myopic', 'furthest_non_seed_1', 'naive_myopic', 'myopic', 'naive_ppr_myopic', 'ppr_myopic', 'naive_bfs_myopic', 'myopic', 'naive_myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'bfs_myopic', 'bfs_myopic', 'furthest_non_seed_0', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_highest_degree_neighbor_1', 'degree_highest_degree_neighbor_1', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'naive_myopic', 'myopic', 'naive_myopic', 'myopic', 'degree_highest_degree_neighbor_1', 'naive_myopic', 'myopic', 'degree_highest_degree_neighbor_1', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'furthest_non_seed_0', 'myopic', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'myopic', 'degree_lowest_centrality_1', 'bfs_myopic', 'degree_highest_degree_neighbor_1', 'myopic', 'myopic', 'gonzales', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'degree_lowest_centrality_1', 'gonzales', 'myopic', 'myopic', 'myopic', 'gonzales', 'degree_highest_degree_neighbor_0', 'naive_myopic', 'myopic', 'myopic', 'myopic', 'degree_highest_degree_neighbor_0', 'naive_myopic', 'naive_myopic', 'degree_highest_degree_neighbor_0', 'degree_lowest_centrality_0', 'degree_highest_degree_neighbor_0', 'degree_highest_degree_neighbor_0', 'degree_highest_degree_neighbor_0', 'myopic', 'myopic', 'naive_myopic', 'myopic', 'degree_lowest_centrality_0', 'degree_highest_degree_neighbor_0', 'naive_myopic', 'degree_lowest_centrality_0', 'myopic', 'bfs_myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_1', 'myopic', 'myopic', 'gonzales', 'myopic', 'myopic', 'myopic', 'myopic', 'naive_myopic', 'myopic', 'myopic', 'bfs_myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'degree_lowest_centrality_0', 'furthest_non_seed_1', 'myopic', 'myopic', 'bfs_myopic', 'naive_myopic', 'myopic', 'degree_lowest_centrality_0', 'ppr_myopic', 'myopic', 'naive_myopic', 'naive_bfs_myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'gonzales', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'myopic', 'ppr_myopic', 'myopic', 'myopic', 'random']
    
    best_dict = {'low': best_algos_low, 'med': best_algos_med, 'high': best_algos_high}

    for spreadability in best_dict.keys():
        best_algos = best_dict[spreadability]
        # create dataframe
        df = pd.DataFrame({'avg_degree': avg_degrees, 'best_algos': best_algos})
        #df = pd.DataFrame({'avg_degree': diameters, 'best_algos': best_algos_low})
    
        # cast first column as floats
        df['avg_degree'] = df['avg_degree'].astype(float)

        # sort by avg_degree
        df = df.sort_values(by='avg_degree')

        # map best algos to integers
        algo_map = {}
        for i,key in enumerate(algo_dict.keys()):
            algo_map[key] = i
        algo_map['inconclusive'] = len(algo_dict.keys())

        # map the algos in y to integers
        y = [algo_map[i] for i in df['best_algos']]

        # how many times is each algo the best?
        best_counts = {}
        for algo in list(algo_dict.keys()) + ['inconclusive']:
            best_counts[algo] = len([i for i in df['best_algos'] if i == algo])

        # plot
        plt.figure(figsize=(6, 4))

        for i in range(len(df)):
            if df['best_algos'].iloc[i] == 'myopic':
                plt.scatter(df['avg_degree'].iloc[i], y[i], color='none', edgecolor='#808080', s=150, alpha=0.7)
            elif df['best_algos'].iloc[i] == 'inconclusive':
                plt.scatter(df['avg_degree'].iloc[i], y[i], color='none', edgecolor='black', s=150, alpha=0.7)
            else:
                plt.scatter(df['avg_degree'].iloc[i], y[i], color='none', edgecolor=COLORS[-1], s=150, alpha=0.7)

        # x and y labels
        plt.xlabel(r'average degree, $\langle k \rangle$', fontsize=FONT_SIZE)
        plt.ylabel('best algorithm', fontsize=FONT_SIZE, labelpad=15)

        # x and y ticks
        plt.xticks(fontsize=FONT_SIZE)

        # because we also rename for the paper, remap from internal algo names to paper names
        labels = [ALGO_RENAME[i] for i in algo_map.keys()]

        # if there are no inconclusives, remove that label
        if best_counts['inconclusive'] == 0:
            labels = labels[:-1]

        # yticks
        plt.yticks(list(range(len(labels))), labels, fontsize=FONT_SIZE)

        # on the right, add the counts
        for i, algo in enumerate(list(algo_dict.keys())+['inconclusive']):
            if best_counts[algo] > 0:
                plt.text(122, i-0.3, f'{best_counts[algo]}', fontsize=FONT_SIZE)

        # save
        plt.tight_layout()

        print(spreadability)
        
        plt.savefig(f'{PATH}/previews/fig_best_vs_avg_degree_{spreadability}.png', bbox_inches='tight')
        # save as eps
        plt.savefig(f'{PATH}/fig_best_vs_avg_degree_{spreadability}.pdf', bbox_inches='tight')


def run_probest_timing():
    iterations_probest = 1000
    p_tags = ['low','med','high']
    #p_tags = ['med']

    # plot
    plt.figure(figsize=(6, 3))

    for p_tag in p_tags:
        times = {}

        # load the cache
        with np.load(f'./cache/timing_probest/times_{p_tag}_{iterations_probest}.npz') as data:
                times = dict(data.items())
                print('Loaded times from cache.')

        n = times['inline_n']
        m = times['inline_m']
        times = times['inline_times']

        # stack n (or m), times
        stacked_n = np.column_stack((n, times))
        stacked_m = np.column_stack((m, times))

        # sort by n
        stacked_n = stacked_n[stacked_n[:,0].argsort()]

        # sort by m
        stacked_m = stacked_m[stacked_m[:,0].argsort()]

        # from stacked_n, drop everything with n below 1000
        stacked_n = stacked_n[stacked_n[:,0] > 1500]
        print(stacked_n)


        # plot n
        plt.plot(stacked_n[:,0], stacked_n[:,1], label=f'{p_tag} spr.')

        # there is a power law relationship between n and runtime
        # and it seems like polynomial
        # we can fit a polynomial to this data
        # so we'll try different degrees
        degrees = [1, 2, 3]

        for degree in degrees:
            # fit a polynomial
            z = np.polyfit(stacked_n[:,0], stacked_n[:,1], degree)

            # get the polynomial
            p = np.poly1d(z)

            # print the polynomial
            print(f'Degree {degree} polynomial:\n{p}')

        print('\n')

        # log scale
        plt.xscale('log')
        plt.yscale('log')

        # x and y labels
        plt.xlabel(r'number of nodes, $n$', fontsize=FONT_SIZE)
        plt.ylabel('runtime (s)', fontsize=FONT_SIZE)

        # tight layout
        plt.tight_layout()

        #legend
        plt.legend()

    # save
    plt.savefig(f'./previews/probest_all_{iterations_probest}', bbox_inches='tight')

def run_algos_timing(alt=False):
    algo_label_y_pos = {
        'myopic': 0.3e+3,
        'naive_myopic': 1.2e+2,        
        'degree_lowest_centrality_0': 0.7e+2,
        'degree_lowest_centrality_1': 0.4e+2,
        'ppr_myopic': 2.3e+1,
        'furthest_non_seed_0': 1.4e+1,
        'furthest_non_seed_1': 0.9e+1,
        'gonzales': 0.5e+1,
        'naive_ppr_myopic': 1e+0,
        'degree_highest_degree_neighbor_0': 3e-3,
        'degree_highest_degree_neighbor_1': 5e-3,
        'bfs_myopic': 2.5e+0,
        'naive_bfs_myopic': 3e-1,
        'random': 0.2e-3,
    }

    p_tag = 'med'
    ### PART 1 ###
    # retrieve networks, get n, m
    # I forgot to cache these earlier
    # also, load precompute costs
    
    # load dataframe
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # get all files in cached evals
    files = os.listdir('./cache/evaluations/')

    precompute_costs = {}
    with np.load('./cache/times_apsp.npz') as data:
        precompute_costs = dict(data.items())

    precompute_costs_used = []
    n = []
    m = []

    for i in range(175):
        # hashed net name
        net_hash = df.iloc[i]['hashed_network_name']

        # get files starting with net_hash
        files_ = [file for file in files if file.startswith(net_hash)]
        
        found = False

        for file in files_:
            # separate string
            parts = file.split('_')

            if parts[1] == p_tag:
                found = True
                break

        if found:
            n.append(df.iloc[i]['number_nodes'])
            m.append(df.iloc[i]['number_edges'])
            precompute_costs_used.append(precompute_costs[net_hash])

    ### PART 2 ###
    # get files in the cache
    files = os.listdir('./cache/timing_algos/')

    # drop nested folders
    files = [file for file in files if file not in ['old', 'low', 'med', 'high']]

    # sort files
    files_with_index = [(int(file.split('_')[2][:-4]), file) for file in files] # make tuples where the first element is the index
    files_with_index.sort() # sorts by the first element of the tuple

    files = [file[1] for file in files_with_index] # take out sorted files

    times = {}

    for file in files:
        # load first file
        with np.load(f'./cache/timing_algos/{file}') as data:
            times_ = dict(data.items())

            # get the keys
            keys = list(times_.keys())

            for key in keys:
                if key not in times:
                    times[key] = times_[key]
                else:
                    # grow the array
                    times[key] = np.append(times[key], times_[key])

    # truncate n, m, precompute to only have whatever has already been processed
    n = n[:len(times['random'])]
    m = m[:len(times['random'])]
    precompute_costs_used = precompute_costs_used[:len(times['random'])]

    print(times.keys())
    # add precompute cost for algos that need it
    # gonzalez, furthestnonnseed, furthestnonseedchooseneighbor, degreelowestcentrality, degreelowestcentralitychooseneighbor
    for key in ['gonzales', 'furthest_non_seed_0', 'furthest_non_seed_1', 'degree_lowest_centrality_0', 'degree_lowest_centrality_1']:
       times[key] += precompute_costs_used

    if alt:
        plt.figure(figsize=(6, 3))
    else:
        plt.figure(figsize=(6, 4))

    # Customizing the y-axis tick labels to milliseconds
    y_ticks = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2])  # y-ticks in seconds
    y_labels = []

    # Adding horizontal gridlines at the specified y-ticks
    for y in y_ticks:
        plt.axhline(y=y, color=[0.8, 0.8, 0.8, 1.0], linewidth=1)

    yvals_myopic = []
    yvals_mindegree_hcn = []

    yvals_fitted_last = {}

    for i,key in enumerate(times.keys()):
        # overried for the smaller figure
        if alt and key not in ['myopic', 'naive_myopic', 'gonzales', 'random']:
            continue

        y = times[key]

        # for estimating distributions later
        if key == 'myopic':
            yvals_myopic = y
        if key == 'degree_lowest_centrality_1':
            yvals_mindegree_hcn = y

        # stack
        stacked_n = np.column_stack((n, y))

        # scatter plot with shapes for datapoints
        plt.scatter(stacked_n[:,0], stacked_n[:,1], label=key, alpha=0.3)

        # fit a line against logged data
        z = np.polyfit(np.log(stacked_n[:,0]), np.log(stacked_n[:,1]), 1)
        p = np.poly1d(z)
        y_fit = np.exp(p(np.log(stacked_n[:,0])))

        # plot fitted line
        line = plt.plot(stacked_n[:,0], y_fit, linewidth = 5)

        name = ALGO_RENAME[key]
        
        if name == 'Myopic':
            name = 'Myopic*'

        if name == 'Naive Myopic':
            name = 'Naive Myopic*'

        yvals_fitted_last[key] = y_fit[-1]

        plt.text(10000, algo_label_y_pos[key], name, fontsize=FONT_SIZE, color = line[0].get_color())

    plt.xlabel(r'number of nodes, $n$', fontsize=FONT_SIZE)
    plt.ylabel('runtime (milliseconds)', fontsize=FONT_SIZE)

    # custom y-tick labels
    for tick in y_ticks:
        if tick >= 1e-3:
            y_labels.append(f'{int(tick * 1000)}')
        else:
            y_labels.append(f'{tick * 1000:.1f}')

    # add more x-ticks
    x_ticks = [500, 1000, 2000, 4000, 7500]
    x_ticks.append(80000) # one more tick to fit algo labels

    # x_tick labels
    x_labels = [f'{int(i)}' for i in x_ticks]

    # whitespace the last tick
    x_labels[-1] = ''

    plt.xticks(x_ticks, x_labels, fontsize=FONT_SIZE)

    plt.yticks(y_ticks, y_labels, fontsize=FONT_SIZE)
    plt.grid(False) # disable stock grid

    # Disable minor ticks
    plt.gca().xaxis.set_minor_locator(NullLocator())
    plt.gca().yaxis.set_minor_locator(NullLocator())

    # Suppress the tick mark for the last x-axis tick
    ax = plt.gca()
    ticks = ax.xaxis.get_major_ticks()
    ticks[-1].tick1line.set_visible(False)  # Hide the tick mark on the bottom
    ticks[-1].tick2line.set_visible(False)  # Hide the tick mark on the top

    # tight
    plt.tight_layout()

    # take the last value of the fitted line for every algo
    keys = list(yvals_fitted_last.keys())
    vals = list(yvals_fitted_last.values())

    # sort by value in descending order
    vals, keys = zip(*sorted(zip(vals, keys), reverse=True))

    # print
    for i in range(len(keys)):
        print(f'{ALGO_RENAME[keys[i]]}: {vals[i]}')

    # alternative running mode for the intersection
    if alt is True:
        xvals = n
        data_length = len(xvals)
        
        # stack data
        stacked_myopic = np.column_stack((xvals, yvals_myopic))
        stacked_mindegree_hcn = np.column_stack((xvals, yvals_mindegree_hcn))
        
        bootstraps = 1000

        bs_myopic = []
        bs_mindegree_hcn = []
        As_myopic = []
        As_mindegree_hcn = []
        intersections = []

        for i in range(bootstraps):
            # sample with replacement
            indices = np.random.choice(len(stacked_myopic), data_length) # sample indices because numpy doesn't like sampling tuples

            # sort in ascending order
            indices = np.sort(indices)
            sample_myopic = stacked_myopic[indices]

            indices = np.random.choice(len(stacked_mindegree_hcn), data_length)

            indices = np.sort(indices)

            sample_mindegree_hcn = stacked_mindegree_hcn[indices]

            # fit a line
            z_myopic = np.polyfit(np.log(sample_myopic[:,0]), np.log(sample_myopic[:,1]), 1)
            z_mindegree_hcn = np.polyfit(np.log(sample_mindegree_hcn[:,0]), np.log(sample_mindegree_hcn[:,1]), 1)

            b_myopic = z_myopic[0]
            b_mindegree_hcn = z_mindegree_hcn[0]
            log_A_myopic = z_myopic[1]
            log_A_mindegree_hcn = z_mindegree_hcn[1]
            A_myopic = np.exp(log_A_myopic)
            A_mindegree_hcn = np.exp(log_A_mindegree_hcn)

            #store
            bs_myopic.append(b_myopic)
            bs_mindegree_hcn.append(b_mindegree_hcn)
            As_myopic.append(A_myopic)
            As_mindegree_hcn.append(A_mindegree_hcn)

            # compute intersection
            intersection = (A_mindegree_hcn / A_myopic) ** (1 / (b_myopic - b_mindegree_hcn))

            intersections.append(np.ceil(intersection))


        # compute confidence interval for intersection
        data = np.array(intersections)

        print(np.mean(data))
        print(max(data))
        print(min(data))

        confidence_interval = stats.norm.interval(0.95, loc=np.mean(data), scale=np.std(data, ddof=1)/np.sqrt(len(data)))

        print(f'Confidence interval for intersection: {confidence_interval}')

def run_ensemble():
    # tabulate algorithms similarly to 3b
    p_tag = 'low'
    k = 10

    # load dataframe
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # get all files in cached evals
    files = os.listdir('./cache/evaluations/')
    file_p_tags = []
    file_net_names = []

    # parse out the p value tags from filenames
    for f in files:
        file_p_tags.append(f.split('_')[1])

    # get the indices of the files with the p tag
    indices = [i for i, x in enumerate(file_p_tags) if x == p_tag]

    files = [files[i] for i in indices]

    # parse out net names from desired files
    for f in files:
        file_net_names.append(f.split('_')[0])

    # get the indices of the net names in the dataframe
    # will help restore order later
    # since timing files are indexed with this
    indices_df = []

    for net_name in file_net_names:
        indices_df.append(df[df['hashed_network_name'] == net_name].index[0])

    algo_betas = {}

    for file in files:

        # read in the file
        with open(os.path.join("./cache/evaluations/", file), 'rb') as f:
            # load file as a dictionary
            d = np.load(f, allow_pickle=True)

            # convert to a regular dictionary
            d = d.item()

            algo_performance = {}

            for algo in d.keys():
                evals = d[algo]

                for e in evals:
                    vals = e[:k]

                    # compute line of best fit
                    a, yfits, _ = fit_line(np.array(range(0, len(vals))), np.array(vals))
                    
                    # store the slope
                    if algo not in algo_performance.keys():
                        algo_performance[algo] = []

                    algo_performance[algo].append(a[0])

                # average the slopes
                algo_performance[algo] = np.mean(algo_performance[algo])

                # store the performance
                if algo not in algo_betas.keys():
                    algo_betas[algo] = []
                
                algo_betas[algo].append(algo_performance[algo])

    algo_scores = {}


    for algo in algo_betas.keys():
        if algo not in algo_scores.keys():
                algo_scores[algo] = []
        for i in range(len(algo_betas[algo])):
            score = algo_betas[algo][i] / algo_betas['myopic'][i]

            if score > 0.8:
                algo_scores[algo].append(1)
            else:
                algo_scores[algo].append(0)
    
    # drop myopic, naive myopic
    algo_scores.pop('myopic')
    algo_scores.pop('naive_myopic')

    algo_combinations = list(itertools.combinations(algo_scores.keys(), 5))

    best_combination = []
    best_score_sum = 0

    for c in algo_combinations:
        scores = np.zeros(len(algo_scores['random']))

        for i in range(len(algo_scores['random'])):
            for algo in c:
                if algo_scores[algo][i] == 1:
                    scores[i] = 1

    
        score_sum = np.sum(scores)

        if score_sum > best_score_sum:
            best_score_sum = score_sum
            best_combination = c

    best_betas = np.zeros(len(algo_betas['random']))
    best_algos = ['random'] * len(algo_betas['random'])

    for algo in best_combination:
        for i in range(len(algo_betas['random'])):
            if algo_betas[algo][i] > best_betas[i]:
                best_betas[i] = algo_betas[algo][i]
                best_algos[i] = algo

    print(best_combination)

    # stack indices_df, best_betas
    stacked_myopic = np.column_stack((indices_df, algo_betas['myopic']))
    stacked_best = np.column_stack((indices_df, best_betas))

    # sort by indices_df

    stacked_myopic = stacked_myopic[stacked_myopic[:,0].argsort()]
    stacked_best = stacked_best[stacked_best[:,0].argsort()]


    # now retrieve all the execution times from earlier
    files = os.listdir(f'./cache/timing_algos/{p_tag}/')

    times_myopic = np.zeros(len(stacked_myopic))
    times_best = np.zeros(len(stacked_best))

    precompute_costs = {}
    with np.load('./cache/times_apsp.npz') as data:
        precompute_costs = dict(data.items())


    for i, file in enumerate(files):
        hashed_network_name = df.iloc[indices_df[i]]['hashed_network_name']
        precompute_for_this_network = precompute_costs[hashed_network_name]

        if best_algos[i] in ['gonzales', 'furthest_non_seed_0', 'furthest_non_seed_1', 'degree_lowest_centrality_0', 'degree_lowest_centrality_1']:
                times_best[i] += precompute_for_this_network

        # load first file
        with np.load(f'./cache/timing_algos/{p_tag}/{file}') as data:
            times_ = dict(data.items())

            # get the keys
            keys = list(times_.keys())

            for key in keys:
                if key == 'myopic':
                    times_myopic[i] = times_[key]
                if key == best_algos[i]:
                    times_best[i] += times_[key]
            
    # stack
    myopic_beta_time = np.column_stack((stacked_myopic[:,1], times_myopic))

    best_beta_time = np.column_stack((stacked_best[:,1], times_best))


    delta_ratio = best_beta_time[:,0] / myopic_beta_time[:,0]

    time_ratio = myopic_beta_time[:,1] / best_beta_time[:,1]

    delta_percentage = (delta_ratio - 1) * 100

    # stack
    stacked_data = np.column_stack((time_ratio, delta_percentage))

    Q1 = 0
    Q2 = 0
    Q3 = 0
    Q4 = 0

    for i in range(len(stacked_data[:,0])):
        if stacked_data[i,0] > 1 and stacked_data[i,1] > 0:
            Q1 += 1
        elif stacked_data[i,0] < 1 and stacked_data[i,1] > 0:
            Q2 += 1
        elif stacked_data[i,0] < 1 and stacked_data[i,1] < 0:
            Q3 += 1
        elif stacked_data[i,0] > 1 and stacked_data[i,1] < 0:
            Q4 += 1


    # remove outliers
    if p_tag != 'low':
        print(stacked_data)
        stacked_data_no_outliers = stacked_data[stacked_data[:,0] < 401]
        stacked_data_no_outliers = stacked_data_no_outliers[stacked_data_no_outliers[:,1] < 200]

    else:
        # stacked_data_no_outliers = stacked_data[stacked_data[:,0] < 1000]
        # stacked_data_no_outliers = stacked_data_no_outliers[stacked_data_no_outliers[:,1] < 1000]
        stacked_data_no_outliers = stacked_data

    # wrap for legibility
    x = stacked_data_no_outliers[:,0]
    y = stacked_data_no_outliers[:,1]

    print(len(x))
    print(len(y))

    # if infinity or nan, remove from both
    to_delete = []
    for i in range(len(x)):
        if np.isinf(x[i]) or np.isnan(x[i]) or np.isinf(y[i]) or np.isnan(y[i]):
            to_delete.append(i)

    x = np.delete(x, to_delete)
    y = np.delete(y, to_delete)


    # AVERAGES WITH STANDARD DEVIATION
    print('Average speedup:', np.mean(x))
    print('Standard deviation speedup:', np.std(x))

    print('Average ratio:', np.mean(y))
    print('Standard deviation ratio:', np.std(y))

    # plot
    fig = plt.figure(figsize=(6, 3))

    gs = fig.add_gridspec(2,2, width_ratios=(7,1), height_ratios=(2.4,7), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)

    ax_main = fig.add_subplot(gs[1,0])

    ax_hist_x = fig.add_subplot(gs[0,0], sharex=ax_main)
    ax_hist_y = fig.add_subplot(gs[1,1], sharey=ax_main)

    # histograms
    ax_hist_x.tick_params(axis="x", labelbottom=False)
    ax_hist_y.tick_params(axis="y", labelleft=False)
    binwidth=10
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_hist_x.hist(x, bins=bins)
    ax_hist_y.hist(y, bins=bins, orientation='horizontal')

    # scatter
    ax_main.scatter(x, y, alpha=0.5, color=COLORS[-1], s=60)

    # dashed lines
    ax_main.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax_main.axvline(x=1, color='black', linestyle='--', alpha=0.3)

    # quadrant counters
    ax_main.text(25, 90, f'{Q1}', fontsize=FONT_SIZE)
    ax_main.text(-20, 90, f'{Q2}', fontsize=FONT_SIZE)
    ax_main.text(-20, -90, f'{Q3}', fontsize=FONT_SIZE)
    ax_main.text(25, -90, f'{Q4}', fontsize=FONT_SIZE)

    # limits
    ax_main.set_xlim(-25, 375)
    ax_main.set_ylim(-125, 125)

    # set x-ticks
    x_ticks = list(range(0, 375, 50))
    ax_main.set_xticks(x_ticks)
    ax_hist_x.set_xticks(x_ticks)

    # set y-ticks
    y_ticks = list(range(-125, 126, 25))
    ax_main.set_yticks(y_ticks)
    ax_hist_y.set_yticks(y_ticks)

    # tick size
    ax_main.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    ax_hist_x.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    ax_hist_y.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    ax_main.set_xlabel('runtime Myopic / runtime Ensemble', fontsize=FONT_SIZE)

    ax_main.set_ylabel(r'% better vs Myopic $\beta$', fontsize=FONT_SIZE)

    plt.savefig(f'{PATH}/previews/fig_ensemble_runtime_ratio_{p_tag}.png', bbox_inches='tight')
    plt.savefig(f'{PATH}/fig_ensemble_runtime_ratio_{p_tag}.pdf', bbox_inches='tight') 
         


def fig_ensemble_ml():

    # similar to the previous ensemble figure but also use ML
    # to approximate the oracle

    network_ratios = {}
    network_speedups = {}

    for i in range(174):
        network_ratios[i] = []
        network_speedups[i] = []

    net_features_dict = {}
    p_tag = 'med'
    k = 10

    greenlit_indices = []

    # load features
    with np.load('./cache/features.npz') as data:
        net_features_dict = dict(data.items())

    # drop features that do not seem to be useful:
    net_features_dict.pop('mean_10_highest_deg_nodes_eccentricity')
    net_features_dict.pop('highest_deg_node_eccentricity')

    # array for each feature
    feature_importances = {}
    for f in net_features_dict.keys():
        feature_importances[f] = []

    # load dataframe
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # get all files in cached evals
    files = os.listdir('./cache/evaluations/')
    file_p_tags = []
    file_net_names = []

    # parse out the p value tags from filenames
    for f in files:
        file_p_tags.append(f.split('_')[1])

    # get the indices of the files with the p tag
    indices = [i for i, x in enumerate(file_p_tags) if x == p_tag]

    files = [files[i] for i in indices]

    # parse out net names from desired files
    for f in files:
        file_net_names.append(f.split('_')[0])

    # get the indices of the net names in the dataframe
    # will help restore order later
    # since timing files are indexed with this
    file_index_to_df = [] # takes eval file index, returns corpus index
    df_index_to_file = np.zeros(174) # takes corpus index, returns file index

    for i, net_name in enumerate(file_net_names):
        file_index_to_df.append(df[df['hashed_network_name'] == net_name].index[0])
        df_index_to_file[file_index_to_df[-1]] = i

    # apsp times for each network
    precompute_costs = {}
    with np.load('./cache/times_apsp.npz') as data:
        precompute_costs = dict(data.items())



    # retrieve and index runtime files
    runtime_files = os.listdir(f'./cache/timing_algos/{p_tag}/')
    runtime_files_ = [file[:-4] for file in runtime_files] # truncate .npz
    runtime_files_ = [file.split('_') for file in runtime_files_] # split
    runtime_files_ = [int(file[2]) for file in runtime_files_] # get index
    greenlit_indices = runtime_files_ # these are corpus-level indices we want

    # put together a union of indices that are ready
    greenlit_indices = [i for i in greenlit_indices if i in file_index_to_df]

    # get ready to retrieve runtimes
    runtimes = {}
    for i in greenlit_indices:
        runtimes[i] = {}

    # stack index against runtime_files
    runtime_files = list(zip(runtime_files, runtime_files_))

    # sort by second column
    runtime_files = sorted(runtime_files, key=lambda x: int(x[1]))



    for file, i in runtime_files:
        hashed_network_name = df.iloc[i]['hashed_network_name']
        precompute_for_this_network = precompute_costs[hashed_network_name]

        # load first file
        with np.load(f'./cache/timing_algos/{p_tag}/{file}') as data:
            times_ = dict(data.items())

            # get the keys
            keys = list(times_.keys())

            for algo in keys:
                if algo in ['gonzales', 'furthest_non_seed_0', 'furthest_non_seed_1', 'degree_lowest_centrality_0', 'degree_lowest_centrality_1']:
                    runtimes[i][algo] = times_[algo][0] + precompute_for_this_network
                else:
                    runtimes[i][algo] = times_[algo][0]

    # load in evaluations
    algo_betas = {}

    for file_index, file in enumerate(files):

        # read in the file
        with open(os.path.join("./cache/evaluations/", file), 'rb') as f:
            
            # load file as a dictionary
            d = np.load(f, allow_pickle=True)

            df_index = file_index_to_df[file_index]

            # convert to a regular dictionary
            d = d.item()

            algo_performance = {}

            for algo in d.keys():
                evals = d[algo]

                for e in evals:
                    vals = e[:k]

                    # compute line of best fit
                    a, yfits, _ = fit_line(np.array(range(0, len(vals))), np.array(vals))
                    
                    # store the slope
                    if algo not in algo_performance.keys():
                        algo_performance[algo] = []

                    algo_performance[algo].append(a[0])

                # average the slopes
                algo_performance[algo] = np.mean(algo_performance[algo])

                # store the performance
                if algo not in algo_betas.keys():
                    algo_betas[algo] = []
                
                algo_betas[algo].append(algo_performance[algo])

    algo_scores = {}

    # Scoring for metalearner training

    for algo in algo_betas.keys():
        if algo not in algo_scores.keys():
                algo_scores[algo] = []
        for i in range(len(algo_betas[algo])):
            score = algo_betas[algo][i] / algo_betas['myopic'][i]

            if score > 0.8:
                algo_scores[algo].append(1)
            else:
                algo_scores[algo].append(0)
    
    # drop myopic, naive myopic
    algo_scores.pop('myopic')
    algo_scores.pop('naive_myopic')

    delta_percentage_list = []

    for k in range(1000):
        algo_combinations = list(itertools.combinations(algo_scores.keys(), 5))
        best_combination = []
        best_score_sum = 0

        # perform a test-train split
        # train on 80% of the data
        # test on 20% of the data

        # sample indices uniformly
        # these are FILE INDICES
        indices = list(range(len(file_index_to_df)))
        # indices = greenlit_indices

        train_indices = np.random.choice(indices, int(0.8 * len(indices)), replace=False)
        test_indices = [i for i in indices if i not in train_indices]

        train_betas = {}
        test_betas = {}
        train_algo_scores = {}
        test_algo_scores = {}

        for algo in algo_scores.keys():
            train_betas[algo] = []
            test_betas[algo] = []
            train_algo_scores[algo] = []
            test_algo_scores[algo] = []

        for i in indices:
            if i in train_indices:
                for algo in algo_scores.keys():
                    train_betas[algo].append(algo_betas[algo][i])
                    train_algo_scores[algo].append(algo_scores[algo][i])
            else:
                for algo in algo_scores.keys():
                    test_betas[algo].append(algo_betas[algo][i])
                    test_algo_scores[algo].append(algo_scores[algo][i])

        for c in algo_combinations:
            scores = np.zeros(len(train_algo_scores['random']))

            for i in range(len(train_algo_scores['random'])):
                for algo in c:
                    if train_algo_scores[algo][i] == 1:
                        scores[i] = 1

            score_sum = np.sum(scores)

            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_combination = c

        best_betas = np.zeros(len(train_betas['random']))


        for algo in best_combination:
            for i in range(len(train_betas['random'])):
                if train_algo_scores[algo][i] == 1:
                    best_betas[i] = train_betas[algo][i]

        # train ml model to choose an ensemble algorithm
        # idea: predict which of the five algorithms will perform best
        # based on the performance of the algorithms on the training data
        # use the test data to evaluate the model

        train_features = {}
        test_features = {}

        for f in net_features_dict.keys():
            train_features[f] = []
            test_features[f] = []

        for i in indices:
            corpus_index = int(file_index_to_df[i])

            if i in train_indices:
                for f in net_features_dict.keys():
                    train_features[f].append(net_features_dict[f][corpus_index])
            else:
                for f in net_features_dict.keys():
                    test_features[f].append(net_features_dict[f][corpus_index])

        # map ensemble algos to integers
        algo_map = {}

        for i, algo in enumerate(best_combination):
            algo_map[algo] = i

        # populate y = best algorithm that produces highest beta
        y_train = np.zeros(len(train_betas['random']))

        for i in range(len(train_betas['random'])):
            best_algo = np.random.choice(list(algo_map.keys()),1)[0]
            best_beta = train_betas[best_algo][i]

            for algo in best_combination:
                if train_betas[algo][i] > best_beta:
                    best_beta = train_betas[algo][i]
                    best_algo = algo

            y_train[i] = algo_map[best_algo]

        y_test = np.zeros(len(test_betas['random']))

        for i in range(len(test_betas['random'])):
            best_algo = np.random.choice(list(algo_map.keys()),1)[0]
            best_beta = train_betas[best_algo][i]

            for algo in best_combination:
                if test_betas[algo][i] > best_beta:
                    best_beta = test_betas[algo][i]
                    best_algo = algo

            y_test[i] = algo_map[best_algo]

        # wrap features in pd dataframes
        train_features = pd.DataFrame(train_features)
        test_features = pd.DataFrame(test_features)

        # drop name column
        train_features = train_features.drop(columns=['name'])
        test_features = test_features.drop(columns=['name'])

        # encode domain
        domain_dict = {'Economic': 0, 'Social': 1, 'Technological': 2, 'Biological': 3, 'Informational': 4, 'Transportation': 5}
        train_features['domain'] = train_features['domain'].map(domain_dict)
        test_features['domain'] = test_features['domain'].map(domain_dict)  

        X_train = train_features
        X_test = test_features
        
        ###
        # hyperparameter search
        # rs_space={'max_depth':list(np.arange(5, 100, step=5)) + [None],
        #             'n_estimators':np.arange(10, 500, step=10),
        #             'max_features':np.arange(1, len(train_features.columns), step=1),
        #             'criterion':['gini','entropy'],
        #             'min_samples_leaf':np.arange(1, 10, step=1),
        #             'min_samples_split':np.arange(2, 10, step=2)
        #         }
        
        # rf = RandomForestClassifier()

        # rf_random = RandomizedSearchCV(rf, rs_space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=3)
        # model_random = rf_random.fit(X_train,y_train)

        # # random random search results
        # print('Best hyperparameters: '+str(model_random.best_params_))
        # print('Best score: '+str(model_random.best_score_))

        ###


        clf = RandomForestClassifier(max_features=1, criterion='gini', min_samples_leaf=1, min_samples_split=8, max_depth=10, n_estimators=70)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # get feature importances
        for j, name in enumerate(train_features.columns):
            feature_importances[name].append(clf.feature_importances_[j])

        # accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # retrieve the corresponding betas
        best_betas = []
        best_algos = []
        for y in y_pred:
            for algo, i in algo_map.items():
                if i == y:
                    best_algos.append(algo) # list of best algos by original name
        
        for i in range(len(test_indices)):
            best_algo = best_algos[i]
            index = test_indices[i]

            beta = algo_betas[best_algo][index]
            best_betas.append(beta)

        # retrieve corresponding myopic betas for test_indices
        myopic_betas = algo_betas['myopic']
        myopic_betas = np.array(myopic_betas)[test_indices]

        # compute the ratio of best_betas to myopic_betas
        ratios = np.array(best_betas) / np.array(myopic_betas)

        for ratios_index, file_index in enumerate(test_indices):
            df_index = file_index_to_df[file_index]

            # save the ratios
            network_ratios[df_index].append(ratios[ratios_index])

            # save the speedups
            myopic_runtime = runtimes[df_index]['myopic']
            best_runtime = runtimes[df_index][best_algos[ratios_index]]
            speedup = myopic_runtime / best_runtime
            network_speedups[df_index].append(speedup)

        # compute the average ratio
        avg_ratio = np.mean(ratios)

        delta_percentage = (avg_ratio - 1) * 100

        delta_percentage_list.append(delta_percentage)

        # print current percentage and average, to two decimal places

        print(f'Finished iteration {k+1} of 1000')
        print(f'Current percentage: {delta_percentage:.2f}%')
        print(f'Average percentage: {np.mean(delta_percentage_list):.2f}%')

    # make feature importances plot
    import seaborn as sns

    print(feature_importances)

    feature_importances.pop('name')
    df_fi = pd.DataFrame(feature_importances)

    # plot
    plt.figure(figsize=(6, 4.4))
    sns.boxplot(data=df_fi, orient='v', palette='colorblind')
    plt.ylabel('mean decr. in impurity', fontsize=FONT_SIZE*1.1)
    plt.xlabel('feature', fontsize=FONT_SIZE*1.1)
    plt.xticks(rotation=25, ha='right',rotation_mode='anchor')

    # y-ticks, two decimal places
    plt.yticks(np.arange(0, 0.17, 0.02))

    plt.tick_params(axis='both',labelsize=FONT_SIZE*1.1)
    plt.tight_layout()

    # save preview
    plt.savefig(f'{PATH}/previews/fig_ml_importances_metalearner.png', bbox_inches='tight')

    # save as pdf
    plt.savefig(f'{PATH}/fig_ml_importances_metalearner.pdf', bbox_inches='tight')

    average_deltas = []
    average_speedups = []

    for i in range(173):
        average_deltas.append((np.mean(network_ratios[i]) - 1) * 100)
        average_speedups.append(np.mean(network_speedups[i]))

    print(average_deltas)
    print(average_speedups)

    # stack average_speedups, average_deltas
    stacked_data = np.column_stack((average_speedups, average_deltas))

    if p_tag != 'low':
        stacked_data_no_outliers = stacked_data[stacked_data[:,0] < 401]
        stacked_data_no_outliers = stacked_data_no_outliers[stacked_data_no_outliers[:,1] < 200]
    else:
        stacked_data_no_outliers = stacked_data[stacked_data[:,0] < 2000]
        stacked_data_no_outliers = stacked_data_no_outliers[stacked_data_no_outliers[:,1] < 2000]

    # AVERAGE
    print('Average ratio:', np.mean(average_deltas))
    print('Average speedup:', np.mean(average_speedups))

    Q1 = 0
    Q2 = 0
    Q3 = 0
    Q4 = 0

    for i in range(len(stacked_data[:,0])):
        if stacked_data[i,0] > 1 and stacked_data[i,1] > 0:
            Q1 += 1
        elif stacked_data[i,0] < 1 and stacked_data[i,1] > 0:
            Q2 += 1
        elif stacked_data[i,0] < 1 and stacked_data[i,1] < 0:
            Q3 += 1
        elif stacked_data[i,0] > 1 and stacked_data[i,1] < 0:
            Q4 += 1

    # wrap for legibility
    x = stacked_data_no_outliers[:,0]
    y = stacked_data_no_outliers[:,1]

    # AVERAGES WITH STANDARD DEVIATION
    print('Average speedup:', np.mean(x))
    print('Standard deviation speedup:', np.std(x))

    print('Average ratio:', np.mean(y))
    print('Standard deviation ratio:', np.std(y))

# plot
    fig = plt.figure(figsize=(6, 3))

    gs = fig.add_gridspec(2,2, width_ratios=(7,1), height_ratios=(2.4,7), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)

    ax_main = fig.add_subplot(gs[1,0])

    ax_hist_x = fig.add_subplot(gs[0,0], sharex=ax_main)
    ax_hist_y = fig.add_subplot(gs[1,1], sharey=ax_main)

    # histograms
    ax_hist_x.tick_params(axis="x", labelbottom=False)
    ax_hist_y.tick_params(axis="y", labelleft=False)
    binwidth=10
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_hist_x.hist(x, bins=bins)
    ax_hist_y.hist(y, bins=bins, orientation='horizontal')

    # scatter
    ax_main.scatter(x, y, alpha=0.5, color=COLORS[-1], s=60)

    # dashed lines
    ax_main.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax_main.axvline(x=1, color='black', linestyle='--', alpha=0.3)

    # quadrant counters
    ax_main.text(25, 90, f'{Q1}', fontsize=FONT_SIZE)
    ax_main.text(-20, 90, f'{Q2}', fontsize=FONT_SIZE)
    ax_main.text(-20, -90, f'{Q3}', fontsize=FONT_SIZE)
    ax_main.text(25, -90, f'{Q4}', fontsize=FONT_SIZE)

    # limits
    ax_main.set_xlim(-25, 375)
    ax_main.set_ylim(-125, 125)

    # set x-ticks
    x_ticks = list(range(0, 375, 50))
    ax_main.set_xticks(x_ticks)
    ax_hist_x.set_xticks(x_ticks)

    # set y-ticks
    y_ticks = list(range(-125, 126, 25))
    ax_main.set_yticks(y_ticks)
    ax_hist_y.set_yticks(y_ticks)

    # tick size
    ax_main.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    ax_hist_x.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    ax_hist_y.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    # tight layout
    #plt.tight_layout()

    ax_main.set_xlabel('runtime Myopic / runtime Ensemble', fontsize=FONT_SIZE)

    ax_main.set_ylabel(r'% better vs Myopic $\beta$', fontsize=FONT_SIZE)

    plt.savefig(f'./previews/fig_ensemble_runtime_ratio_{p_tag}.png', bbox_inches='tight')
    
    # save as pdf
    plt.savefig(f'./previews/fig_ensemble_runtime_ratio_{p_tag}.pdf', bbox_inches='tight')
