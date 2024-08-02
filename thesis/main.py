import networks as networks
import sys
import runners
import runners_figs

# suppress networkx future warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# suppress deprecated warning
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# # command line arguments
command = sys.argv[1]
args = sys.argv[2:]
algo_dict = {
        "random": False,
        "greedy": False,
        "myopic": False,
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


# python main.py corpus 0 0.5 5 random,myopic,naive_myopic,gonzales,furthest_non_seed_0,furthest_non_seed_1,bfs_myopic,naive_bfs_myopic,ppr_myopic,naive_ppr_myopic,degree_lowest_centrality_0,degree_lowest_centrality_1,degree_highest_degree_neighbor_0,degree_highest_degree_neighbor_1

if command == "algo_assessment":
    # run algorithm assessment
    # args: graph_names, p_vals, k
    graph_names = args[0].split(",")
    p_vals = [float(p) for p in args[1].split(",")]

    # set algos to True
    for algo in args[2].split(","):
        algo_dict[algo] = True

    # get k
    k = int(args[3])

    runners.run_algorithm_assessment(graph_names, p_vals, algo_dict, k)

if command == "bruteforce":
    # run bruteforce

    # args: graph_name, p, k

    graph_names = args[0].split(",")
    p_vals = [float(p) for p in args[1].split(",")]
    k = int(args[2])

    runners.run_bruteforce(graph_names, p_vals, k)

if command == "corpus":
    # run corpus workload
    # args: graph_index, p_val, k, algos

    graph_index = int(args[0])
    p_val = float(args[1])
    k = int(args[2])

    # set algos to True
    for algo in args[3].split(","):
        algo_dict[algo] = True

    runners.run_corpus(graph_index, p_val, k, algo_dict)

if command == "corpus_multi":
    # wrapper for the corpus workload that runs multiple datasets with multiple p-values
    # this also performs the search for low/med/high p-values
    # the algorithms used here are the ones we standardized earlier

    # arguments are lower and upper indices within the corpus

    # args: index
    index= int(args[0])

    # algorithms we agreed on
    algo_dict = {
        "random": True,
        "greedy": False,
        "myopic": True,
        "naive_myopic": True,
        "gonzales": True,
        "furthest_non_seed_0": True,
        "furthest_non_seed_1": True,
        "bfs_myopic": True,
        "naive_bfs_myopic": True,
        "ppr_myopic": True,
        "naive_ppr_myopic": True,
        "degree_lowest_centrality_0": True,
        "degree_lowest_centrality_1": True,
        "degree_highest_degree_neighbor_0": True,
        "degree_highest_degree_neighbor_1": True,
    }

    runners.run_corpus_multi(algo_dict, index)

if command == 'test':
    import os
    import numpy as np
    files = os.listdir('./cache/evaluations')

    # open first file
    with open(f'./cache/evaluations/{files[200]}', 'rb') as f:
        d = np.load(f, allow_pickle=True)

        # convert to a regular dictionary
        d = d.item()

        print(d['myopic'])
    


if command == 'features':
    runners.run_features()

if command == 'ml1':
    runners_figs.run_ml_1()
    
if command == 'ml2':
    runners.run_ml_2()

if command == 'fig1':
    runners_figs.run_fig1()

if command == 'fig2':
    runners_figs.run_fig2()

if command == 'fig3a':
    runners_figs.run_fig3a()
    
if command == 'fig3a_temp':
    runners_figs.run_fig3a_temp()

if command == 'fig3b':
    runners_figs.run_fig3b()
    # runners_figs.run_fig3b_distribution()

if command == 'fig3c':
    runners_figs.run_fig3c()

if command == 'fig3d':
    runners_figs.run_fig3d()

if command == 'fig4':
    print('test')

if command == 'fig_spreadability':
    runners_figs.run_fig_spreadability()

if command == 'fig_avg_deg_best':
    runners_figs.run_fig_avg_deg_vs_best()

if command == 'corpus_stats':
    runners_figs.run_corpus_stats()

if command == 'bmatrix':
    runners.run_bmatrix()

if command == "augment_corpus":
    # augment corpus
    runners.run_augment_corpus()

if command == "timing_algos":
    algo_dict = {
        "random": True,
        "greedy": False,
        "myopic": True,
        "naive_myopic": True,
        "gonzales": True,
        "furthest_non_seed_0": True,
        "furthest_non_seed_1": True,
        "bfs_myopic": True,
        "naive_bfs_myopic": True,
        "ppr_myopic": True,
        "naive_ppr_myopic": True,
        "degree_lowest_centrality_0": True,
        "degree_lowest_centrality_1": True,
        "degree_highest_degree_neighbor_0": True,
        "degree_highest_degree_neighbor_1": True,
    }

    p_tag = args[0]
    index = int(args[1])

    # timing
    runners.run_algorithm_timing(algo_dict, p_tag, index)

if command == 'timing_probest':

    start = int(args[0])
    end = int(args[1])
    indices = list(range(start, end))
    p_tag = args[2]
    iterations_probest = args[3]

    runners.run_probest_timing(algo_dict, indices, p_tag, iterations_probest)

if command == 'fig_probest_timing':
    runners_figs.run_probest_timing()

if command == 'fig_algos_timing':
    alt = args[0] # toggle for smaller figure, 0 or 
    if alt == '0':
        alt = False
    else:
        alt = True
        
    runners_figs.run_algos_timing(alt)

if command == 'apsp':
    runners.run_fast_apsp()

if command == 'fig_ensemble':
    runners_figs.run_ensemble()

if command == 'ensemble_ml':
    runners.run_ensemble_ml()