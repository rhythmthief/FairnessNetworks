import time
import networks as networks
import experiments as exp
import bruteforce as bf
import numpy as np
import matplotlib.pyplot as plt
import experiments as exp
import networks as networks
import os
import numpy as np
import networkx as nx
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import spreadability as spread
import probability as prob

algo_dict = {
        "random": False,
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


def fit_line(x, y, intercept=None):
    # fit a line to the data
    a, b = np.polyfit(x, y, 1)

    # return the slope and the fitted y values
    return [a], a * x + b, b

# suppress networkx future warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# suppress deprecated warning
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def run_algorithm_assessment(graph_names, p_vals, algo_dict, k):
    graphs = []
    # default values
    iterations = 1
    use_cache = False # this is for in-algo cache, separate from caching evaluations

    if graph_names != 'corpus':
        for graph_name in graph_names:
            graphs.append(networks.get_graph(graph_name))
    else:
        networks.get_corpus_graphs()


    for G in graphs:
        for p in p_vals:
            print("Running {} with p = {}".format(graph_name, p))
            exp.run_specified_experiments(G.copy(), k=k, p=p, iterations=iterations, use_cache=use_cache, algo_dict=algo_dict, draw_fig=True)


def run_bruteforce(graph_names, p_vals, k):
    for graph_name in graph_names:
        G = networks.get_graph(graph_name)
        
        for p in p_vals:
            print("Running bruteforce on {} with p = {}".format(graph_name, p))
            bf.run_bruteforce(G.copy(), p=p, k=k)

def run_corpus(graph_index, p_val, k, algo_dict):

    iterations = 20

    G = networks.get_corpus_graph(graph_index)

    exp.run_specified_experiments(G.copy(), k=k, p=p_val, iterations=iterations, use_cache=False, algo_dict=algo_dict, save_evals=True, draw_fig=False)

def run_corpus_multi(algo_dict, graph_index):
    p_vals_dict = {"05": 0.5, "04": 0.4, "03": 0.3}
    iterations = 20
    k = 10

    G = networks.get_corpus_graph(graph_index)

    print(f'\nRunning {G.name}')
    print(f'Performing spreadability search')
    # search for pvals that give low, med, and high spreadability
    p_vals_dict_ = spread.search(G, 1000) # 1000 trials

    # extend p_vals_dict with p_vals_dict_
    p_vals_dict.update(p_vals_dict_)

    print(f'Search successful.')
    print(f'p_vals_dict = {p_vals_dict}')

    for p_tag in p_vals_dict.keys():
        exp.run_specified_experiments(G.copy(), k=k, p=p_vals_dict[p_tag], iterations=iterations, use_cache=False, algo_dict=algo_dict, save_evals=True, draw_fig=False, p_tag=p_tag)

def run_algorithm_timing(algo_dict, p_tag, index):
    # run algorithm timing

    # does cache file already exist?
    if os.path.exists(f'./cache/timing_algos/{p_tag}/times_{p_tag}_{index}.npz'):
        # load the file
        with np.load(f'./cache/timing_algos/{p_tag}/times_{p_tag}_{index}.npz') as data:
            times = dict(data.items())
            print('Loaded times from cache.')
    else:
        # read in network corpus
        df = pd.read_pickle('../datasets/corpus_augmented.pkl')
        #num_nets = 10

        times = {}

        # get cached evaluation filenames
        filenames = os.listdir("./cache/evaluations/")

        # split filenames with _ delim
        filenames_split = [filename.split('_') for filename in filenames]

        i = index

        print(f'Processing network {i}')

        # get hashed net name and net itself
        # assumes that sorting had not been changed
        network_hash = df['hashed_network_name'][i]
        G = networks.get_corpus_graph(i)

        # find evaluation files for the network
        eval_files = [filename for filename in filenames if filename.startswith(network_hash)]

        if len(eval_files) == 0:
            # skip for now
            return

        p = 0

        filenames_split = [eval_files.split('_') for eval_files in eval_files]

        # find p_tag and assign p
        for split in filenames_split:
            if split[1] == p_tag:
                p = float(split[2][:-4])
                break

        if p == 0:
            return
        
        res = exp.run_timing_experiment(G, algo_dict, p=p, p_tag=p_tag)

        # save the time
        for key in res.keys():
            if key not in times:
                times[key] = []
            times[key].append(res[key])

        # flatten each list of times
        for key in times.keys():
            times[key] = [item for sublist in times[key] for item in sublist]

        # save the runtimes to a file
        np.savez(f'./cache/timing_algos/times_{p_tag}_{index}.npz', **times)

def run_probest_timing(algo_dict, indices, p_tag, iterations_probest):
    # is it cached?

    iterations_exp = 10

    times = {}
    times['inline_times'] = []
    times['inline_n'] = []
    times['inline_m'] = []

    # does cache file already exist?
    if os.path.exists(f'./cache/timing_probest/times_{p_tag}_{iterations_probest}.npz'):
        # load the file
        with np.load(f'./cache/timing_probest/times_{p_tag}_{iterations_probest}.npz') as data:
            times = dict(data.items())
            print('Loaded times from cache.')
    else:
        # read in network corpus
        df = pd.read_pickle('../datasets/corpus_augmented.pkl')

        # get cached evaluation filenames
        filenames = os.listdir("./cache/evaluations/")

        # split filenames with _ delim
        filenames_split = [filename.split('_') for filename in filenames]

        # get each network
        for i in indices:
            print(f'Processing network {i+1} of {len(indices)}')

            # get hashed net name and net itself
            # assumes that sorting had not been changed
            network_hash = df['hashed_network_name'][i]
            G = networks.get_corpus_graph(i)

            # find evaluation files for the network
            eval_files = [filename for filename in filenames if filename.startswith(network_hash)]

            if len(eval_files) == 0:
                # skip for now
                continue

            p = 0

            filenames_split = [eval_files.split('_') for eval_files in eval_files]

            # find p_tag and assign p
            for split in filenames_split:
                if split[1] == p_tag:
                    p = float(split[2][:-4])
                    break

            if p == 0:
                continue # skip
            
            # roll random seeds
            initial_seeds = np.random.choice(G.nodes, size=iterations_exp, replace=False)

            delta = 0

            for seed in initial_seeds:
                start_time = time.time()
                prob.estimate(G, p, [seed], int(iterations_probest), threads=1)
                end_time = time.time()
                delta += end_time - start_time

            delta = delta / iterations_exp

            print(delta)

            times[network_hash] = delta
            times['inline_times'].append(delta)
            times['inline_n'].append(G.number_of_nodes())
            times['inline_m'].append(G.number_of_edges())

        #save the runtimes to a file
        np.savez(f'./cache/timing_probest/times_{p_tag}_{iterations_probest}.npz', **times)

    print(times)

def run_features():
    def compute_features(G):
        number_nodes = len(G.nodes())
        number_edges = len(G.edges())
        avg_degree = np.mean([G.degree(node) for node in G.nodes()])
        max_degree = np.max([G.degree(node) for node in G.nodes()])
        degree_variance = np.var([G.degree(node) for node in G.nodes()])
        transitivity = nx.transitivity(G)
        avg_shortest_path = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
        assortativity = nx.degree_assortativity_coefficient(G)
        highest_deg_node_eccentricity = nx.eccentricity(G, v=np.argmax([G.degree(node) for node in G.nodes()])) # eccentricity of the highest-degree node
        mean_10_highest_deg_nodes_eccentricity = np.mean([nx.eccentricity(G, v=node) for node in np.argsort([G.degree(node) for node in G.nodes()])[-10:]]) # mean eccentricity of the 10 highest-degree nodes

        return number_nodes, number_edges, avg_degree, max_degree, degree_variance, transitivity, avg_shortest_path, diameter, assortativity, highest_deg_node_eccentricity, mean_10_highest_deg_nodes_eccentricity

    features = ['name', 'domain', 'number_nodes', 'number_edges', 'avg_degree', 'max_degree', 'degree_variance', 'transitivity', 'avg_shortest_path', 'diameter', 'assortativity', 'highest_deg_node_eccentricity', 'mean_10_highest_deg_nodes_eccentricity']

    data = []

    net_features_dict = {}

    for f in features:
        net_features_dict[f] = []

    for i in range(174):
        print('processing:', i+1, 'of 174')
        # we are skipping network 174 for now
        G, domain = networks.get_corpus_graph(i)

        # compute features
        data = [G.name] + [domain] + list(compute_features(G)) 

        for f, d in zip(features, data):
            net_features_dict[f].append(d)

    # save npz file
    np.savez('./cache/features.npz', **net_features_dict)

def run_bmatrix():
    # code from early problem exploration
    
    def net_stats(networks):
        # compute the average number of nodes
        avg_nodes = np.mean([len(G.nodes()) for G in networks])

        # compute the average number of edges
        avg_edges = np.mean([len(G.edges()) for G in networks])

        # compute the average degree
        avg_degree = np.mean([np.mean([G.degree(node) for node in G.nodes()]) for G in networks])

        # compute the average clustering coefficient
        avg_clustering = np.mean([nx.average_clustering(G) for G in networks])

        # compute the average shortest path length
        avg_shortest_path = np.mean([nx.average_shortest_path_length(G) for G in networks])

        # return
        return len(networks), avg_nodes, avg_edges, avg_degree, avg_clustering, avg_shortest_path

    # load corpus dataframe
    corpus = pd.read_pickle('../datasets/corpus_augmented.pkl')

    domains = corpus['networkDomain'].unique()

    for dom in domains:
        # get row indices where networkDomain is dom
        indices = list(corpus[corpus['networkDomain'] == dom].index)

        # make a list of networks for each row in corpus
        nets = [networks.get_corpus_graph(i) for i in indices]
        # compute stats
        stats = net_stats(nets)
        print(dom, stats)

    for dom in domains:

        print('\n###', dom, '###')

        for p in [0.5, 0.4, 0.3]:
            algo_dict_betas = {
                "random": [],
                "myopic": [],
                "naive_myopic": [],
                "gonzales": [],
                "furthest_non_seed_0": [],
                "furthest_non_seed_1": [],
                "bfs_myopic": [],
                "naive_bfs_myopic": [],
                "ppr_myopic": [],
                "naive_ppr_myopic": [],
                "degree_lowest_centrality_0": [],
                "degree_lowest_centrality_1": [],
                "degree_highest_degree_neighbor_0": [],
                "degree_highest_degree_neighbor_1": [],
            }

            performance_dict = {
                "random": 0,
                "myopic": 0,
                "naive_myopic": 0,
                "gonzales": 0,
                "furthest_non_seed_0": 0,
                "furthest_non_seed_1": 0,
                "bfs_myopic": 0,
                "naive_bfs_myopic": 0,
                "ppr_myopic": 0,
                "naive_ppr_myopic": 0,
                "degree_lowest_centrality_0": 0,
                "degree_lowest_centrality_1": 0,
                "degree_highest_degree_neighbor_0": 0,
                "degree_highest_degree_neighbor_1": 0,
            }

            print(f"--- p = {p}")

            for filename in os.listdir("./cache/evaluations/"):
                network_hash = filename[:-8]

                # get network row using the hash
                row = corpus[corpus['hashed_network_name'] == network_hash]

                # get network domain
                network_domain = row['networkDomain'].values[0]


                if filename[-7:-4] == str(p) and network_domain == dom: # only consider files with the correct p value
                    with open(os.path.join("./cache/evaluations/", filename), 'rb') as f:
                        # load file as a dictionary
                        d = np.load(f, allow_pickle=True)

                        # convert to a regular dictionary?
                        d = d.item()

                        for key in d.keys():
                            
                            # append 0 to the begining of vals
                            vals = list(d[key])

                            # truncate vals to the first 10 entries
                            # vals = vals[:10]

                            vals.insert(0, 0)

                            # compute slope of the line of best fit for vals
                            slope, intercept = np.polyfit(range(0, len(vals)), vals, 1)

                            # append slope to the list of slopes for the algorithm
                            algo_dict_betas[key].append(slope)

            performance_dict = {}

            # compute average of each list
            for key in algo_dict_betas.keys():
                performance_dict[key] = np.mean(algo_dict_betas[key])

            # sort the dictionary by value in descending order
            performance_dict = dict(sorted(performance_dict.items(), key=lambda item: item[1], reverse=True))

            # print the dictionary
            # https://bobbyhadz.com/blog/python-add-zeros-after-decimal
            for key in performance_dict.keys():
                print(f'{performance_dict[key]:.6f}', '\t',key)

def run_augment_corpus():
    # augment the corpus provided for this project
    # with more diverse networks

    # read read each pickle file as dataframe in datasets/domains, one at a time

    names = ['cfn_aug_soc_final_red.pkl', 'cfn_econ_aug_final.pkl', 'cfn_subset_bio_proj.pkl', 'cfn_subset_info.pkl', 'cfn_subset_tech.pkl', 'cfn_subset_tran.pkl']

    # check if path exists
    # if not, create it
    # initial read to get columns
    df = pd.read_pickle(f'../datasets/corpus_old/domains/{names[0]}')

    # print column names
    cols = df.columns

    net_names = []

    print(cols)

    df_full = None

    def reset_index(G):
    # Reset node labels to be indexed from 0
    # This is basically just to make the data play nice with the code
    # https://networkx.org/documentation/stable/reference/generated/networkx.relabel.relabel_nodes.html
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

        return G

    def simplify_graph(G):
        # convert to undirected
        G = G.to_undirected()
        
        # remove self loops
        G.remove_edges_from(nx.selfloop_edges(G))

        # find gcc
        G = G.subgraph(max(nx.connected_components(G), key=len))

        # reset index
        G = reset_index(G)

        return G


    for name in names:
        df = pd.read_pickle(f'../datasets/corpus_old/domains/{name}')

        # some of the networks are missing some of the columns, making the entire setup clumsy
        # so we add the missing columns with None values
        # so as not to lose any data
        for col in ['multigraph','timestamps','from_cfn_corpus']:
            if col not in df.columns:
                # add as a new column with all values as None
                df[col] = None

        if name == 'cfn_aug_soc_final_red.pkl':
            # we need to augment this domain with some other networks
            
            # how mady networks did we start with?

            # Norwegian networks are indexed 1, 2, ..., 38
            # and there is 38 of them in total,
            # we will remove every other network, since they are all very similar and we want to diversify the dataset
            for i in range(2,38, 2):
                df = df.drop(i)

            # there are 14 networks from Facebook100 that match our criteria
            # we will add them to the dataframe
            for filename in os.listdir("../datasets/facebook100/"):
                with open(os.path.join("../datasets/facebook100/", filename), 'r') as f:
                    G = nx.read_edgelist(f)

                    G = simplify_graph(G)

                    # prepare a new dataframe entry
                    new_entry = {
                        'title': 'Facebook 100', 
                        'nodes_id': list(G.nodes()), 
                        'edges_id': list(G.edges()), 
                        'network_name': filename[:-4], 
                        'networkDomain': 'Social', 
                        'sourceUrl': 'http://arxiv.org/abs/1102.2166', 
                        'citation': 'A.L. Traud, P.J. Mucha, and M.A. Porter. "Social structure of Facebook networks." Physica A, 391(16), 4165–4180 (2012)', 
                        'from_cfn_corpus': False, 
                        'number_edges': len(list(G.edges())), 
                        'number_nodes': len(list(G.nodes())), 
                        'multigraph': False, 
                        'timestamps': False}

                    # add to df
                    df = df.append(new_entry, ignore_index=True)


            # A network from Copenhagen Networks Study fits our criteria
            # we will add it to the dataframe
            # two others, calls and sms, are offline-ish but after simplification
            # they are too small

            df_fb = pd.read_csv('../datasets/fb_friends.csv')

            # for fb, column 1 is source node, column 2 is target node

            # create from fb edgelist
            edges_fb = [(source, target) for source, target in zip(df_fb['# user_a'], df_fb['user_b'])]

            # create new graph
            G_fb = nx.from_edgelist(edges_fb)
            
            # simplify
            G_fb = simplify_graph(G_fb)

            new_entry_fb = {
                'title': 'Copenhagen Networks Study',
                'nodes_id': list(G_fb.nodes()),
                'edges_id': list(G_fb.edges()),
                'network_name': 'copenhagen_fb',
                'networkDomain': 'Social',
                'sourceUrl': 'https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433/1',
                'citation': 'P. Sapiezynski, et al., "Interaction data from the Copenhagen Networks Study." Scientific Data 6, 315 (2019), https://doi.org/10.1038/s41597-019-0325-x [@sci-hub]',
                'from_cfn_corpus': False,
                'number_edges': len(list(G_fb.edges())),
                'number_nodes': len(list(G_fb.nodes())),
                'multigraph': False,
                'timestamps': False
            }

            # add to df
            df = df.append(new_entry_fb, ignore_index=True)

            # next network we are using is out.dnc-corecipient
            # we skip the first line, since it is a comment
            df_dnc = pd.read_csv('../datasets/out.dnc-corecipient', sep='\t', skiprows=1)

            df_dnc.columns = ['from', 'to', 'count']

            # column 1 is source node, column 2 is target node
            edges_dnc = [(source, target) for source, target in zip(df_dnc['from'], df_dnc['to'])]

            # create a new graph
            G_dnc = nx.from_edgelist(edges_dnc)

            G_dnc = simplify_graph(G_dnc)

            # prepare a new dataframe entry
            new_entry_dnc = {
                'title': 'DNC emails (2016)',
                'nodes_id': list(G_dnc.nodes()),
                'edges_id': list(G_dnc.edges()),
                'network_name': 'dnc',
                'networkDomain': 'Social',
                'sourceUrl': 'http://konect.cc/networks/dnc-corecipient/',
                'citation': 'J. Kunegis, "DNC emails co-recipients." KONECT, the Koblenz Network Collection (2016), https://doi.org/10.1145/2487788.2488173 [@sci-hub]',
                'from_cfn_corpus': False,
                'number_edges': len(list(G_dnc.edges())),
                'number_nodes': len(list(G_dnc.nodes())),
                'multigraph': False,
                'timestamps': False
            }

            # add to df
            df = df.append(new_entry_dnc, ignore_index=True)


            # read in scientific collaboration network collaboration_network.graphml
            G_scientific = nx.read_graphml('../datasets/collaboration_network.graphml')

            # simplify graph
            G_scientific = simplify_graph(G_scientific)

            # prepare a new dataframe entry
            new_entry_scientific = {
                'title': 'New Zealand scientific collaborations (2015)',
                'nodes_id': list(G_scientific.nodes()),
                'edges_id': list(G_scientific.edges()),
                'network_name': 'nz_scientific_collabs',
                'networkDomain': 'Social',
                'sourceUrl': 'https://doi.org/10.6084/m9.figshare.5705167',
                'citation': 'S. Aref, D. Friggens, and S. Hendy, "Analysing scientific collaborations of New Zealand institutions using Scopus bibliometric data." Proc. Australasian Comp. Sci. Week Multiconf. (ACSW ’18), Article 49, 1-10 (2018), https://doi.org/10.1145/3167918.3167920 [@sci-hub]',
                'from_cfn_corpus': False,
                'number_edges': len(list(G_scientific.edges())),
                'number_nodes': len(list(G_scientific.nodes())),
                'multigraph': False,
                'timestamps': False
            }

            # add to df
            df = df.append(new_entry_scientific, ignore_index=True)



            # add Bitcoin Alpha network
            # read in csv
            df_bitcoin = pd.read_csv('../datasets/soc-sign-bitcoinalpha.csv', sep=',', header=None)

            # SOURCE, TARGET, RATING, TIME
            df_bitcoin.columns = ['source', 'target', 'rating', 'time']

            # create a new graph
            G_bitcoin = nx.from_pandas_edgelist(df_bitcoin, source='source', target='target', edge_attr='rating')

            # simplify graph
            G_bitcoin = simplify_graph(G_bitcoin)

            # prepare a new dataframe entry
            new_entry_bitcoin = {
                'title': 'Bitcoin Alpha trust network (2017)',
                'nodes_id': list(G_bitcoin.nodes()),
                'edges_id': list(G_bitcoin.edges()),
                'network_name': 'bitcoin_alpha',
                'networkDomain': 'Social',
                'sourceUrl': 'http://snap.stanford.edu/data/soc-sign-bitcoinalpha.html',
                'citation': 'S. Kumar, F. Spezzano, V.S. Subrahmanian, and C. Faloutsos, "Edge weight prediction in weighted signed networks." IEEE 16th International Conference on Data Mining (ICDM), 221-230 (2016), https://doi.org/10.1109/icdm.2016.0033 [@sci-hub]',
                'from_cfn_corpus': False,
                'number_edges': len(list(G_bitcoin.edges())),
                'number_nodes': len(list(G_bitcoin.nodes())),
                'multigraph': False,
                'timestamps': False
            }

            # add to df
            df = df.append(new_entry_bitcoin, ignore_index=True)


        if name == 'cfn_econ_aug_final.pkl':
            # we need to augment this domain with some other networks

            # EU networks are indexed 2-43
            # we will remove every other network
            for i in range(2,44, 2):
                df = df.drop(i)

            # There are 21 networks in the Austrian dataset that we can use

            # read csv into dataframe with numbers 0-5 as column names
            df_austrian = pd.read_csv('../datasets/OGDEXT_BINNENWAND_1.csv', sep=';')
            df_austrian.columns = [str(i) for i in range(6)]

            # get all unique values in the first column
            years = df_austrian['0'].unique()

            for y in years:      
                # get all rows where the first column is equal to y
                df_ = df_austrian[df_austrian['0'] == y]

                # second column is the source node
                # third column is the target node

                # get all unique values in the second column
                nodes = list(df_['1'].unique())

                # get all unique values in the third column
                nodes_ = list(df_['2'].unique())

                # create a mapping from the old node labels to new node labels
                mapping = {node: i for i, node in enumerate(nodes + nodes_)}

                # map nodes to new node labels in a list
                edges = [(mapping[source], mapping[target]) for source, target in zip(df_['1'], df_['2'])]

                # create a new graph
                G = nx.from_edgelist(edges)

                # simplify graph
                G = simplify_graph(G)

                # prepare a new dataframe entry
                new_entry = {
                    'title': 'Austrian internal migrations (2002-2022)',
                    'nodes_id': list(G.nodes()),
                    'edges_id': list(G.edges()),
                    'network_name': f'austrian_{y}',
                    'networkDomain': 'Economic',
                    'sourceUrl': 'https://data.statistik.gv.at/web/meta.jsp?dataset=OGDEXT_BINNENWAND_1',
                    'citation': '"Internal migration within Austria acc.to communes since 2002 (status of 2022)", Statistik Austria, https://data.statistik.gv.at',
                    'from_cfn_corpus': False,
                    'number_edges': len(G.edges()),
                    'number_nodes': len(G.nodes()),
                    'multigraph': False,
                    'timestamps': False
                    }

                # add to df
                df = df.append(new_entry, ignore_index=True)
        
        # reset index for the dataframe
        df = df.reset_index(drop=True)

        # we are constructing a single large dataframe
        if df_full is None:
            df_full = df
        else:
            df_full = df_full.append(df)


    # reset index for the dataframe
    df_full = df_full.reset_index(drop=True)

    # unfortunately, some of the networks left in the corpus by the predecessor have self-loops
    # so instead of chasing them down, we will just bulk-process the entire network corpus again
    # and remove all self-loops
    # this is a bit of a hack, but it works

    # iterate over each row in the dataframe
    for i, row in df_full.iterrows():
        # read in the network
        G = nx.from_edgelist(row['edges_id'])

        # simplify the network
        G = simplify_graph(G)

        # update the dataframe
        df_full.at[i, 'edges_id'] = list(G.edges())
        df_full.at[i, 'nodes_id'] = list(G.nodes())
        df_full.at[i, 'number_edges'] = len(G.edges())
        df_full.at[i, 'number_nodes'] = len(G.nodes())

        # for manual debugging
        if len(G.nodes()) < 500:
            print('uh-oh')

            # what row is it?
            print(row)            

    # hash network_name column with md5 and insert as a new column

    # read out network_name column
    network_names = df_full['network_name'].to_list()
    hashed_names = []

    for n in network_names:
        hashed_names.append(hashlib.md5(n.encode()).hexdigest())

    df_full['hashed_network_name'] = hashed_names

    # sort the dataframe by number_nodes
    df_full.sort_values(by=['number_nodes'], ascending=True, inplace=True)


    # reset index for the dataframe, again, now that we've sorted
    df_full = df_full.reset_index(drop=True)

    # extract edges_id column
    edges_id = df_full['edges_id']

    # save as a pickle file
    df_full.to_pickle('../datasets/corpus_augmented.pkl')

    # print column of number_nodes
    print(df_full['number_nodes'])

    print('Corpus augmented.')


def export_corpus_gml():
    # export corpus to gml format

    # read in the augmented corpus
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # citations file to write into
    with open('../datasets/_CITATIONS.txt', 'w') as citations:
        # iterate over each row in the dataframe
        for i, row in df.iterrows():
            # read in the network
            G = nx.from_edgelist(row['edges_id'])

            # print citation, source url, hostedby
            print(row['hashed_network_name'])
            print(row['citation'])
            print(row['sourceUrl'])

            # write the the citation and source url to a file
            citations.write(f'{row["hashed_network_name"]}\n')
            citations.write(f'{row["citation"]}\n')
            citations.write(f'{row["sourceUrl"]}\n')
            citations.write('\n')
            

            # write out the network
            nx.write_gml(G, f'../datasets/gml/{row["hashed_network_name"]}.gml')

    print('Exported corpus to gml format.')