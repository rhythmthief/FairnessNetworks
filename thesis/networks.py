import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import hashlib
import pandas as pd


def get_graph(name):
    if name == 'eu':
        return get_eu()
    elif name == 'arenas':
        return get_arenas()
    elif name == 'irvine':
        return get_irvine()
    elif name == 'synth0':
        return get_synth0()
    elif name == 'synth1':
        return get_synth1()
    elif name == 'synth2':
        return get_synth2()
    elif name == 'synth3':
        return get_synth3()
    elif name == 'synth4':
        return get_synth4()
    elif name == 'synth5':
        return get_synth5()
    elif name == 'synth6':
        return get_synth6()
    elif name == 'synth7':
        return get_synth7()
    elif name == 'synth8':
        return get_synth8()
    elif name == 'synth9':
        return get_synth9()
    else:
        raise ValueError("Invalid network name: {}".format(name))

def wrap_node_list(G):
    # wrap each node as a list for 
    # this was used early on for constructing the b-matrix
    seed_lists = []
    for i in range(G.number_of_nodes()):
        seed_lists.append([i])

    return seed_lists

def save_net_figure(G):
    # draw graph without labels and with small node size
    nx.draw(G, pos=nx.spring_layout(G, seed=1337), with_labels=False, node_size=10)

    # draw number of nodes and edges on the plot
    plt.text(0.7, 0.7, 'Nodes: ' + str(G.number_of_nodes()) + '\nEdges: ' + str(G.number_of_edges()), fontsize=12)
    
    # save figure
    plt.savefig(f'../figures/{G.name}_graph.png', bbox_inches='tight')
    plt.clf()

def reset_index(G):
    # Reset node labels to be indexed from 0
    # This is basically just to make the data play nice with the code
    # https://networkx.org/documentation/stable/reference/generated/networkx.relabel.relabel_nodes.html
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    return G

def get_eu():
    '''
    Returns the largest strongly connected component of the email-Eu-core network.

    Nodes: 803
    Edges: 24729

    Source: https://snap.stanford.edu/data/email-Eu-core.html
    '''

    # read in the directed graph
    G = nx.read_edgelist('../datasets/email-Eu-core.txt',
                         create_using=nx.DiGraph(), nodetype=int)
    G.name = 'eu'

    # Get the strongly connected components
    scc = list(nx.strongly_connected_components(G))

    # Find the largest component
    largest_scc = max(scc, key=len)

    # Create a subgraph with the largest SCC
    G = G.subgraph(largest_scc)

    # convert to undirected graph
    G = G.to_undirected()

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    G = reset_index(G)

    return G


def get_arenas():
    '''
    Source: http://konect.cc/networks/arenas-email/
    '''

    G = nx.read_edgelist('../datasets/out.arenas-email', nodetype=int)
    G.name = 'arenas'
    G = reset_index(G)

    return G


def get_irvine():
    '''
    Source: http://konect.cc/networks/opsahl-ucsocial/
    '''

    edge_list = []

    # for every line in the file, extract first two numbers to the edge list
    # the other two numbers are metadata
    for line in open('../datasets/out.opsahl-ucsocial', 'r'):
        split = line.split(" ")
        edge_list.append((int(split[0]), int(split[1])))

    # create a new graph from the edge list
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    G.name = 'irvine'

    # Get the strongly connected components
    scc = list(nx.strongly_connected_components(G))

    # Find the largest component
    largest_scc = max(scc, key=len)

    # Create a subgraph with the largest SCC
    G = G.subgraph(largest_scc)

    # convert to undirected graph
    G = G.to_undirected()

    G = reset_index(G)

    return G

###########################
# SYNTHETIC TEST NETWORKS #
###########################

def get_synth0():
    # returns a simple chain-like graph for testing
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4),
                     (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)])

    G.name = 'synth0'

    return G

def get_synth1():
    # returns a simple loop-like graph for testing
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])

    G.name = 'synth1'

    return G


def get_synth2():
    # returns a simple core-periphery graph for testing
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])

    G.name = 'synth2'

    return G


def get_synth3():
    # returns a simple graph for testing algorithms
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (4, 5), (2, 6)])

    G.name = 'synth3'

    return G

def get_synth4():
    # returns a simple erdos renyi graph for testing algorithms
    G = nx.erdos_renyi_graph(200, 0.01, seed=1337)

    # Find the largest component
    largest_cc = max(nx.connected_components(G), key=len)

    # Create a subgraph with the largest CC
    G = G.subgraph(largest_cc)

    G = reset_index(G)

    G.name = 'synth4'

    save_net_figure(G)

    return G

def get_synth5():
    # simple triangle graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    G.name = 'synth5'

    return G

def get_synth6():
    # simple fully connected graph with four nodes
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1,3), (2,3) ])

    G.name = 'synth6'

    return G


def get_synth7():
    # simple fully connected graph with four nodes
    G = nx.Graph()
    G.add_edges_from([(0,1), (0, 2), (1, 2),(2,0),(3,2)])

    G.name = 'synth7'

    return G

def get_synth8():
    # a simple graph based on the degree distribution of the EU network
    G = get_arenas()

    # get the degree distribution
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

    # sample a degree sequence from the degree distribution
    np.random.seed(42)
    degree_sequence = np.random.choice(degree_sequence, size=50, replace=False)

    if sum(degree_sequence) % 2 != 0:
        degree_sequence[0] = degree_sequence[0] + 1

    # create a new graph with the sampled degree sequence
    G = nx.configuration_model(degree_sequence, seed=42)

    # turn into a simple graph
    G = nx.Graph(G)

    # remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Find the largest component
    largest_cc = max(nx.connected_components(G), key=len)

    # Create a subgraph with the largest CC
    G = G.subgraph(largest_cc)

    G = reset_index(G)

    G.name = 'synth8'

    save_net_figure(G)

    return G

def get_synth9():
    # returns a core-periphery graph for testing
    # with a core of 5 fully connected nodes and a periphery of 10 nodes for each core
    # this graph shows that Myopic does not return optimal seed sets
    G = nx.Graph()

    # get all combinations of the core nodes
    nodes = list(itertools.combinations(range(5), 2))

    for node in range(5):
        for node_ in range(node * 10+5, node * 10 + 15):
            nodes.append((node, node_))

    G.add_edges_from(nodes)

    G.name = 'synth9'

    save_net_figure(G)

    return G


def get_synth10():
    # returns a star-shaped graph, center node, four strands, two nodes each

    G = nx.Graph()

    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1,5), (2,6), (3,7), (4,8)])

    G.name = 'synth10'

    # visualize

    import matplotlib.pyplot as plt
    
    nx.draw(G, pos=nx.spring_layout(G, seed=1337), with_labels=True, node_size=10)

    plt.show()


    return G

def augment_corpus():
    return None

def get_corpus_graph(index, get_domain=False):
    # returns a graph from the corpus

    # read in the corpus dataframe
    df = pd.read_pickle('../datasets/corpus_augmented.pkl')

    # create a graph from the edge list
    nodes = df.iloc[index]['nodes_id']
    edges = df.iloc[index]['edges_id']
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    G.name = df.iloc[index]['hashed_network_name']
    
    # print(index)
    # print(df.iloc[index]['title'])
    # print(df.iloc[index]['networkDomain'])
    # print(df.iloc[index]['network_name'])

    if get_domain:
        return G, df.iloc[index]['networkDomain']
    else:
        return G
