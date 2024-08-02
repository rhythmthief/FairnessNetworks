import networkx as nx
import numpy as np
import probability as prob
from multiprocessing import Pool
from collections import deque  # efficient queue implementation
from copy import deepcopy
import time


def get_algorithm(algo_name):
    '''
    Returns the algorithm class for the specified algorithm name.
    '''

    if algo_name == "random":
        return Random
    elif algo_name == "greedy":
        return Greedy
    elif algo_name == "myopic":
        return Myopic
    elif algo_name == "naive_myopic":
        return NaiveMyopic
    elif algo_name == "gonzales":
        return Gonzalez
    elif algo_name == "furthest_non_seed_0":
        return FurthestNonSeed
    elif algo_name == "furthest_non_seed_1":
        return FurthestNonSeedChooseNeighbor
    elif algo_name == "bfs_myopic":
        return BFSMyopic
    elif algo_name == "naive_bfs_myopic":
        return NaiveBFSMyopic
    elif algo_name == "ppr_myopic":
        return PPRMyopic
    elif algo_name == "naive_ppr_myopic":
        return NaivePPRMyopic
    elif algo_name == "degree_lowest_centrality_0":
        return DegreeLowestCentrality
    elif algo_name == "degree_lowest_centrality_1":
        return DegreeLowestCentralityChooseNeighbor
    elif algo_name == "degree_highest_degree_neighbor_0":
        return DegreeHighestDegreeNeighbor
    elif algo_name == "degree_highest_degree_neighbor_1":
        return DegreeHighestDegreeNeighborChooseNeighbor
    elif algo_name == "fig2":
        return [Myopic, Gonzalez, BFSMyopic, DegreeLowestCentrality, DegreeHighestDegreeNeighbor]
    else:
        raise Exception("Algorithm not found")

class Algorithm:
    '''
    Base class for algorithms.
    '''

    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.G = G
        self.p = p
        self.k = k
        self.ic_trials = ic_trials
        self.evaluations = []
        self.use_cache = use_cache
        self.threads = threads
        self.precompute_time = 0

        # initialize the seeds if needed
        if seeds == None:
            if use_cache:
                # initialize with a cache file
                self.load_cache()
            else:
                self.initialize_seeds()
        else:
            self.seeds = seeds

        # sanity check
        if self.k > len(self.G.nodes()):
            raise Exception(
                "k must be less than or equal to the number of nodes in the graph")

    def override_seeds(self, seeds):
        '''
        Overrides the seeds with the specified seeds.
        '''
        self.seeds = seeds

    def get_algo_name(self):
        '''
        Returns the name of the algorithm.
        '''
        return self.algo_name

    def initialize_seeds(self):
        '''
        Default seed initialization that picks the node with the highest degree.
        '''
        # get the index of the highest degree node
        i = np.argmax(np.array(self.G.degree())[:, 1])

        # initialize the seeds and decrement k
        self.seeds = [list(self.G.nodes())[i]]
        self.k -= 1

    def load_cache(self):
        '''
        Loads the cache for the algorithm.
        '''
        # try to read the file
        try:
            # read the file
            self.seeds = list(np.loadtxt(self.cache_filename, dtype=np.int32))
            self.k = self.k - len(self.seeds) # decrement k
            print('[{}] Loaded {} cached seeds.'.format(self.algo_name, len(self.seeds)))
            
            if len(self.seeds) == 0: # in case the file is empty
                self.initialize_seeds()
        
        except FileNotFoundError:
            # if the file does not exist, we start from scratch
            print('[{}] Could not locate cache.'.format(self.algo_name))
            open(self.cache_filename, 'w').close() # create the file

            # initialize the seeds normally
            self.initialize_seeds()

    def save_cache(self):
        '''
        Saves the cache for the algorithm.
        '''
        # save the seeds to the file
        np.savetxt(self.cache_filename, self.seeds, fmt='%d')

    def predict(self):
        '''
        Predicts the next k seeds.
        '''
        pass

    # legacy code for a different parallelization startegy
    def eval_mt_helper(self, G, p, seeds, ic_trials):
        return prob.estimate_single_thread(G, p, seeds, ic_trials)
    
    def evaluate(self):
        '''
        Evaluates the algorithm by running it
        '''
        # run the algorithm

        seeds = self.predict()

        subsets = []

        # create subsets of seeds to evaluate
        for i in range(1, len(seeds) + 1):
            # chose first i seeds
            subsets.append(seeds[:i])

        for i in range(len(subsets)):
            result = prob.estimate(self.G, self.p, subsets[i], self.ic_trials)
            self.evaluations.append(np.min(result))

        # find and return the minimum probability
        return self.evaluations


class Random(Algorithm):
    '''
    Picks k seeds uniformly at random,
    including the initial seed.
    '''

    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'random'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        self.precompute_time = 0
        super(Random, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def initialize_seeds(self):
        # Random algorithm does not need an initial seed
        self.seeds = []

    def predict(self):
        '''
        Picks k seeds uniformly at random,
        including the initial seed.
        '''

        if self.k > 0: # if we need to predict more seeds
            # pick k random nodes

            self.seeds = np.random.choice(
                self.G.nodes(), self.k, replace=False)

            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

# same as in the paper
class Greedy(Algorithm):
    '''
    Picks k seeds greedily.
    An approach where the next seed is chosen s.t. the lowest probability is maximized.
    '''

    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'greedy'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(Greedy, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def initialize_seeds(self):
        # Greedy algorithm does not need an initial seed
        self.seeds = []

    def greedy_mt_helper(self, G, p, seeds, ic_trials):
        '''
        Helper function for Multi-Threaded Greedy.
        '''

        probs = prob.estimate_single_thread(G, p, seeds, ic_trials)

        # get the minimum probability
        min_val = np.min(probs)

        return min_val

    def predict(self):
        if self.k > 0: # if we need to predict more seeds
            for k in range(self.k):
                # diagnostics because greedy is silly slow
                print("[greedy] {}/{}".format(k+1, self.k+1))

                # get difference of nodes and seeds
                candidates = list(set(self.G.nodes()) - set(self.seeds))

                # stores minimum probabilities for each candidate
                next_min = np.zeros(len(candidates))

                # multiprocessing
                # I have found that multiprocessing subsets is faster
                # than multiprocessing cascades
                next_min = np.array(Pool().starmap(self.greedy_mt_helper, [(self.G, self.p, self.seeds + [c], self.ic_trials) for c in candidates]))

                # get the index of the candidate with the highest minimum probability
                choice = np.argmax(next_min)

                # append a new seed
                self.seeds.append(candidates[choice])

                # save partial cache
                if self.use_cache:
                    # save the seeds to the file
                    self.save_cache()

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

class Myopic(Algorithm):
    '''
    Picks k seeds myopically.
    The initial seed is the node with the highest degree.
    '''

    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'myopic'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(Myopic, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def predict(self):

        if self.k > 0: # if we need to predict more seeds
            for _ in range(self.k):
                # get the probabilities for the current seed set
                probs = prob.estimate(self.G, self.p, self.seeds, self.ic_trials, self.threads)

                # get the index of the node with the minimum probability
                # choice = np.argmin(probs) # old oneliner that chooses the first minimum

                # choose a random node with the minimum probability
                min_val = np.min(probs)
                candidates = [i for i, j in enumerate(probs) if j == min_val]
                choice = np.random.choice(candidates)

                # append a new seed
                self.seeds.append(list(self.G.nodes())[choice])

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

class NaiveMyopic(Algorithm):
    '''
    Picks k seeds myopically from the initial probability estimate.
    The initial seed is the node with the highest degree.
    '''

    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'naive_myopic'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(NaiveMyopic, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def predict(self):
        if self.k > 0: # if we need to predict more seeds
            # get the probabilities for the current seed set
            probs = prob.estimate(self.G, self.p, self.seeds, self.ic_trials, self.threads)

            # choose k nodes with the lowest probabilities
            # by default, sorts with quicksort (O(n log n))
            # https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
            self.seeds.extend(list(np.argsort(probs)[:self.k]))

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds


class Gonzalez(Algorithm):
    '''
    Picks k seeds furthest from the seed set.
    The initial seed is the node with the highest degree.
    '''

    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'gonzalez'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(Gonzalez, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def predict(self):
        if self.k > 0: # if we need to predict more seeds
            # compute all pairs shortest paths

            time_start = time.time()
            apsp = dict(nx.all_pairs_shortest_path_length(self.G))
            self.precompute_time += time.time() - time_start
            
            for _ in range(self.k):

                # get difference of nodes and seeds
                candidates = list(set(self.G.nodes()) - set(self.seeds))

                # initialize a list of distances from each node to the seed set
                distances = np.zeros(len(candidates))

                # compute average distances from each node to the seed set
                for i, c in enumerate(candidates):
                    for s in self.seeds:
                        distances[i] += apsp[c][s]
                    distances[i] /= len(self.seeds)

                # get the index of the node with the maximum average distance
                choice = np.argmax(distances)

                # append a new seed
                self.seeds.append(candidates[choice])

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds


class FurthestNonSeed(Algorithm):
    '''
    Picks k seeds furthest from the non-seeds in terms of centrality
    The initial seed is the node with the highest degree.
    '''

    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'furthest_non_seed'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(FurthestNonSeed, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    # minimize the distance to the center of the non-seed nodes
    def predict(self):
        if self.k > 0: # if we need to predict more seeds

            time_start = time.time()
            # compute closeness centrality for all nodes
            closeness = nx.closeness_centrality(self.G)
            self.precompute_time += time.time() - time_start

            for _ in range(self.k):
                # choose node with min closeness centrality s.t.
                # it is not in the seed set

                # get difference of nodes and seeds
                candidates = list(set(self.G.nodes()) - set(self.seeds))

                # get the node with the lowest closeness centrality
                choice = min(candidates, key=closeness.get)

                # append as a new seed
                self.seeds.append(choice)

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

class FurthestNonSeedChooseNeighbor(Algorithm):
    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'furthest_non_seed_choose_neighbor'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(FurthestNonSeedChooseNeighbor, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    # minimize the distance to the center of the non-seed nodes
    def predict(self):
        if self.k > 0: # if we need to predict more seeds

            time_start = time.time()
            # compute closeness centrality for all nodes
            closeness = nx.closeness_centrality(self.G)
            self.precompute_time += time.time() - time_start

            for _ in range(self.k):
                # choose node with min closeness centrality s.t.
                # it is not in the seed set

                # get difference of nodes and seeds
                candidates = list(set(self.G.nodes()) - set(self.seeds))

                # get the node with the lowest closeness centrality
                choice = min(candidates, key=closeness.get)

                # get the neighbors of the node that aren't in the seed set
                neighbors = list(set(self.G.neighbors(choice)) - set(self.seeds))

                if len(neighbors) > 0:
                    # get the node with the highest degree among the neighbors
                    # otherwise just pick the node we found earlier
                    choice = max(neighbors, key=self.G.degree)

                # append as a new seed
                self.seeds.append(choice)

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

class BFSMyopic(Algorithm):
    '''
    The initial seed is the node with the highest degree.
    Currently suffers from precision loss?
    '''

    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'bfs_myopic'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(BFSMyopic, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    class Node():
        def __init__(self, id, p):
            self.id = id
            self.from_parent_prob = []
            self.from_neighbor_prob = []
            self.to_neighbor_prob = None
            self.to_child_prob = None
            self.children = []
            self.p = p
            self.activation_probs = [] # activation probabilities from all seeds
            self.final_activation_prob = None

        def compute_to_neighbor_prob(self):
            # compute the probability of transmission to neighbors in the same layer relative to the seed
            # this takes into account probabilities from parents only

            # if the check fails, it's actually a seed node
            if self.to_neighbor_prob is None:
                 # probabilities for each event that it does not happen
                probs_log = np.log1p(-np.exp(self.from_parent_prob))

                # compute the log of the probability that none of the transmissions occur
                prob_no_transmission_log = np.sum(probs_log)

                # compute the log of the probability that at least one transmission occurs
                prob_at_least_one_transmission_log = np.log1p(-np.exp(prob_no_transmission_log))

                # set probability that at least one of the transmissions occurs
                self.to_neighbor_prob = prob_at_least_one_transmission_log + np.log(self.p)

        def compute_to_child_prob(self):
            # compute the probability of transmission to children in the next layer relative to the seed
            # this takes into account probabilities from parents and probabilities from neighbors

            # if this check fails, it's actually a seed node
            if self.to_child_prob is None:
                probs_log = np.concatenate((self.from_parent_prob, self.from_neighbor_prob))

                # compute the probability for events not to happen
                probs_log = np.log1p(-np.exp(probs_log))

                # compute the log of the probability that none of the transmissions occur
                prob_no_transmission_log = np.sum(probs_log)

                # compute the probability that at least one transmission occurs
                prob_at_least_one_transmission_log = np.log1p(-np.exp(prob_no_transmission_log))

                self.activation_probs.append(prob_at_least_one_transmission_log)
                self.to_child_prob = prob_at_least_one_transmission_log + np.log(self.p)

        def compute_activation_prob(self):
            # computes the probability of activation for this node
            # takes into account all seeds

            if self.final_activation_prob !=0: # 0 because of log space
                if np.any(np.array(self.activation_probs) > -1e-15):
                    # if any of the activation probabilities is very close to zero,
                    # this node is basically guaranteed to be activated
                    # so we skip computations and assign a probability of 1
                    # if we were to compute, np.log1p would throw a division by zero warning
                    self.final_activation_prob = np.log(1)
                else:
                    # compute the probability for events not to happen
                    probs_log = np.log1p(-np.exp(self.activation_probs))

                    # compute the log of the probability that none of the transmissions occur
                    prob_no_transmission_log = np.sum(probs_log)

                    # compute the log of the probability that at least one transmission occurs
                    prob_at_least_one_transmission_log = np.log1p(-np.exp(prob_no_transmission_log))

                    # set the final activation probability
                    self.final_activation_prob = prob_at_least_one_transmission_log

        def reset(self):
            self.from_parent_prob.clear()
            self.from_neighbor_prob.clear()
            self.to_child_prob = None
            self.to_neighbor_prob = None
            self.children.clear()
            self.final_activation_prob = None

    def predict(self):

        nodes_ds = [self.Node(id, self.p) for id in self.G.nodes()]

        while self.k > 0:
            cur_layer = set() # nodes in the current level
            next_layer = set() # nodes in the next level
            last_layer = set()

            cur_layer.add(self.seeds[-1]) # add initial seed

            nodes_ds[self.seeds[-1]].from_parent_prob.append(np.log(1))
            nodes_ds[self.seeds[-1]].from_neighbor_prob.append(-np.inf) # log(0) is undefined
            nodes_ds[self.seeds[-1]].to_child_prob = np.log(self.p)
            nodes_ds[self.seeds[-1]].to_neighbor_prob = np.log(self.p)
            nodes_ds[self.seeds[-1]].activation_probs = [np.log(1)]
            nodes_ds[self.seeds[-1]].final_activation_prob = np.log(1)

            while cur_layer: # this runs until there is nothing left to check in the graph
                for node in cur_layer:

                    # get neighbors of the node
                    neighbors = self.G.neighbors(node)

                    # compute the probability of transmission to neighbors in the same layer relative to the seed
                    # this takes into account probabilities from parents only
                    nodes_ds[node].compute_to_neighbor_prob()

                    for nei in neighbors:
                        # check if the neighbor is actually among the parents
                        # of nodes in the current layer
                        if nei not in last_layer:
                            if nei in cur_layer:
                                # this is a neighbor in the same level
                                nodes_ds[nei].from_neighbor_prob.append(nodes_ds[node].to_neighbor_prob)
                            else:
                                # this is a child living in the next layer
                                next_layer.add(nei)
                                nodes_ds[node].children.append(nei) # we save the child's id for later

                # current layer is done, from_neighbor_prob is now filled for all nodes in the current layer
                for node in cur_layer:
                    # compute to_child probs for next layer
                    nodes_ds[node].compute_to_child_prob()

                    # set from_parent_prob for all children
                    for child in nodes_ds[node].children:
                        nodes_ds[child].from_parent_prob.append(nodes_ds[node].to_child_prob)            

                # we finished the current level
                # move on to the next level
                last_layer = cur_layer.copy()
                cur_layer = next_layer.copy()
                next_layer.clear()

            # we finished the BFS
            # compute the activation probability for each node
            for node in nodes_ds:
                node.compute_activation_prob()

            # sort nodes by activation probability
            nodes_ds.sort(key=lambda x: x.final_activation_prob, reverse=False)

            # get the single lowest k node
            self.seeds.append(nodes_ds[0].id)

            # restore original sort order
            nodes_ds.sort(key=lambda x: x.id, reverse=False)

            self.k -= 1

            # zero out the data structure except for activation_probs
            for node in nodes_ds:
                node.reset()

        # save full cache
        if self.use_cache:
            # save the seeds to the file
            self.save_cache()

        return self.seeds

class NaiveBFSMyopic(Algorithm):
    '''
    The initial seed is the node with the highest degree.
    Currently suffers from precision loss?
    '''

    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'naive_bfs_myopic'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(NaiveBFSMyopic, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    class Node():
        def __init__(self, id, p):
            self.id = id
            self.from_parent_prob = []
            self.from_neighbor_prob = []
            self.to_neighbor_prob = None
            self.to_child_prob = None
            self.children = []
            self.neighbor_targets = []
            self.p = p
            self.activation_prob = None

        def compute_to_neighbor_prob(self):
            # compute the probability of transmission to neighbors in the same layer relative to the seed
            # this takes into account probabilities from parents only

            # if the check fails, it's actually a seed node
            if self.to_neighbor_prob is None:
                 # probabilities for each event that it does not happen
                probs_log = np.log1p(-np.exp(self.from_parent_prob))

                # compute the log of the probability that none of the transmissions occur
                prob_no_transmission_log = np.sum(probs_log)

                # compute the log of the probability that at least one transmission occurs
                prob_at_least_one_transmission_log = np.log1p(-np.exp(prob_no_transmission_log))

                # set probability that at least one of the transmissions occurs
                self.to_neighbor_prob = prob_at_least_one_transmission_log + np.log(self.p)

        def compute_to_child_prob(self):
            # compute the probability of transmission to children in the next layer relative to the seed
            # this takes into account probabilities from parents and probabilities from neighbors

            # if this check fails, it's actually a seed node
            if self.to_child_prob is None:
                probs_log = np.concatenate((self.from_parent_prob, self.from_neighbor_prob))

                # compute the probability for events not to happen
                probs_log = np.log1p(-np.exp(probs_log))

                # compute the log of the probability that none of the transmissions occur
                prob_no_transmission_log = np.sum(probs_log)

                # compute the probability that at least one transmission occurs
                prob_at_least_one_transmission_log = np.log1p(-np.exp(prob_no_transmission_log))

                self.activation_prob = prob_at_least_one_transmission_log
                self.to_child_prob = prob_at_least_one_transmission_log + np.log(self.p)

 
    def predict(self):

        if self.k > 0:
            depth = 1

            nodes_ds = [self.Node(id, self.p) for id in self.G.nodes()]

            cur_layer = set() # nodes in the current level

            next_layer = set() # nodes in the next level

            last_layer = set()

            cur_layer.add(self.seeds[-1]) # add initial seed

            nodes_ds[self.seeds[-1]].from_parent_prob.append(np.log(1))
            nodes_ds[self.seeds[-1]].from_neighbor_prob.append(-np.inf)
            nodes_ds[self.seeds[-1]].to_child_prob = np.log(self.p)
            nodes_ds[self.seeds[-1]].to_neighbor_prob = np.log(self.p)
            nodes_ds[self.seeds[-1]].activation_prob = np.log(1)

            while cur_layer: # this runs until there is nothing left to check in the graph
                for node in cur_layer:

                    # get neighbors of the node
                    neighbors = self.G.neighbors(node)

                    # compute the probability of transmission to neighbors in the same layer relative to the seed
                    # this takes into account probabilities from parents only
                    nodes_ds[node].compute_to_neighbor_prob()

                    for nei in neighbors:
                        # check if the neighbor is actually among the parents
                        # of nodes in the current layer
                        if nei not in last_layer:
                            if nei in cur_layer:
                                # this is a neighbor in the same level
                                nodes_ds[nei].from_neighbor_prob.append(nodes_ds[node].to_neighbor_prob)
                                #nodes_ds[node].neighbor_targets.append(nei)
                            else:
                                # this is a child living in the next layer
                                next_layer.add(nei)
                                nodes_ds[node].children.append(nei) # we save the child's id for later

                    
                # current layer is done, from_neighbor_prob is now filled for all nodes in the current layer
                for node in cur_layer:
                    # compute to_child probs for next layer
                    nodes_ds[node].compute_to_child_prob()

                    # set from_parent_prob for all children
                    for child in nodes_ds[node].children:
                        nodes_ds[child].from_parent_prob.append(nodes_ds[node].to_child_prob)            


                # we finished the current level
                # move on to the next level
                last_layer = cur_layer.copy()
                cur_layer = next_layer.copy()
                next_layer.clear()

                depth += 1

            # for node in nodes_ds:
            #     print(node.id, node.activation_prob, node.from_parent_prob, node.from_neighbor_prob, node.to_neighbor_prob, node.to_child_prob)


            # sort nodes by activation probability
            nodes_ds.sort(key=lambda x: x.activation_prob, reverse=False)

            # get the lowest k nodes
            self.seeds.extend([node.id for node in nodes_ds[:self.k]])

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

class PPRMyopic(Algorithm):
    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'ppr_myopic'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(PPRMyopic, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def predict(self):
        # initial attempt
        if self.k > 0:
            for _ in range(self.k):
                # compute personalized page rank with machine precision tolerance
                ppr = nx.pagerank(self.G, alpha=0.3, tol=1e-16, personalization={node: 1 for node in self.seeds}, max_iter=1000)

                # sort nodes by activation probability
                sorted_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=False)

                # pick the lowest node
                self.seeds.append(sorted_ppr[0][0])

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

class NaivePPRMyopic(Algorithm):
    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'naive_ppr_myopic'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(NaivePPRMyopic, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def predict(self):
        # initial attempt
        if self.k > 0:
            # compute personalized page rank with machine precision tolerance
            ppr = nx.pagerank(self.G, alpha=0.3, tol=1e-16, personalization={node: 1 for node in self.seeds}, max_iter=1000)

            # sort nodes by activation probability
            sorted_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=False)

            # get the lowest k nodes
            self.seeds.extend([node[0] for node in sorted_ppr[:self.k]])

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

class DegreeLowestCentrality(Algorithm):
    '''
        Initially seeded with the highest degree node
    '''
    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'degree_lowest_centrality'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(DegreeLowestCentrality, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def initialize_seeds(self):
        # algorithm does not need an initial seed
        self.seeds = []

    def predict(self):
        if self.k > 0: # if we need to predict more seeds
            # bin all nodes by degree
            # degree 1
            # degree 2
            # degree 3
            # ...

            # get highest degree in the graph
            max_degree = np.max(np.array(self.G.degree())[:, 1])

            time_start = time.time()
            centrality = nx.harmonic_centrality(self.G) # old approach
            self.precompute_time += time.time() - time_start

            for k in range(1, max_degree+1):
                # pick all nodes in the network with degree k
                degree_k = [node for node in self.G.nodes() if self.G.degree(node) == k]

                # compute closeness for each node in degree_k
                centrality_k = [centrality[node] for node in degree_k]

                # make an np array with two columns: degree_k and closeness_k
                temp = np.array([degree_k, centrality_k]).T

                # sort by second column in ascending order
                # lowest closeness is better
                temp = temp[temp[:, 1].argsort()]

                # read back out with new sorting
                degree_k = temp[:, 0].astype(int)

                choices = degree_k[:self.k]
                self.seeds.extend(choices)
                self.k -= len(choices)
                
                if self.k <= 0:
                    break

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds
        
class DegreeLowestCentralityChooseNeighbor(Algorithm):
    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'degree_lowest_centrality_choose_neighbor'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(DegreeLowestCentralityChooseNeighbor, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def initialize_seeds(self):
        # algorithm does not need an initial seed
        self.seeds = []

    def predict(self):
        if self.k > 0: # if we need to predict more seeds
            # bin all nodes by degree
            # degree 1
            # degree 2
            # degree 3
            # ...

            # get highest degree in the graph
            max_degree = np.max(np.array(self.G.degree())[:, 1])

            time_start = time.time()
            centrality = nx.harmonic_centrality(self.G)
            self.precompute_time += time.time() - time_start

            for k in range(1, max_degree+1):
                # pick all nodes in the network with degree k
                degree_k = [node for node in self.G.nodes() if self.G.degree(node) == k]

                # compute centrality for each node in degree_k
                centrality_k = [centrality[node] for node in degree_k]

                # make an np array with two columns: degree_k and closeness_k
                temp = np.array([degree_k, centrality_k]).T

                # sort by second column in ascending order
                # lowest closeness is better
                temp = temp[temp[:, 1].argsort()]

                # read back out with new sorting
                degree_k = temp[:, 0].astype(int)

                choices = degree_k[:self.k]

                # for each choice, pick the neighbor with the highest degree that isn't in the seed set
                for choice in choices:
                    neighbors = list(set(self.G.neighbors(choice)) - set(self.seeds))

                    if len(neighbors) > 0:
                        # get the node with the highest degree among the neighbors
                        # otherwise just pick the node we found earlier
                        choice = max(neighbors, key=self.G.degree)

                    self.seeds.append(choice)
                    self.k -= 1

                if self.k <= 0:
                    break

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds


class DegreeHighestDegreeNeighbor(Algorithm):
    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'degree_highest_degree_neighbor'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(DegreeHighestDegreeNeighbor, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def initialize_seeds(self):
        # Greedy algorithm does not need an initial seed
        self.seeds = []

    def predict(self):
        if self.k > 0: # if we need to predict more seeds
            # bin all nodes by degree
            # degree 1
            # degree 2
            # degree 3
            # ...

            # get highest degree in the graph
            max_degree = np.max(np.array(self.G.degree())[:, 1])

            for k in range(1, max_degree+1):
                # pick all nodes in the network with degree k
                degree_k = [node for node in self.G.nodes() if self.G.degree(node) == k]

                # for each node in degree_k, get the nighest neighbor that isn't in the seed set

                best_neighbors = []
                best_degrees = []

                for node in degree_k:
                    neighbors = list(set(self.G.neighbors(node)) - set(self.seeds))

                    if len(neighbors) > 0:
                        # get the node with the highest degree among the neighbors
                        # otherwise just pick the node we found earlier
                        best_neighbors.append(max(neighbors, key=self.G.degree))
                        best_degrees.append(self.G.degree(best_neighbors[-1]))
                    elif node not in self.seeds:
                        best_neighbors.append(node)
                        best_degrees.append(k)

                # make an np array with three columns: degree_k, best_neighbor, best_degree
                temp = np.array([degree_k, best_neighbors, best_degrees]).T

                # sort by third column in descending order
                # highest degree is better
                temp = temp[temp[:, 2].argsort()[::-1]]

                # read first column back out with new sorting
                degree_k = temp[:, 0].astype(int)

                choices = degree_k[:self.k]
                self.seeds.extend(choices)
                self.k -= len(choices)
                
                if self.k <= 0:
                    break

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

class DegreeHighestDegreeNeighborChooseNeighbor(Algorithm):
    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'degree_highest_degree_neighbor_choose_neighbor'
        self.cache_filename = "./cache/algo_cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(DegreeHighestDegreeNeighborChooseNeighbor, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def initialize_seeds(self):
        # algorithm does not need an initial seed
        self.seeds = []
    
    def predict(self):
        if self.k > 0: # if we need to predict more seeds
            # bin all nodes by degree
            # degree 1
            # degree 2
            # degree 3
            # ...

            # get highest degree in the graph
            max_degree = np.max(np.array(self.G.degree())[:, 1])

            for k in range(1, max_degree+1):
                # pick all nodes in the network with degree k
                degree_k = [node for node in self.G.nodes() if self.G.degree(node) == k and node not in self.seeds]

                # for each node in degree_k, get the highest neighbor that isn't in the seed set

                best_neighbors = []
                best_degrees = []
                to_remove = []

                for node in degree_k:
                    neighbors = list(set(self.G.neighbors(node)) - set(self.seeds))

                    if len(neighbors) > 0:
                        # get the node with the highest degree among the neighbors
                        # otherwise just pick the node we found earlier
                        best_neighbors.append(max(neighbors, key=self.G.degree))
                        best_degrees.append(self.G.degree(best_neighbors[-1]))
                    elif node not in self.seeds:
                        best_neighbors.append(node)
                        best_degrees.append(k)

                # remove nodes that are already in the seed set
                degree_k = list(set(degree_k) - set(to_remove))

                # make an np array with three columns: degree_k, best_neighbor, best_degree
                #print(len(degree_k))
                #print(len(best_neighbors))
                #print(len(best_degrees))
                temp = np.array([degree_k, best_neighbors, best_degrees]).T

                # sort by third column in descending order
                # highest degree is better
                temp = temp[temp[:, 2].argsort()[::-1]]

                # read second column back out with new sorting
                best_neighbors = temp[:, 1].astype(int)

                # it is possible that we have duplicates in the best_neighbors set
                # we need to remove them so as not to add a node twice
                # this might be a problem if we have a small k and a large number of duplicates
                # but we can ignore this for now

                # remove duplicates
                best_neighbors = list(set(best_neighbors))

                choices = best_neighbors[:self.k]
                self.seeds.extend(choices)
                self.k -= len(choices)
                
                if self.k <= 0:
                    break

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

##############################
# Incomplete / testing stuff #
##############################


class Test(Algorithm):
    def __init__(self, G, p=0.5, k=100, seeds=None, ic_trials=1000, use_cache=False, threads=0):
        self.algo_name = 'test'
        self.cache_filename = "./cache/{}/{}_{}.txt".format(self.algo_name, G.name, p)
        super(Test, self).__init__(G, p, k, seeds, ic_trials, use_cache, threads)

    def predict(self):
        #return self.predict_degree_by_centrality()
        return self.predict_degree_by_centrality_not_neighbor()
        #return self.predict_using_trees()

    def predict_degree_by_length(self):
        if self.k > 0: # if we need to predict more seeds
            # bin all nodes by degree
            # degree 1
            # degree 2
            # degree 3
            # ...

            sp = nx.shortest_path_length(self.G, source=self.seeds[-1])

            # get highest degree in the graph
            max_degree = np.max(np.array(self.G.degree())[:, 1])

            for k in range(1, max_degree+1):
                # pick all nodes in the network with degree k
                degree_k = [node for node in self.G.nodes() if self.G.degree(node) == k]

                lengths = []
                for node in degree_k:
                    lengths.append(sp[node])

                # make an np array with two columns: degree_k and lengths
                temp = np.array([degree_k, lengths]).T

                # sort by second column in reverse order
                temp = temp[temp[:, 1].argsort()[::-1]]

                # read back out with new sorting
                degree_k = temp[:, 0].astype(int)

                # shuffle the nodes with degree and add to seed set
                # np.random.shuffle(degree_k)
                choices = degree_k[:self.k]
                self.seeds.extend(choices)
                self.k -= len(choices)
                
                if self.k <= 0:
                    break


            #self.seeds.extend(seeds)

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds
    

    def predict_degree_by_centrality(self):
        if self.k > 0: # if we need to predict more seeds
            # bin all nodes by degree
            # degree 1
            # degree 2
            # degree 3
            # ...

            # get highest degree in the graph
            max_degree = np.max(np.array(self.G.degree())[:, 1])

            for k in range(1, max_degree+1):
                # pick all nodes in the network with degree k
                degree_k = [node for node in self.G.nodes() if self.G.degree(node) == k]

                # get a graph of non-seed nodes
                #G_non_seed = self.G.copy()
                #G_non_seed.remove_nodes_from(self.seeds)

                # compute closeness centrality for all non-seed nodes
                #closeness = nx.closeness_centrality(self.G)
                centrality = nx.harmonic_centrality(self.G)

                # compute closeness for each node in degree_k
                centrality_k = [centrality[node] for node in degree_k]

                # make an np array with two columns: degree_k and closeness_k
                temp = np.array([degree_k, centrality_k]).T

                # sort by second column in ascending order
                # lowest closeness is better
                temp = temp[temp[:, 1].argsort()]

                # read back out with new sorting
                degree_k = temp[:, 0].astype(int)

                # shuffle the nodes with degree and add to seed set
                # np.random.shuffle(degree_k)
                choices = degree_k[:self.k]
                self.seeds.extend(choices)
                self.k -= len(choices)
                
                if self.k <= 0:
                    break


            #self.seeds.extend(seeds)

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds
    

    def predict_degree_by_centrality_not_neighbor(self):
        if self.k > 0: # if we need to predict more seeds
            # bin all nodes by degree
            # degree 1
            # degree 2
            # degree 3
            # ...

            # get highest degree in the graph
            max_degree = np.max(np.array(self.G.degree())[:, 1])

            for k in range(1, max_degree+1):
                # pick all nodes in the network with degree k
                degree_k = [node for node in self.G.nodes() if self.G.degree(node) == k]

                # get a graph of non-seed nodes
                #G_non_seed = self.G.copy()
                #G_non_seed.remove_nodes_from(self.seeds)

                # compute closeness centrality for all non-seed nodes
                #closeness = nx.closeness_centrality(self.G)
                centrality = nx.harmonic_centrality(self.G)

                # compute closeness for each node in degree_k
                centrality_k = [centrality[node] for node in degree_k]

                # make an np array with two columns: degree_k and closeness_k
                temp = np.array([degree_k, centrality_k]).T

                # sort by second column in ascending order
                # lowest closeness is better
                temp = temp[temp[:, 1].argsort()]

                # read back out with new sorting
                degree_k = temp[:, 0].astype(int)

                ix = 0

                choices = []

                while self.k > 0:
                    # add node at index to the seed set
                    self.seeds.append(degree_k[ix])
                    self.k -= 1

                    # find next index: it can only be that of a node that is not a neighbor of
                    # anything in the seed set
                    ix_ = ix
                    while True:
                        ix_ += 1
                        # get neighbors of node at index
                        if ix_ >= len(degree_k):
                            break

                        neighbors = self.G.neighbors(degree_k[ix_])

                        # check if any of the neighbors are in the seed set
                        if not any([nei in self.seeds for nei in neighbors]):
                            break

                    if ix_ >= len(degree_k):
                        # we couldn't find any other seeds to meet the criteria
                        # so we just fill seed set with the rest of the nodes
                        # after the latest node we added

                        # get nodes starting from ix
                        #choices = degree_k[ix:]

                        # limit choices to first k
                        #choices = choices[:self.k]

                        break
                    else:
                        ix = ix_



                #choices = degree_k[:self.k]
                #choices_copy = choices.copy()


                # turn choices_copy into a list
                # for each node in choices, check if its neighbor is elsewhere in choices
                # for node in choices:
                #     neighbors = self.G.neighbors(node)
                #     for nei in neighbors:
                #         if nei in choices_copy:
                #             choices_copy.remove(nei)

                self.seeds.extend(choices)
                self.k -= len(choices)
                
                if self.k <= 0:
                    break


            #self.seeds.extend(seeds)

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds

    def predict_using_trees(self):
        # uses the idea from that one paper where they grow trees at random

        if self.k > 0:
            # we won't be using the default seed set used in other testing methods
            self.seeds = []
            self.k = 100

            # get 100 random nodes
            random_nodes = np.random.choice(self.G.nodes(), size=100, replace=False)

            # counter for each node in the graph
            # this will be used to keep track of how many times each node is seen in trees
            node_counter = {node: 0 for node in self.G.nodes()}

            # for each random node, grow a tree

            for node in random_nodes:
                to_visit = set()
                traversed = set()

                traversed.add(node)

                # add neighbors of node
                to_visit.update(self.G.neighbors(node))

                tree_size = 1

                # while there are still nodes to visit
                while to_visit and tree_size < 10:
                    # pick a random node from to_visit
                    random_node = np.random.choice(list(to_visit))

                    # add to traversed
                    traversed.add(random_node)

                    # remove from to_visit
                    to_visit.remove(random_node)

                    # add neighbors of random_node to to_visit
                    to_visit.update(self.G.neighbors(random_node))

                    tree_size += 1

                # increment counter for each node in the tree
                for node in traversed:
                    node_counter[node] += 1

            # sort nodes by counter in ascending order
            sorted_nodes = sorted(node_counter.items(), key=lambda x: x[1], reverse=False)

            # pick the lowest k nodes
            self.seeds.extend([node[0] for node in sorted_nodes[:self.k]])
            self.k = 0

            #self.seeds.extend(seeds)

            # save full cache
            if self.use_cache:
                # save the seeds to the file
                self.save_cache()

        return self.seeds
