import numpy as np
from collections import deque  # efficient queue implementation


class IndependentCascade:
    def __init__(self, G, p, seeds):
        self.seeds = seeds
        self.G = G
        self.p = p

    def run(self, random_state):
        # set random state, this is important for multiprocessing
        # otherwise, all processes will use the same random state
        rng = np.random.default_rng(random_state)

        n = self.G.number_of_nodes()
        activated = np.zeros(n)  # initialize a vector of activated nodes

        # initialize a queue for the activated nodes
        q = deque()

        # insert seeds from seed_list into the queue
        for s in self.seeds:
            q.append(s)
            activated[s] = 1

        # main loop
        while q:
            node = q.popleft()  # get a node from the queue
            # get neighbors of the active node
            neighbors = list(self.G.neighbors(node))

            # leveraging numpy's speedy vectorized operations
            roll_vector = rng.random(len(neighbors))
            to_activate = np.logical_and(roll_vector < self.p, activated[neighbors] == 0)
            new_activated = np.array(neighbors)[to_activate]
            
            # Add activated neighbors to the queue and set them as activated
            q.extend(new_activated)
            activated[new_activated] = 1

        return activated
