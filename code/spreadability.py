import probability as prob
import numpy as np
import networks

def search(G, num_seeds):
    '''
    Scan values of p to find ones that produce low, medium, and high spreadability.
    '''

    LOW = 0.2
    MED = 0.5
    HIGH = 0.8

    p_vals_dict = {}

    # sample seeds at random
    seeds = np.random.choice(G.nodes, size=num_seeds, replace=True)

    spreadability = []
    p_vals = list(np.arange(0.01, 1, 0.01))

    for p in p_vals:
        num_activated = []

        for s in seeds:
            test = prob.estimate(G, p, [s], 1)

            num_activated.append(np.sum(test))
        
        spreadability.append(np.mean(num_activated) / G.number_of_nodes())

    # linearly search the values of p for the desired spreadabilities

    p_low = 0
    p_med = 0
    p_high = 0

    for i in range(len(spreadability)):
        if spreadability[i] < LOW:
            p_low = p_vals[i]

        if spreadability[i] < MED:
            p_med = p_vals[i]

        if spreadability[i] < HIGH:
            p_high = p_vals[i]


    # fallback to the first, middle, and last values of p if the search fails
    if p_low == 0:
        p_low = p_vals[0]

    if p_med == 0:
        p_med = p_vals[len(p_vals) // 2]

    if p_high == 0 or p_high == p_vals[-1]:
        p_high = p_vals[-1]
    else:
        # earlier, we found a value that's the closest to 0.8 but smaller
        # so we need to find the next value that's bigger than 0.8
        p_high = p_vals[p_vals.index(p_high) + 1]
    

    p_vals_dict['low'] = p_low
    p_vals_dict['med'] = p_med
    p_vals_dict['high'] = p_high

    return p_vals_dict