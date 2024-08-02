import itertools
import algorithms as alg
import numpy as np
import probability as prob

def run_bruteforce(G, p, k):
    # bruteforces the optimal seed set for a given graph and p
    # this is silly slow, don't run for anything other than the SMALL synthetic graphs

    # get every possible seed set
    all_seed_sets = list(itertools.combinations(range(G.number_of_nodes()), k))

    # run myopic 20 times and average the results
    myopic_eval = []

    for i in range(20):
        myopic = alg.Myopic(G, k=k, p=p)
        myopic_eval.append(myopic.evaluate())

    # average last column
    myopic_eval = np.mean(myopic_eval, axis=0)

    to_beat = myopic_eval[-1]

    print("Myopic score to beat: ", to_beat)
    print("Seed sets to consider: ", len(all_seed_sets))

    # run bruteforce
    for i, seeds in enumerate(all_seed_sets):
        # print progress
        if i % 1000 == 0:
            print(f"{i}/{len(all_seed_sets)}")

        # get IC evaluation
        ic_eval = prob.estimate(G, p, seeds, 1000)
        
        # get the min probability
        eval = np.min(ic_eval)

        # if IC is better than myopic, print it
        if eval > to_beat:
            print("Found better seed set!")
            print(f"Seeds: {seeds}")
            print(f"IC eval: {eval}")

    #print(len(all_seed_sets))
