import independent_cascade as ic
from multiprocessing import Pool
import numpy as np
import networkx as nx
import ctypes

def estimate(G, p, seeds, ic_trials, threads=0):
    # prepare cpp arguments

    # Convert the graph to an adjacency matrix (1D array)
    A = nx.to_numpy_array(G, dtype=np.int32).flatten()
    n = G.number_of_nodes()

    prob_est_cpp = ctypes.CDLL('./cpp/prob_est')

    result = np.zeros(n, dtype=np.float32) # for storing the results of the cpp program

    # prepare array pointers
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    ss_ptr = np.array(seeds, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the C++ function with the 2D array
    prob_est_cpp.estimate(ctypes.c_int(threads), ctypes.c_float(p), ctypes.c_int(n), ctypes.c_int(ic_trials), ctypes.c_int(len(seeds)), A_ptr, ss_ptr, result_ptr)

    return result

# LEGACY CODE FROM BEFORE THE CPP REWRITE

def estimate_legacy(G, p, seeds, ic_trials):
    # initialize a cascade object
    cascade = ic.IndependentCascade(G=G, p=p, seeds=seeds)

    # prepare arguments for multiprocessing
    # randint rolls between 0 and 2147483647 here according to np docs
    args = zip([cascade for _ in range(ic_trials)], np.random.randint(
        0, np.iinfo(np.int32(10)).max, ic_trials))

    # run trials in parallel
    results = np.array(Pool().starmap(ic.IndependentCascade.run, args))

    # sum over columns and return
    return np.mean(results, axis=0, dtype=np.float64)

def estimate_single_thread(G, p, seeds, ic_trials):
    # this version of estimate runs on a single thread

    # initialize a cascade object
    cascade = ic.IndependentCascade(G=G, p=p, seeds=seeds)

    # run trials sequentially
    results = np.array([ic.IndependentCascade.run(cascade, np.random.randint(0, np.iinfo(np.int32(10)).max))
                        for _ in range(ic_trials)])

    # sum over columns and return
    return np.mean(results, axis=0, dtype=np.float64)
