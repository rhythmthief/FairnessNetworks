# FairnessNetworks
## Abstract
When information spreads across a network via pairwise sharing, large disparities in information access can arise from the network's structural heterogeneity. Algorithms to improve the fairness of information access seek to minimize the access gap by sequentially selecting new nodes to seed with the spreading information. However, existing algorithms are computationally expensive, which limits their scalability. Here, we develop and evaluate a set of 10 new scalable algorithms to improve information access in social networks, compare them to the existing state-of-the-art, introduce both a new metric for evaluating and comparing algorithm performance and a new benchmark data set of networks, and investigate the degree to which algorithm performance on minimizing information access gaps can be predicted ahead of time from features of a network's structure. We find that while no algorithm is strictly superior to all others across networks, our new scalable algorithms are competitive with the state-of-the-art algorithms and orders of magnitude faster. We introduce an ensemble approach that combines all of the best new algorithms to achieve results close to the state-of-the-art while being 9 times faster.

## Network Corpus
The most recent network corpus used in the study is `./datasets/corpus_augmented.pkl`. This file is a `pandas` dataframe, network citations are provided in the file itself.

## Codebase
### Data pre-processing
The codebase is designed to work with a pre-processed network corpus. The code for pre-processing the networks for inclusion in the corpus can be found in `./code/runners.py`, mainly the method `run_augment_corpus()`.

### Algorithm Implementations
All algorithm implementations are found in `./code/algorithms.py`.
The codebase includes implementations of the following algorithms from prior literature:
- Random
- Greedy
- Myopic
- Naive Myopic
- Gonzales

Additionally, the codebase includes the following novel algorithms:
- Myopic BFS
- Naive Myopic BFS
- Myopic PPR
- Naive Myopic PPR
- MinDegree_hc
- MinDegree_hcn
- MinDegree_nd
- MinDegree_ndn
- LeastCentral
- LeastCentral_n

### Algorithm Performance Evaluations
To produce algorithm performance evaluations on a given network, run `python main.py corpus_multi [index]`, where index is an integer in [0, 174]. This will also compute independent cascade parameters for three select spreadabilities, and additionally evaluate under several select preset independent cascade parameters. The output files are stored in `./cache/evaluations/`.

### Hyperparameter Tuning
Our hyperparameter tuning strategy is included as commented-out code in `./code/runners_figs.py`, lines 2340-2360. The results of this search step were originally cached and analyzed later. Our final selection of hyperparameters reflects a choice of hyperparameters that deliver the highest prediction accuracy on average across the network corpus used in this study, and can be found in `./code/runners_figs/`, line 2362.


## Repository Overview

```
.
├── code
│   ├── cache
│   │   ├── algo_cache // various algorithm seed set caches
│   │   │   └── ...
│   │   ├── evaluations // evaluation results for corpus networks
│   │   │   └── ...
│   │   ├── features.npz // corpus network features
│   │   ├── times_apsp.npz // apsp times per-network
│   │   ├── timing_algos // algorithm timing measurements
│   │   │   └── ...
│   │   └── timing_probest
│   │       └── [various ProbEst timing experiments]
│   ├── cpp // fast implementation of ProbEst
│   │   ├── Makefile
│   │   ├── prob_est
│   │   └── prob_est.cpp
│   ├── algorithms.py // algorithm implementations
│   ├── bruteforce.py // ideal combinatoric bruteforce algorithm
│   ├── experiments.py // experimental setups
│   ├── independent_cascade.py // independent cascade helper code for slow implementation of ProbEst
│   ├── main.py // main executable
│   ├── networks.py // various synthetic and corpus networks
│   ├── probability.py // ProbEst implementations
│   ├── runners_figs.py // figure plotting code
│   ├── runners.py // code for running experiments, augmenting network corpus, etc.
│   ├── spreadability.py // spreadability comptutation code
│   └── timing_runner.sh // bash scheduler for timing experiments
├── datasets // various data sets used in the study
│   ├── corpus_augmented.pkl // most recent corpus
│   └── ... // other loose files are a part of the corpus compilation process
└── README.md

```
