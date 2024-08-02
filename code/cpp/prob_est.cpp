

#include <iostream>

#include <queue> // for queue

#include <random>

#include <omp.h>

using namespace std;

// a faster implementation of prob_est

// using cpp and openmp

class RandomGenerator
{
public:
    RandomGenerator() : gen(static_cast<unsigned>(time(0))), dist(0.0f, 1.0f) {}

    float get()
    {
        return dist(gen);
    }

    void seed(int seed)
    {
        gen.seed(time(0) + seed);
    }

private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

void ic(RandomGenerator *gen, bool **adj, int *ss, int *activated_, int n_, int len_ss, float p)
{
    // queue of nodes we are spreading from
    queue<int> q;

    // initialize queue to the seed set
    for (int i = 0; i < len_ss; i++)
    {
        q.push(ss[i]);
        activated_[ss[i]] = true;
    }

    // while queue isn't empty

    while (!q.empty())
    {
        // get the first element the pop
        int node = q.front();

        q.pop();

        for (int i = 0; i < n_; i++)
            if (adj[node][i] && !activated_[i] && gen->get() < p)
            {
                // check if activation occurs
                // if the edge is activated
                activated_[i] = 1;
                q.push(i);
            }
    }
}

extern "C"
{
    void estimate(int threads, float p_, int n_, int iters_, int len_ss_, int *adj1d_, int *ss_, float *result_)
    {
        int **activated_global = new int *[iters_];
        bool **adj = new bool *[n_];

        // convert flat adj1d to adj
        for (int i = 0; i < n_; i++)
        {
            adj[i] = new bool[n_];
            for (int j = 0; j < n_; j++)
                adj[i][j] = adj1d_[i * n_ + j];
        }

#pragma omp parallel for num_threads(threads)
        for (int i = 0; i < iters_; i++)
        {
            // make a local random number generator and seed it
            RandomGenerator gen;
            gen.seed(i);

            // initialize activations matrix
            int *activated = new int[n_];

            for (int j = 0; j < n_; j++)
                activated[j] = 0;

            // run ic
            ic(&gen, adj, ss_, activated, n_, len_ss_, p_);

            // add to global activations matrix
            activated_global[i] = activated;
        }

        // compute results and save to result_
        for (int i = 0; i < iters_; i++)
            for (int j = 0; j < n_; j++)
                result_[j] += activated_global[i][j];

        for (int i = 0; i < n_; i++)
            result_[i] /= iters_;

        // free memory
        for (int i = 0; i < n_; i++)
            delete[] adj[i];

        delete[] adj;

        for (int i = 0; i < iters_; i++)
            delete[] activated_global[i];

        delete[] activated_global;
    }
}