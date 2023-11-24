#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double damping_factor = damping / numNodes;
  double equal_prob_damping = (1.0 - damping) / numNodes;
  double add_to_score_new = equal_prob_damping;

  double *score_old = new double[numNodes];
  double *score_new = new double[numNodes];
  int *no_outgoing = new int[numNodes];

  int no_outgoing_nums = 0;

  // Find nodes with no outgoing edges
  for (int i = 0; i < numNodes; ++i)
  {
    if (outgoing_size(g, i) == 0)
    {
      no_outgoing[no_outgoing_nums++] = i;
    }
  }

// Initialize scores
#pragma omp parallel for
  for (int i = 0; i < numNodes; i++)
  {
    score_old[i] = equal_prob;
  }

  bool converged = false;
  while (!converged)
  {
    double no_outgoing_sum = 0.0;
#pragma omp parallel for reduction(+ : no_outgoing_sum)
    for (int i = 0; i < no_outgoing_nums; i++)
    {
      no_outgoing_sum += score_old[no_outgoing[i]];
    }

    no_outgoing_sum *= damping_factor;
    add_to_score_new = equal_prob_damping + no_outgoing_sum;

    // calculate score_new[i] for all nodes i
#pragma omp parallel for
    for (int i = 0; i < numNodes; i++)
    {
      double sum = 0.0;

      // sum up score_old[j] for all in-edges j -> i
      const Vertex *start = incoming_begin(g, i);
      const Vertex *end = incoming_end(g, i);
      for (const Vertex *j = start; j != end; ++j)
      {
        sum += score_old[*j] / outgoing_size(g, *j);
      }

      score_new[i] = (damping * sum) + add_to_score_new;
    }

    // check convergence
    double global_diff = 0.0;
#pragma omp parallel for reduction(+ : global_diff)
    for (int i = 0; i < numNodes; ++i)
    {
      global_diff += abs(score_new[i] - score_old[i]);
    }

    converged = (global_diff < convergence);

    std::swap(score_old, score_new);
  }

#pragma omp parallel for
  for (int i = 0; i < numNodes; i++)
  {
    solution[i] = score_old[i];
  }

  delete[] score_new;
  delete[] score_old;
}
