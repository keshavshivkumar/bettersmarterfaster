# U*

## Overview

- The utility, `Ustar` is defined to be the minimal expected number of rounds for the agent to catch the prey.
- With Ustar, an optimal policy can be recovered to ensure the agent can succeed in catching the prey.

## Design

- The U* for any state can be calculated using value iteration: 
    - Bellman Equation -> $U^*(s) = max_{a∈A(s)}[r_{s, a} + \beta \sum_{a∈A(s)} p_{s,s'}^a U^*(s') ]$
    - Value Iteration -> $U^*_{k+1}(s)$ = $max_{a∈A(s)}$ $[ \sum_{a∈A(s)} p_{s,s'}^a U^*_k(s') ]$

## Implementation

- The initial U(s) ∀ s ∈ state space S is set to a constant, 30.
- The U(s'), where s' implies the states where both the agent and predator exist, is set to 1,000,000. The U* for such a state is always set to 1,000,000 whenever encountered.
- The U(s''), where s'' implies the states where both the agent and prey exist, is set to 0.
- A prey transition matrix is stored to track the probabilities of where the prey may move to.
- Similarly, the predator propogated belief is tracked as a dictionary.
- The U* values converge over time using value iteration until the optimal U* values for each state are obtained. 