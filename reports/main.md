# Better, Smarter, Faster
> CS-520 (Introduction to Artificial Intelligence) Project 3  
> > By: Keshav S. (ks1830), Shashwath S. (sks272)

[Github repo](https://github.com/keshavshivkumar/bettersmarterfaster)

## Overview

- Better, smarter, faster is the third project under the Rutgers' CS-520 class taught by Dr. Charles Cowan.
- The project aims to build on top of the previous project by making the agent catch the prey in the least number of tries while avoiding the predator.

## Grey box questions

1. If we consider all the states possible, number of states = 50 \* 50 \* 50 = 125000.

    If we consider only possible starting states, number of states = 50\*1\*49 + 50\*49\*48 = 120050

2. $U^*$ is easy to determine for states:
   - Prey and Agent in same node, $U^* = 0$
   - Predator and Agent in same node, $U^* \approx \infty$
   - Agent is one step away from prey (when prey and predator are not in the same node), $U^* = 1$ since the agent is moving optimally.



