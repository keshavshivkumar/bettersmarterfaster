# $U_{partial}$#

## Overview

- $U_{partial}$# or `UpartialSharp` is a variant of the estimated utility.
- Instead of estimating the utility off of $U^{*}$, the utility is estimated using the predictions of the `V_model`.

## Design

- The `UpartialSharp` works very similarly to `Upartial`.
- The estimated utility is calculated using the predictions of `V`, rather than $U^{*}$.
- $$U_{partial}(s_{agent}, s_{predator}, \underbar{p}) = \sum_{s_{prey}} p_{s_{prey}} V(s_{agent}, s_{predator}, s_{prey}, dist(s_{agent}, s_{prey}), dist(s_{prey}, s_{predator}), dist(s_{predator}, s_{agent}))$$ 
- Using the above equation, `UpartialSharp` is calculated.

## Implementation

- The implementation does not deviate from `Upartial`.

## Observations

- The win rate of the `UpartialSharp` Agent is around 99.8%.
- The average # of timesteps taken was 27.3775.
- There is no significant difference between the performance of `Upartial` and `UpartialSharp`.