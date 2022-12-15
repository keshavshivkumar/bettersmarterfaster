# $V_{partial}$

## Overview

- Earlier, `Upartial` was estimated out of `Ustar` to provide us the estimated utility in a partial information environment (the prey's location is not known).
- A model, `Vpartial` was built to predict the $U_{partial}(s)$ value for state s.

## Design

- `Vpartial` is a neural network regression model in the same vein as `V_model`, with 1 input layer, 3 hidden layers and 1 output layer, and the output being the predicted $U_{partial}$ value.
- The difference is in the layer architecture.
- The output from the model can be used to construct the $V_{partial}$ agent.

## Implementation

### $V_{partial}$ model:
- The `Vpartial` model uses the same infrastructure as the V model, with the only difference in the layer architecture.
- `Vpartial` takes in 52 features as a NumPy array:
    - The agent position
    - The predator position
    - The 50 different prey beliefs
- While computing `Upartial`, the features are stored in `upartial.csv` along with the label, `upartial`.
- The final model we settled on: 
  - Input(52) -> FClayer(256) 
  - Activation (ReLU) 
  - FClayer(256) -> FClayer(256)
  - Activation  (ReLU)
  - FClayer(256) -> Output(1)
  - Loss (MAE loss) 
- The `Vpartial` model is trained over 5000 epochs.
- Overfitting becomes an issue here because there are infinite possible states due to the range of prey beliefs.
- To avoid overfitting, the training should be stopped when the training and testing loss starts to vary too strongly.
- On plotting the graphs of training and testing losses, the testing loss correlated to the training loss, implying the model did not overfit.
![Agent1t0](../comparisons/training_testing_loss.png)

### $V_{partial}$ Agent:

- Similar to the `V_agent`, the `Vpartial_agent` bootstraps off the predictions of $U_{partial}$ made by the `Vpartial` model.
- The `V_agent` uses the predicted $U_{partial}$ value from `Vpartial` to make a move.

## Observations

- MAE error after 5000 epochs: ~ 0.37
- Win rate of the agent on average: 99.8%
- The average # of timesteps was 35.648.
- The agent may lose due to timeout.

## Grey Box Questions

- Performance against the `Upartial` agent:
    - Both perform in the same way; both win rates are always between 98-100%.