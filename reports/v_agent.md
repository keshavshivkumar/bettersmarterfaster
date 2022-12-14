# V Model

## Overview

- The V model is a model that is built to predict the value of U*(s) for a state s ∊ state space S.

## Design

- The V model is a regression neural network model that is made of 1 input layer, 3 hidden layers and 1 output layer.
- The output layer outputs the predicted U* value for each state.
- Using the output of the V model, an agent can be constructed to attempt to catch the prey in the graph.

## Implementation

### V model

- The V model is trained using the U-table as input, which contains the calculated U* values for each state.
- The input is passed as a NumPy array of arrays. The features (columns), as specified before, are stored in each sub-array (rows).
- There are 6 features passed as input to the neural network: 
  - the agent's position node
  - the prey's position node
  - the predator's position node
  - the distance between the agent and prey positions
  - the distance between the prey and predator positions
  - the distance between the predator and agent positions.
- The weight and bias are numpy arrays initialized to random standard normalized values.
- The model is backpropagated using the concept of Automatic Differentiation (Suggested by Aravind Sivaramakrishnan):
  - Store values that are needed to compute gradient in each layer $\forall l ∈ [1,n]$.
  -  While backpropagating, compute the gradient of each layer $l$ using the stored value and derivative of previous layer $l+1$.
  - After gradients are calculated for each layer, update parameters of each layer with any optimizer of choice. 
  - In our case we have implemented a crude version of the Adam Optimizer
- There are 2 types of layers that are used:
    - Fully Connected Layer
    - Activation Layer
- The activation function used is Rectified Linear Unit (ReLU) and the loss function used is Mean Absolute Error (MAE).
- Different activation functions, loss functions and optimizers were trialed. 
- The layers are arranged in the following order: 
  - Input(6) -> FClayer(24) 
  - Activation (ReLU) 
  - FClayer(24) -> FClayer(24) 
  - Activation  (ReLU)
  - FClayer(24) -> Output(1)
  - Loss (MAE loss) 
- The V model is trained for 5,000 epochs.

### V Agent

- The V Agent bootstraps off the predictions made by the V model.
- Works the same as `U* Agent` but by replacing U* value with predicted U* value from V.

## Observations

- The loss of the training converged to ~ 0.61
- The agent performed with a winrate of 100%.

## References

- [Neural Network in 100 lines](https://eisenjulian.github.io/deep-learning-in-100-lines/index.html)