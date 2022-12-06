'''
neural network classes
'''

import numpy as np


class Layer:
    def __init__(self, FC_or_Act, input_len=None, output_len=None, activation=None, activation_prime=None) -> None:
        self.input = None
        self.output = None
        self.input_len = input_len
        self.output_len = output_len
        if input_len != None:
            self.weights = np.random.rand(self.input_len, self.output_len) -0.5
            self.bias = np.random.rand(1, self.output_len) -0.5
        self.activation = activation
        self.activation_prime = activation_prime
        self.layer_bool = FC_or_Act

    def forward_propagate(self, input_data):
        self.input = input_data
        if not self.layer_bool: # Fully connected Layer
            self.output = np.dot(self.input, self.weights) + self.bias
        else:                   # Activation Layer  
            self.output = self.activation(self.input)
        return self.output

    def back_propagate(self, loss, l_rate):
        if not self.layer_bool:
            error = np.dot(loss, self.weights.T)
            # print(f'loss: {loss}, weight: {self.weights}, \n')
            weight_loss = np.dot(self.input.T, loss)
            self.weights -= l_rate * weight_loss
            self.bias -= l_rate * loss
        else:
            # print(f'loss: {loss}, input: {self.input}, \n')
            error = self.activation_prime(self.input) * loss
        return error

'''
Activation Functions
'''

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_prime(x):
    return 1-(tanh(x)**2)

def relu(x):
    return x * (x >= 0) + x * 0.1 * (x < 0)

def relu_prime(x):
    return 1 * (x >= 0) + 0.1 * (x < 0)

'''
Loss Functions
'''

def mse(expected, observed):
    return np.square(expected-observed).mean()

def mse_prime(expected, observed):
    return 2 * (observed-expected)/expected.size