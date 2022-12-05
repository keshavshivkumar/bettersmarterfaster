'''
neural network classes
'''

# from abc import abstractmethod
import numpy as np


class Layer:
    def __init__(self, FC_or_Act, input_len=None, output_len=None, activation=None, activation_prime=None) -> None:
        self.input = None
        self.output = None
        self.input_len = input_len
        self.output_len = output_len
        self.weights = np.random.rand(self.input_len, self.output_len)
        self.bias = np.random.rand(self.output_len)
        self.activation = activation
        self.activation_prime = activation_prime
        self.layer_bool = FC_or_Act

    def weight_bias_initialize(self):
        self.weights = np.random.rand(self.input_len, self.output_len)
        self.bias = np.random.rand(self.output_len,)

    def forward_propogate(self, input_data):
        self.input = input_data
        if not self.layer_bool: # Fully connected Layer
            self.output = np.dot(self.input, self.weights) + self.bias
        else:                   # Activation Layer
            self.output = self.activation(self.input)
        return self.output

    def back_propogate(self, loss, l_rate):
        if not self.layer_bool:
            error = np.dot(loss, self.weights.T)
            weight_loss = np.dot(self.input.T, loss)
            self.weights -= l_rate * weight_loss
            self.bias -= l_rate * loss
        else:
            error = self.activation_prime(self.input) * loss
        return error


# class FCLayer(Layer):
#     def __init__(self, weights, bias) -> None:
#         super().__init__()
#         self.weights = weights
#         self.bias = bias
    
#     def forward_propogate(self, input):
#         self.input=input
#         self.output=np.dot(self.input,self.weights)+self.bias
#         return self.output

#     def backward_propogate(self, loss, l_rate):
#         return super().backward_propogate(loss, l_rate)


# class ActivationLayer(Layer):
#     def __init__(self, activation_func, activation_derivative_func) -> None:
#         super().__init__()
#         self.activation=activation_func
#         self.activation_derivative=activation_derivative_func

#     def forward_propogate(self, input):
#         self.input=input
#         self.output=self.activation(self.input)
#         return self.output

#     def backward_propogate(self, loss, l_rate):
#         return super().backward_propogate(loss, l_rate)


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
    return max(0,x)

def relu_prime(x):
    return 1 * (x>0)

'''
Loss Functions
'''

def mse(expected, observed):
    return np.square(expected-observed).mean()

def mse_prime(expected, observed):
    return 2 * (observed-expected)/expected.size