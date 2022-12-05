'''
V
'''

from nn_base import Layer, tanh, tanh_prime, mse, mse_prime
import time
import numpy as np
import pickle as pk

class V:
    def __init__(self, epochs) -> None:
        self.layers=[]
        self.epochs=epochs
        self.result=[]

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def loss_function(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime=loss_prime

    def train(self, x_train, y_train, l_rate):
        start = time.time()
        for iter in range(self.epochs):
            error = 0
            for node in range(len(x_train)):
                out = x_train[node]
                for layer in self.layers:
                    out=layer.forward_propogate(out)
                error+=self.loss(y_train[node], out)
                error_prime=self.loss_prime(y_train[node], out)
                rev_layers=self.layers[::-1]
                for layer in rev_layers:
                    error_prime=layer.back_propogate(error_prime, l_rate)
            error/=len(x_train)
            print(f'Epoch {iter}, Elapsed Time: {time.time()-start}, Error: {error}')

    def predict(self, input):
        for i in range(len(input)):
            out=input[i]
            for layer in self.layers:
                out=layer.forward_propogate(out)
            self.result.append(out)
        return self.result

def main():
    # X=None
    # y=None

    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    v_model=V(epochs=1000)
    v_model.add_layer(Layer(0, input_len=2, output_len=3))
    v_model.add_layer(Layer(1, activation=tanh, activation_prime=tanh_prime))
    v_model.add_layer(Layer(0, input_len=3, output_len=1))
    v_model.add_layer(Layer(1, activation=tanh, activation_prime=tanh_prime))

    v_model.loss_function(mse, mse_prime)
    v_model.train(x_train, y_train, l_rate=0.1)

    pk.dump(v_model, 'v.pkl', 'wb')

if __name__=='__main__':
    main()