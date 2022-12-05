'''
V
'''

from nn_base import Layer
import time

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
        for iter in range(self.epochs):
            start = time.time()
            accuracy=0
            error = 0
            for node in range(len(x_train)):
                out = x_train[node]
                for layer in self.layers:
                    out=layer.forward_propogate(out)
                if y_train[node] == out:
                    accuracy+=1
                error+=self.loss(y_train[node], out)
                error_prime=self.loss_prime(y_train[node], out)
                rev_layers=self.layers.reverse()
                for layer in rev_layers:
                    error_prime=layer.back_propogate(error_prime, l_rate)
            error/=len(x_train)
            accuracy=accuracy / len(y_train) * 100
            print(f'Epoch {iter}, Elapsed Time: {time.time()-start}, Error: {error}, Accuracy: {accuracy}')

    def predict(self, input):
        for i in range(len(input)):
            out=input[i]
            for layer in self.layers:
                out=layer.forward_propogate(out)
            self.result.append(out)
        return self.result

def main():
    X=None
    y=None
    v_model=V()