'''
V
'''

from env import Graph, Node
from nn_base import Layer, relu, relu_prime, mse, mse_prime
from graph_utils import bfs
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
                    out=layer.forward_propagate(out)
                error+=self.loss(y_train[node], out)
                error_prime=self.loss_prime(y_train[node], out)

                for layer in reversed(self.layers):
                    error_prime=layer.back_propagate(error_prime, l_rate)
            error/=len(x_train)
            print(f'Epoch {iter+1}, Elapsed Time: {time.time()-start}, Error: {error}')

    def predict(self, input):
        for i in range(len(input)):
            out=input[i]
            for layer in self.layers:
                out=layer.forward_propagate(out)
            self.result.append(out)
        return self.result

def main():
    with open('./done/qtable_graph_1.pkl', 'rb') as f:
        q = pk.load(f)
    with open('./done/graph_1.pkl', 'rb') as f:
        g = pk.load(f)
    
    X = [list(x) for x in q if x[0] != x[2] and x[0] != x[1]]
    y = [[[min(q[tuple(x)])]] for x in X]
    dist = dict()
    for i in X:
        if (i[0], i[1]) not in dist:
            dist[(i[0], i[1])] = len(bfs(g.graph_nodes[i[0]], g.graph_nodes[i[1]]))
        if (i[0], i[2]) not in dist:
            dist[(i[0], i[2])] = len(bfs(g.graph_nodes[i[0]], g.graph_nodes[i[2]]))
        if (i[1], i[2]) not in dist:
            dist[(i[1], i[2])] = len(bfs(g.graph_nodes[i[1]], g.graph_nodes[i[2]]))
        i.append(dist[(i[0], i[1])])
        i.append(dist[(i[0], i[2])])
        i.append(dist[(i[1], i[2])])
    X = [[x] for x in X]
    X=np.array(X, dtype=float)
    y=np.array(y, dtype=float)
    # print(f'X dims: {X.shape}, y dims: {y.shape}')
    
    v_model=V(epochs=100)
    v_model.add_layer(Layer(0, input_len=6, output_len=12))
    v_model.add_layer(Layer(1, activation=relu, activation_prime=relu_prime))
    v_model.add_layer(Layer(0, input_len=12, output_len=24))
    v_model.add_layer(Layer(1, activation=relu, activation_prime=relu_prime))
    v_model.add_layer(Layer(0, input_len=24, output_len=48))
    v_model.add_layer(Layer(1, activation=relu, activation_prime=relu_prime))
    v_model.add_layer(Layer(0, input_len=48, output_len=24))
    v_model.add_layer(Layer(1, activation=relu, activation_prime=relu_prime))
    v_model.add_layer(Layer(0, input_len=24, output_len=12))
    v_model.add_layer(Layer(1, activation=relu, activation_prime=relu_prime))
    v_model.add_layer(Layer(0, input_len=12, output_len=6))
    v_model.add_layer(Layer(1, activation=relu, activation_prime=relu_prime))
    v_model.add_layer(Layer(0, input_len=6, output_len=1))
    v_model.add_layer(Layer(1, activation=relu, activation_prime=relu_prime))

    v_model.loss_function(mse, mse_prime)
    v_model.train(X, y, l_rate=0.0001)
    print(len(X))
    # pk.dump(v_model, 'v.pkl', 'wb')

if __name__=='__main__':
    main()