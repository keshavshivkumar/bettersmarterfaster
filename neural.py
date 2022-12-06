import numpy as np
import matplotlib.pyplot as plt
from graph_utils import bfs
import pickle as pk
from env import Graph, Node

class Parameter():
  def __init__(self, tensor):
    self.tensor = tensor
    self.gradient = np.zeros_like(self.tensor)

class Layer:
  def __init__(self):
    self.parameters = []

  def forward(self, X):
    return X, lambda D: D

  def build_param(self, tensor):
    param = Parameter(tensor)
    self.parameters.append(param)
    return param

  def update(self, optimizer):
    for param in self.parameters: optimizer.update(param)

class Linear(Layer):
  def __init__(self, inputs, outputs):
    super().__init__()
    self.weights = self.build_param(np.random.randn(inputs, outputs) * np.sqrt(1 / inputs))
    self.bias = self.build_param(np.zeros(outputs))
    
  def forward(self, X):
    def backward(D):
      self.weights.gradient += X.T @ D
      self.bias.gradient += D.sum(axis=0)
      return D @ self.weights.tensor.T
    return X @ self.weights.tensor + self.bias.tensor, backward
  
class Sequential(Layer):
  def __init__(self, *layers):
    super().__init__()
    self.layers = layers
    for layer in layers:
      self.parameters.extend(layer.parameters)
    
  def forward(self, X):
    backprops = []
    Y = X
    for layer in self.layers:
      Y, backprop = layer.forward(Y)
      backprops.append(backprop)
    def backward(D):
      for backprop in reversed(backprops):
        D = backprop(D)
      return D
    return Y, backward

class ReLu(Layer):
  def forward(self, X):
    mask = X > 0
    return X * mask, lambda D: D * mask
  
class Sigmoid(Layer):
  def forward(self, X):
    S = 1 / (1 + np.exp(-X))
    def backward(D):
      return D * S * (1 - S)
    return S, backward

def mse_loss(Y_, Y):
  diff = Y_ - Y.reshape(Y_.shape)
  return np.square(diff).mean(), 2 * diff / len(diff)
  
def ce_loss(Y_, Y):
  num = np.exp(Y_)
  den = num.sum(axis=1).reshape(-1, 1)
  prob = num / den
  log_den = np.log(den)
  ce = np.inner(Y_ - log_den, Y)
  return ce.mean(), Y - prob / len(Y)

class SGDOptimizer():
  def __init__(self, lr=0.1):
    self.lr = lr

  def update(self, param):
    param.tensor -= self.lr * param.gradient
    param.gradient.fill(0)

class Learner():
  def __init__(self, model, loss, optimizer):
    self.model = model
    self.loss = loss
    self.optimizer = optimizer
      
  def fit_batch(self, X, Y):
    Y_, backward = self.model.forward(X)
    L, D = self.loss(Y_, Y)
    backward(D)
    self.model.update(self.optimizer)
    return L

  def fit(self, X, Y, epochs, bs):
    losses = []
    for epoch in range(epochs):
      p = np.random.permutation(len(X))
      X, Y = X[p], Y[p]
      loss = 0.0
      for i in range(0, len(X), bs):
        loss += self.fit_batch(X[i:i + bs], Y[i:i + bs])
      losses.append(loss)
      print(f'Epoch: {epoch}, Loss: {loss/bs}')
    return losses

def main():
    num_features = 6 
    epochs = 10000
    batch_size = 10
    learning_rate = 0.001
    model = Sequential(
        Linear(6, 12),
        ReLu(),
        Linear(12, 24),
        ReLu(),
        Linear(24, 12),
        ReLu(),
        Linear(12, 6),
        ReLu(),
        Linear(6, 1),
        ReLu(),
        )
    l = Learner(model, mse_loss, SGDOptimizer(lr=learning_rate))

    with open('./done/qtable_graph_1.pkl', 'rb') as f:
        q = pk.load(f)
    with open('./done/graph_1.pkl', 'rb') as f:
        g = pk.load(f)
    X = [list(x) for x in q if x[0] != x[2] and x[0] != x[1]]
    y = [[min(q[tuple(x)])] for x in X]
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
    X=np.array(X, dtype=float)
    y=np.array(y, dtype=float)
    # X = np.random.randn(num_samples, num_features)
    W = np.random.randn(num_features, 1)
    B = np.random.randn(1)
    # Y = X @ W + B + 0.01 * np.random.randn(num_samples, 1)
    loss=l.fit(X, y, epochs=epochs, bs=batch_size)
    print(loss[-1])
    plt.plot(loss)
    plt.show()
    # print('Weight Matrix Error', np.linalg.norm(m.weights.tensor - W))
    # print('Bias error', np.abs(m.bias.tensor - B)[0])

if __name__=='__main__':
    main()