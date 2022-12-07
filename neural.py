import numpy as np
import matplotlib.pyplot as plt
from graph_utils import bfs
import pickle as pk
from env import Graph, Node

class Parameter():
  def __init__(self, tensor):
    self.tensor = tensor
    self.gradient = np.zeros_like(self.tensor)
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.m = 0
    self.v = 0

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
    for param in self.parameters:      
      param.beta1, param.beta2, param.m, param.v = optimizer.update(param, param.beta1, param.beta2, param.m, param.v)

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

def mae_loss(Y_, Y):
  diff = abs(Y_ - Y.reshape(Y_.shape))
  dprime = 1*(Y_>=Y) + -1*(Y_<Y)
  return diff.mean(), dprime
  
def ce_loss(Y_, Y):
  num = np.exp(Y_)
  den = num.sum(axis=1).reshape(-1, 1)
  prob = num / den
  log_den = np.log(den)
  ce = np.inner(Y_ - log_den, Y)
  return ce.mean(), Y - prob / len(Y)

class AdamOptimizer():
  def __init__(self, lr=0.1):
    self.lr = lr

  def update(self, param, beta1, beta2, m, v):
    g = param.gradient
    m = 0.9*m + (1-0.9)*g
    v = 0.999*v + (1-0.999)*(g)**2

    beta1 *= beta1
    beta2 *= beta2

    mhat = m/(1-beta1)
    vhat = v/(1-beta2)

    param.tensor -= (self.lr * mhat)/(vhat**0.5 + 1e-8)    
    # param.tensor -= self.lr * g
    param.gradient.fill(0)

    return (beta1, beta2, m, v)

class Learner():
  def __init__(self, model, loss, optimizer):
    self.model = model
    self.loss = loss
    self.optimizer = optimizer
      
  def fit_batch(self, X, Y, epochs):
    losses=[]
    for epoch in range(epochs):
      Y_, backward = self.model.forward(X)
      L, D = self.loss(Y_, Y)
      backward(D)
      self.model.update(self.optimizer)
      losses.append(L)
      print(f'Epoch: {epoch}, Loss: {L}')
    return losses

  def fit(self, X, Y, epochs, bs):
    losses = []
    for epoch in range(epochs):
      p = np.random.permutation(len(X))
      X, Y = X[p], Y[p]
      loss = 0.0
      for i in range(0, len(X), bs):
        loss += self.fit_batch(X[i:i + bs], Y[i:i + bs])
      losses.append(loss)
      print(f'Epoch: {epoch}, Loss: {loss}')
    return losses

  def predict(self, X):
    Y, _ = self.model.forward(X)
    return Y

def main():
    num_features = 6 
    epochs = 5000
    batch_size = 1
    learning_rate = 0.01
    model = Sequential(
        Linear(6, 256),
        ReLu(),
        Linear(256, 128),
        ReLu(),
        Linear(128, 24),
        ReLu(),
        Linear(24, 1),
        )
    l = Learner(model, mae_loss, AdamOptimizer(lr=learning_rate))

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
    loss=l.fit_batch(X, y, epochs=epochs)
    print(loss[-1])
    # plt.plot(loss)
    # plt.show()

    with open('v2.pkl', 'wb') as f:
      pk.dump(l, f)

if __name__=='__main__':
    main()