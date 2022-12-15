from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from graph_utils import bfs
import pickle as pk
from env import Graph, Node
import pandas as pd

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

  @abstractmethod
  def forward(self, X):
    pass

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
    self.bias = self.build_param(np.random.randn(outputs) - 0.5)

    self.X = None
  
  def backprop(self, D):
    self.weights.gradient += np.matmul(self.X.T, D)
    self.bias.gradient += D.sum(axis=0)
    return np.matmul(D, self.weights.tensor.T)

  def forward(self, X):
    self.X = X
    return np.matmul(X, self.weights.tensor) + self.bias.tensor
  
class Sequential(Layer):
  def __init__(self, *layers):
    super().__init__()
    self.layers = layers
    for layer in layers:
      self.parameters.extend(layer.parameters)

    self.backprops = []

  def backprop_model(self, D):
    for backprop in reversed(self.backprops):
      D = backprop(D)
    
  def forward(self, X):
    self.backprops = []
    Y = X
    for layer in self.layers:
      Y = layer.forward(Y)
      self.backprops.append(layer.backprop)
    return Y

class ReLu(Layer):
  def __init__(self):
    super().__init__()
    self.mask = None

  def forward(self, X):
    self.mask = X > 0
    return X * self.mask

  def backprop(self, D):
    return D * self.mask

class Model():
  def __init__(self, model, loss, optimizer):
    self.model = model
    self.loss = loss
    self.optimizer = optimizer
      
  def fit_batch(self, X, Y, epochs):
    losses=[]
    for epoch in range(epochs):
      Y_pred = self.model.forward(X)
      Loss, Derivative = self.loss(Y_pred, Y)
      self.model.backprop_model(Derivative)
      self.model.update(self.optimizer)
      losses.append(Loss)
      print(f'Epoch: {epoch}, Loss: {Loss}')
    return losses

  def test_train(self, X_train, y_train, X_test, y_test, epochs):
    train_losses=[]
    test_losses=[]
    for epoch in range(epochs):
      #train
      Y_pred = self.model.forward(X_train)
      Loss, Derivative = self.loss(Y_pred, y_train)
      self.model.backprop_model(Derivative)
      self.model.update(self.optimizer)
      train_losses.append(Loss)

      #test
      Y_pred = self.model.forward(X_test)
      Loss, Derivative = self.loss(Y_pred, y_test)
      test_losses.append(Loss)
      print(f'Epoch: {epoch}, Loss: {Loss}')
    return train_losses, test_losses

  def predict(self, X):
    Y = self.model.forward(X)
    return Y


def mse_loss(Y_pred, Y):
  diff = Y_pred - Y.reshape(Y_pred.shape)
  return np.square(diff).mean(), 2 * diff / len(diff)

def mae_loss(Y_pred, Y):
  diff = abs(Y_pred - Y.reshape(Y_pred.shape))
  dprime = 1*(Y_pred>=Y) + -1*(Y_pred<Y)
  return diff.mean(), dprime

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
    param.gradient.fill(0)

    return (beta1, beta2, m, v)

def train_test_split(X,y):
  l = len(X)
  idxes = np.arange(0, l)
  np.random.shuffle(idxes)
  train_idx = int(0.8 * l)
  X_train = X[idxes[:train_idx]]
  X_test = X[idxes[train_idx:]]
  y_train = y[idxes[:train_idx]]
  y_test = y[idxes[train_idx:]]

  return X_train, y_train, X_test, y_test
  
def main():
    epochs = 5000
    learning_rate = 0.001
    model = Sequential(
        Linear(52, 256),
        ReLu(),
        Linear(256, 256),
        ReLu(),
        Linear(256, 1),
        )
    l = Model(model, mae_loss, AdamOptimizer(lr=learning_rate))
    X, y = vpartial(1)
    X=np.array(X, dtype=float)
    y=np.array(y, dtype=float)
    loss=l.fit_batch(X, y, epochs=epochs)

    # X_train, y_train, X_test, y_test = train_test_split(X,y)
    # trainloss, testloss = l.test_train(X_train, y_train, X_test, y_test, 5000)
    # plt.plot(trainloss, label="train loss")
    # plt.plot(testloss, label="test loss")
    # plt.legend(loc="upper right")
    # plt.show()
    print(loss[-1])
    with open('vpartial.pkl', 'wb') as f:
      pk.dump(l, f)

def vstar():
  with open('./utable/utable_graph_1.pkl', 'rb') as f:
    q = pk.load(f)
  with open('./graphs/graph_1.pkl', 'rb') as f:
    g = pk.load(f)
  X = [list(x) for x in q if x[0] != x[2]]
  y = [[q[tuple(x)]] for x in X]
  dist = dict()

  #pre-processing
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
  return X, y

def vpartial(sharp):
  if sharp:
    u = pd.read_csv('usharp.csv')
  else:
    u = pd.read_csv('upartial.csv')
  X = u.iloc[:,1:]
  X = X.to_numpy()
  y = u.iloc[:,:1]
  y = y.to_numpy()
  return X, y

if __name__=='__main__':
    main()