from abc import abstractmethod
import random
from graph_utils import bfs
from math import inf
import numpy as np

class Agent:
    def __init__(self, node = None) -> None:
        self.node = node
        self.graph_nodes = None
        self.correct_prey_guess=0
        self.correct_predator_guess=0

    @abstractmethod
    def move(self) -> None:
        pass

class Prey:
    def __init__(self, node = None) -> None:
        self.node = node
    
    def move(self):
        moveset = list(self.node.neighbors)
        moveset.append(self.node)
        next_node = random.choice(moveset)
        self.node.prey = False
        self.node = next_node
        self.node.prey = True

class Predator:
    def __init__(self, node = None) -> None:
        self.node = node

    def move(self):
        min_length = inf
        positions = []
        for neighbor in self.node.neighbors:
            path = bfs(neighbor)
            if len(path) < min_length:
                positions = [neighbor]
                min_length = len(path)
            elif len(path) == min_length:
                positions.append(neighbor)
        
        position = random.choice(positions)
        self.node.predator = False
        self.node = position
        self.node.predator = True

class DistractedPredator:
    def __init__(self, node = None) -> None:
        self.node = node

    def move_closer(self):
        min_length = inf
        positions = []
        for neighbor in self.node.neighbors:
            path = bfs(neighbor)
            if len(path) < min_length:
                positions = [neighbor]
                min_length = len(path)
            elif len(path) == min_length:
                positions.append(neighbor)
        
        position = random.choice(positions)
        self.node.predator = False
        self.node = position
        self.node.predator = True

    def move_random(self):
        moveset = list(self.node.neighbors)
        position = random.choice(moveset)
        self.node.predator = False
        self.node = position
        self.node.predator = True

    def move(self):
        rng = np.random.uniform()
        if rng <= 0.6:
            self.move_closer()
        else:
            self.move_random()
    