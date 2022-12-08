# environment for Circle of Life
import random
import shutil
from typing import Tuple
import global_variables as g_v
import pickle as pk
from entities import Agent, Prey, DistractedPredator
import os

class Node:
    def __init__(self, pos, agent = False, prey = False, predator = False) -> None:
        self.pos: int = pos
        self.agent = agent
        self.prey = prey
        self.predator = predator
        self.neighbors = set()
        self.u_star = None

    def degree(self):
        return len(self.neighbors)

class Graph:
    def init_nodes(self):
        # create new nodes
        for i in range(self.numNodes):
            new_node = Node(i)
            self.graph_nodes.append(new_node)
    
    def make_circle(self):
        # adds prev and next neighbor to each node
        for i,node in enumerate(self.graph_nodes):
            next_node = (i+1)%g_v.Number_of_nodes
            prev_node = (i-1)%g_v.Number_of_nodes
            node.neighbors.add(self.graph_nodes[next_node])
            node.neighbors.add(self.graph_nodes[prev_node])

    def node_selection(self, node: Node):
        x=node.pos
        potential_nodes=[]
        for i in range(x-5, x+6):
            temp = i % g_v.Number_of_nodes
            temp_node = self.graph_nodes[temp]

            if temp not in [x-1, x, x+1] and temp_node.degree() < 3:
                potential_nodes.append(temp_node)
        
        # return a random potential node
        if potential_nodes != []:
            return random.choice(potential_nodes)
        else:
            return None

    def add_new_edges(self):
        indexes = list(range(g_v.Number_of_nodes))
        while(len(indexes) > 0):
            random_index = random.choice(indexes)
            indexes.remove(random_index)
            curr_node = self.graph_nodes[random_index]

            if curr_node.degree() > 2:
                continue

            choice_node = self.node_selection(curr_node)
            curr_node.neighbors.add(choice_node)
            if choice_node:
                choice_node.neighbors.add(curr_node)

        # remove neigbors valued None
        for node in self.graph_nodes:
            if None in node.neighbors:
                node.neighbors.remove(None)


    def __init__(self) -> None:
        self.numNodes = g_v.Number_of_nodes
        # list to store the node objects
        self.graph_nodes : list[Node] = []
        self.init_nodes()
        self.make_circle()
        # self.add_new_edges()

    def get_random_positions(self):
        positions = list(range(g_v.Number_of_nodes))
        prey_pos = random.choice(positions)
        pred_pos = random.choice(positions)

        positions.remove(prey_pos)
        if pred_pos in positions:
            positions.remove(pred_pos)
        
        agent_pos = random.choice(positions)

        return (prey_pos, pred_pos, agent_pos)

 
    def spawn_entities(self, agent, prey, predator):
        prey_pos, pred_pos, agent_pos = self.get_random_positions()

        self.graph_nodes[prey_pos].prey = True
        prey.node = self.graph_nodes[prey_pos]

        self.graph_nodes[pred_pos].predator = True
        predator.node = self.graph_nodes[pred_pos]

        self.graph_nodes[agent_pos].agent = True
        agent.node = self.graph_nodes[agent_pos]


def main():
    
    # if os.path.exists(g_v.graph_folder):
    #     shutil.rmtree(g_v.graph_folder)
    # os.mkdir(g_v.graph_folder)
    for i in range(3,4):
        g = Graph()
        filename= f'{g_v.graph_folder}/graph_{i+1}.pkl'
        with open(filename, 'wb') as filehandler:
            pk.dump(g, filehandler)

if __name__ == "__main__":
    main()