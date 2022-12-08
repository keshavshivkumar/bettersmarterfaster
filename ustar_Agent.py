import random
from entities import Agent
from env import Node
import global_variables as g_v
import numpy as np
from graph_utils import agent_bfs, bfs
from math import inf
import numpy as np

class UstarAgent(Agent):

    def propagate_pred_belief(self, pred_node_pos ,agent_node_pos):
        new_belief = np.zeros(g_v.Number_of_nodes)

        curr_pred_node = self.graph_nodes[pred_node_pos]
        future_positions: list[Node] = []
        min_length = inf
        for neighbor in curr_pred_node.neighbors:     
            length = len(bfs(neighbor, self.graph_nodes[agent_node_pos])) # dist from future pred to agent
            if length < min_length:
                future_positions = [neighbor]
                min_length = length
            elif length == min_length:
                future_positions.append(neighbor)

        len_positions = len(future_positions)
        for node in future_positions:
            new_belief[node.pos] += 0.6/len_positions
            
        len_positions = self.graph_nodes[pred_node_pos].degree()
        positions = [x.pos for x in self.graph_nodes[pred_node_pos].neighbors]
        for pos in positions:
            new_belief[pos] += 0.4/len_positions

        d = dict()
        for i,prob in enumerate(new_belief):
            if prob:
                d[i] = prob

        return d

    def prey_transition(self):
        '''
        Returns dict
            key: current pos of prey
            value: List of all possible position after 1 timestep
        '''
        matrix = dict()
        for node_pos in range(g_v.Number_of_nodes):
            positions = [x.pos for x in self.graph_nodes[node_pos].neighbors]
            positions.append(node_pos)
            matrix[node_pos] = positions
        return matrix

    def __init__(self, node=None, utable = None) -> None:
        super().__init__(node)
        self.utable = utable
        self.init = True

    def best_action(self, curr_state):
        agent_pos, prey_pos, pred_pos = curr_state
        ustars = []
        neighbors = list(self.node.neighbors)
        for action in range(len(neighbors)+1):
            if action != len(neighbors):
                next_agent_pos = neighbors[action].pos
            else:
                next_agent_pos = agent_pos
            
            if next_agent_pos == pred_pos:
                ustars.append(10000)
                continue

            s = 0
            prey_prob = 1/len(self.prey_transition_matrix[prey_pos])
            pred_prob_dict = self.propagate_pred_belief(pred_pos, next_agent_pos)
            for next_prey_pos in self.prey_transition_matrix[prey_pos]:
                for next_pred_pos in pred_prob_dict:
                    new_state = (next_agent_pos, next_prey_pos, next_pred_pos)
                    s+= prey_prob * pred_prob_dict[next_pred_pos] * self.utable[new_state]
            s+=1
            ustars.append(s)
        best_action = np.argmin(ustars)
        # beststar = min(ustars, key=lambda x: x[1])
        # best_action, best_u  = beststar[0], beststar[1]
        if best_action != len(neighbors):
            chosen = neighbors[best_action]
        else:
            chosen = self.node
        # print(curr_state, min(self.utable[curr_state]), chosen.pos)
        return chosen
            

    def move(self):
        if self.init:
            self.prey_transition_matrix = self.prey_transition()
            self.init = False

        curr_dist_from_prey, curr_dist_from_pred = agent_bfs(self.node)
        prey_pos = curr_dist_from_prey[0].pos
        pred_pos = curr_dist_from_pred[0].pos
        curr_state = (self.node.pos, prey_pos, pred_pos)
        chosen_neighbor = self.best_action(curr_state)

        self.node.agent = False
        self.node = chosen_neighbor
        self.node.agent = True
