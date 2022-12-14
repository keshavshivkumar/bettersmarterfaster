import random
from entities import Agent
from env import Node
import global_variables as g_v
import numpy as np
from graph_utils import agent_bfs, bfs
from math import inf
import numpy as np
import csv

class UpartialAgent(Agent):

    def initialize_belief(self):
        belief_list = np.zeros(g_v.Number_of_nodes) + 1/(g_v.Number_of_nodes - 1)
        belief_list[self.node.pos] = 0
        self.belief = dict()
        for i,prob in enumerate(belief_list):
            if prob:
                self.belief[i] = prob
    
    # propogates the beliefs to non-zero beliefs after agent makes its move
    def propagate_prey_belief(self):
        new_belief = np.zeros(g_v.Number_of_nodes)
        for node_pos in self.belief:
            len_positions = self.graph_nodes[node_pos].degree() + 1
            positions = [x.pos for x in self.graph_nodes[node_pos].neighbors]
            positions.append(node_pos)
            for pos in positions:
                new_belief[pos] += self.belief[node_pos]/len_positions

        self.belief = dict()
        for i,prob in enumerate(new_belief):
            if prob:
                self.belief[i] = prob
    
    # adjusts the beliefs after picking a node to survey
    def distribute_prob(self, node_pos):
        if node_pos not in self.belief:
            return
        prob = self.belief.pop(node_pos)
        denominator = 1-prob

        for node_pos in self.belief:
            self.belief[node_pos] /= denominator


    def get_random_highest_prob(self):
        max_prob = max(self.belief.values())
        max_belief = dict()
        for node_pos in self.belief:
            if self.belief[node_pos] == max_prob:
                max_belief[node_pos] = max_prob

        return random.choice(list(max_belief.keys()))
        
    def survey_node(self):
        self.distribute_prob(self.node.pos)

        node_pos_with_highest_prob = self.get_random_highest_prob()
        if self.graph_nodes[node_pos_with_highest_prob].prey == False:
            self.distribute_prob(node_pos_with_highest_prob)
        else:
            self.belief = dict()
            self.belief[node_pos_with_highest_prob] = 1
            self.correct_prey_guess+=1

    def get_prey_location(self) -> Node:
        node_pos = self.get_random_highest_prob()
        return self.graph_nodes[node_pos]

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
        self.belief = None

    def write_to_csv(self, upartial, agent_pos, predator_pos):
        with open('upartial.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            state = [upartial, agent_pos, predator_pos]
            temp_state=[]
            for i in range(50):
                if i in self.belief:
                    temp_state.append(self.belief[i])
                else:
                    temp_state.append(0)
            state.extend(temp_state)
            writer.writerow(state)

    def get_upartial(self, agent_pos, pred_pos):
        upartial = 0
        for prey_pos in self.belief:
            upartial += self.belief[prey_pos] * self.utable[(agent_pos, prey_pos, pred_pos)]

        # if upartial < 50:
        #     self.write_to_csv(upartial, agent_pos, pred_pos)
        return upartial

    def best_action(self, curr_state):
        agent_pos, pred_pos = curr_state
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
            # prey_prob = 1/len(self.prey_transition_matrix[prey_pos])
            pred_prob_dict = self.propagate_pred_belief(pred_pos, next_agent_pos)
            # for next_prey_pos in self.prey_transition_matrix[prey_pos]:
            for next_pred_pos in pred_prob_dict:
                # new_state = (next_agent_pos, next_prey_pos, next_pred_pos)
                upartial = self.get_upartial(next_agent_pos, next_pred_pos)
                s+= pred_prob_dict[next_pred_pos] * upartial
            s+=1
            ustars.append(s)
        
        best_action = np.argmin(ustars)
        if best_action != len(neighbors):
            chosen = neighbors[best_action]
        else:
            chosen = self.node
        return chosen
            
    def get_pred_pos(self):
        _, curr_dist_from_pred = agent_bfs(self.node)
        pred_pos = curr_dist_from_pred[0].pos

        return pred_pos


    def move(self):
        if self.init:
            self.prey_transition_matrix = self.prey_transition()
            self.initialize_belief()
            self.init = False

        self.survey_node()
        pred_pos = self.get_pred_pos()

        curr_state = (self.node.pos, pred_pos)
        chosen_neighbor = self.best_action(curr_state)

        self.node.agent = False
        self.node = chosen_neighbor
        self.node.agent = True

        self.propagate_prey_belief()