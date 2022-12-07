from math import inf
import os
import random
import shutil
from env import Graph, Node
import numpy as np
import pickle as pk
import global_variables as g_v
from graph_utils import bfs


class UstarGen:
    def mk_initial_table(self):
        if f'qtable_{self.filename}' in os.listdir(g_v.qtable_folder):
            print('here')
            with open(f'{g_v.qtable_folder}/qtable_{self.filename}', 'rb') as f:
                q = pk.load(f)
        else:
            q = dict()
            for agent in range(g_v.Number_of_nodes):
                for prey in range(g_v.Number_of_nodes):
                    for predator in range(g_v.Number_of_nodes):
                        degree = self.graph_nodes[agent].degree() + 1
                        actions = np.zeros(degree) + 1000
                        if predator == agent:
                            actions +=1000000
                        elif prey == agent:
                            actions = np.zeros(degree)

                        q[(agent,prey,predator)] = actions
        return q

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

    def __init__(self, g: Graph, filename) -> None:
        self.epochs = 100

        self.graph_nodes :list[Node]= g.graph_nodes
        self.state = (41,14,14)
        self.filename = filename
        self.prey_transition_matrix = self.prey_transition()


        self.qtable = self.mk_initial_table()

    def save_qtable(self):
        filename = f'{g_v.qtable_folder}/qtable_{self.filename}'
        with open(filename, 'wb') as f:
            pk.dump(self.qtable, f)

    def run(self):
        for i in range(self.epochs):
            for state in self.qtable:
                number_of_actions = len(self.qtable[state])
                ustars = []
                agent_pos, prey_pos, pred_pos = state
                if agent_pos == prey_pos or agent_pos == pred_pos:
                    continue
                neighbors = list(self.graph_nodes[agent_pos].neighbors)
                for action in range(number_of_actions):
                    if action != len(neighbors):
                        next_agent_pos = neighbors[action].pos
                    else:
                        next_agent_pos = agent_pos
                    
                    # if next_agent_pos == prey_pos and next_agent_pos != pred_pos:
                    #     ustars = [(action, 1)]
                    #     break
                    # if pred_pos == next_agent_pos:
                    #     ustars.append((action, 10000))
                    #     continue

                    s = 0
                    prey_prob = 1/len(self.prey_transition_matrix[prey_pos])
                    pred_prob_dict = self.propagate_pred_belief(pred_pos, next_agent_pos)
                    for next_prey_pos in self.prey_transition_matrix[prey_pos]:
                        for next_pred_pos in pred_prob_dict:
                            new_state = (next_agent_pos, next_prey_pos, next_pred_pos)
                            s+= prey_prob * pred_prob_dict[next_pred_pos] * min(self.qtable[new_state])
                    s+=1
                    ustars.append((action, s))

                beststar = min(ustars, key=lambda x: x[1])
                best_action, best_u  = beststar[0], beststar[1]
                self.qtable[state][best_action] = best_u
            print(f'{min(self.qtable[self.state])}, {self.state}, {i+1}')
            self.save_qtable()

if __name__ == '__main__':
    if not os.path.exists(g_v.qtable_folder):
        os.mkdir(g_v.qtable_folder)
    g : Graph = None
    for filename in os.listdir(g_v.graph_folder)[1:]:
        with open(f'{g_v.graph_folder}/{filename}', 'rb') as f:
            g = pk.load(f)        
        
        ql = UstarGen(g, filename)
        ql.run()