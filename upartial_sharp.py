import random
from upartial_Agent import UpartialAgent
from env import Node
import global_variables as g_v
import numpy as np
from graph_utils import agent_bfs, bfs
import pickle as pk
import numpy as np
import csv

class UpartialSharp(UpartialAgent):
    def __init__(self, node=None, utable=None) -> None:
        super().__init__(node, utable)
        self.dist=dict()
        with open('v5.pkl','rb') as f:
            self.v = pk.load(f)
    
    def write_to_csv(self, usharp, agent_pos, predator_pos):
        with open('usharp.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            state = [usharp, agent_pos, predator_pos]
            temp_state=[]
            for i in range(50):
                if i in self.belief:
                    temp_state.append(self.belief[i])
                else:
                    temp_state.append(0)
            state.extend(temp_state)
            writer.writerow(state)

    def get_upartial_sharp(self, agent_pos, pred_pos):
        usharp=0
        for prey_pos in self.belief:
            usharp_state = [agent_pos, prey_pos, pred_pos]
            if (usharp_state[0], usharp_state[1]) not in self.dist:
                self.dist[(usharp_state[0], usharp_state[1])] = len(bfs(
                    self.graph_nodes[usharp_state[0]], 
                    self.graph_nodes[usharp_state[1]]
                    ))
            if (usharp_state[0], usharp_state[2]) not in self.dist:
                self.dist[(usharp_state[0], usharp_state[2])] = len(bfs(
                    self.graph_nodes[usharp_state[0]], 
                    self.graph_nodes[usharp_state[2]]
                    ))
            if (usharp_state[1], usharp_state[2]) not in self.dist:
                self.dist[(usharp_state[1], usharp_state[2])] = len(bfs(
                    self.graph_nodes[usharp_state[1]], 
                    self.graph_nodes[usharp_state[2]]
                    ))
            usharp_state.append(self.dist[(usharp_state[0], usharp_state[1])])
            usharp_state.append(self.dist[(usharp_state[1], usharp_state[2])])
            usharp_state.append(self.dist[(usharp_state[0], usharp_state[2])])

            if agent_pos == pred_pos:
                usharp += self.belief[prey_pos] * 1000000
            else:
                usharp += self.belief[prey_pos] * self.v.predict(np.array(usharp_state))[0]
        # if usharp < 50:
        #     self.write_to_csv(usharp, agent_pos, pred_pos)
        return usharp
    
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
            pred_prob_dict = self.propagate_pred_belief(pred_pos, next_agent_pos)
            for next_pred_pos in pred_prob_dict:
                usharp = self.get_upartial_sharp(next_agent_pos, next_pred_pos)
                s+= pred_prob_dict[next_pred_pos] * usharp
            s+=1
            ustars.append(s)
        best_action = np.argmin(ustars)
        if best_action != len(neighbors):
            chosen = neighbors[best_action]
        else:
            chosen = self.node
        return chosen