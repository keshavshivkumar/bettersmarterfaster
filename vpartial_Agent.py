from upartial_Agent import UpartialAgent
import global_variables as g_v
import numpy as np
import pickle as pk

class VpartialAgent(UpartialAgent):
    def __init__(self, node=None, utable=None) -> None:
        super().__init__(node, utable)
        with open('vpartial.pkl','rb') as f:
            self.v = pk.load(f)
    
    def get_vpartial(self, agent_pos, pred_pos):
        vpartial = 0
        v_state = [agent_pos, pred_pos]
        prey_belief=[]
        p=0
        for i in range(50):
            if i in self.belief:
                p=self.belief[i]
            else:
                p=0
            prey_belief.append(p)
        v_state.extend(prey_belief)
        vpartial = self.v.predict(np.array(v_state))[0]
        return vpartial

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
                vpartial = self.get_vpartial(next_agent_pos, next_pred_pos)
                if next_agent_pos == next_pred_pos:
                    s+= pred_prob_dict[next_pred_pos] * 1000000
                else:   
                    s+= pred_prob_dict[next_pred_pos] * vpartial
            s+=1
            ustars.append(s)
        
        best_action = np.argmin(ustars)
        if best_action != len(neighbors):
            chosen = neighbors[best_action]
        else:
            chosen = self.node
        return chosen