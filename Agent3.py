import random
from entities import Agent
from env import Node
import global_variables as g_v
import numpy as np
from graph_utils import agent_bfs
from math import inf

class Agent3(Agent):
    def __init__(self, node=None) -> None:
        super().__init__(node)
        self.belief = None

    # setting up the beliefs of each node to 1/49 (except the current agent)
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

    def move_rulewise(self, prey_node):
        curr_dist_from_prey, curr_dist_from_pred = agent_bfs(self.node, prey = prey_node)
        chosen_neighbor=None
        priority=inf # variable to allow the better neighbor
        for neighbor in self.node.neighbors:
            path_from_prey, path_from_pred = agent_bfs(neighbor, prey = prey_node)
            # neighbor is closer to prey
            if len(path_from_prey)<len(curr_dist_from_prey):
                # neighbor is farther from predator
                if len(path_from_pred)>len(curr_dist_from_pred):
                    if priority == 1:
                        chosen_neighbor = random.choice([neighbor, chosen_neighbor])
                    else:
                        priority=1
                        chosen_neighbor=neighbor
                # neighbor is not closer to predator
                elif len(path_from_pred)==len(curr_dist_from_pred):
                    if priority>1:
                        if priority == 2: # breaking ties at random
                            chosen_neighbor = random.choice([neighbor, chosen_neighbor])
                        else:
                            priority=2
                            chosen_neighbor=neighbor
                # neighbor is closer to predator, so sit still
                else:
                    if priority>6:
                        if priority == 7:
                            chosen_neighbor = random.choice([neighbor, chosen_neighbor])
                        else:
                            priority=7
                            chosen_neighbor=self.node
            # neighbor is not farther from prey
            elif len(path_from_prey)==len(curr_dist_from_prey):
                # neighbor is farther from predator
                if len(path_from_pred)>len(curr_dist_from_pred):
                    if priority>2:
                        if priority == 3:
                            chosen_neighbor = random.choice([neighbor, chosen_neighbor])
                        else:
                            priority=3
                            chosen_neighbor=neighbor
                # neighbor is not closer to predator
                elif len(path_from_pred)==len(curr_dist_from_pred):
                    if priority>3:
                        if priority == 4:
                            chosen_neighbor = random.choice([neighbor, chosen_neighbor])
                        else:
                            priority=4
                            chosen_neighbor=neighbor
                # neighbor is closer to predator, so sit still
                else:
                    if priority>7:
                        if priority == 8:
                            chosen_neighbor = random.choice([neighbor, chosen_neighbor])
                        else:
                            priority=8
                            chosen_neighbor=self.node
            else:
                # neighbor is farther from predator
                if len(path_from_pred)>len(curr_dist_from_pred):
                    if priority>4:
                        if priority == 5:
                            chosen_neighbor = random.choice([neighbor, chosen_neighbor])
                        else:
                            priority=5
                            chosen_neighbor=neighbor
                # neighbor is not closer to predator
                elif len(path_from_pred)==len(curr_dist_from_pred):
                    if priority>5:
                        if priority == 6:
                            chosen_neighbor = random.choice([neighbor, chosen_neighbor])
                        else:
                            priority=6
                            chosen_neighbor=neighbor
                # neighbor is closer to predator, so sit still
                else:
                    if priority>8:
                        if priority == 9:
                            chosen_neighbor = random.choice([neighbor, chosen_neighbor])
                        else:
                            priority=9
                            chosen_neighbor=self.node
        self.node.agent=False
        self.node=chosen_neighbor
        self.node.agent=True

    def move(self):
        if self.belief == None:
            self.initialize_belief()
        
        self.survey_node()
        prey = self.get_prey_location()
        self.move_rulewise(prey)
        self.propagate_prey_belief()