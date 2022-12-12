import copy
from time import perf_counter
from env import Graph, Node
from entities import Predator, Prey, Agent, DistractedPredator
from viz import graph_viz
from Agent1 import Agent1
from Agent2 import Agent2
from Agent3 import Agent3
from Agent4 import Agent4
from ustar_Agent import UstarAgent
from upartial_Agent import UpartialAgent
from vpartial_Agent import VpartialAgent
from v_Agent import VAgent
import numpy as np
import pickle as pk
import os
import imageio
from neural import Model, Sequential, Parameter, Linear, ReLu, mse_loss, mae_loss, AdamOptimizer


class Game:
    def __init__(self, agent: Agent, graph: Graph) -> None:
        self.agent = agent
        self.prey = Prey()
        self.predator = DistractedPredator()
        self.graph = graph
        self.graph.spawn_entities(self.agent, self.prey, self.predator)
        self.agent.graph_nodes = graph.graph_nodes
        self.maxtimestep = 150
        self.timestep = 0
        self.victory = (False, False)

    def running(self):
        if self.agent.node.predator:
            self.victory = (False, True)
            return False
        if self.agent.node.prey:
            self.victory = (True, True)
            return False
        if self.timestep == self.maxtimestep:
            self.victory = (True, False)
            return False

        return True
    
    def game_viz(self):
        graph_viz(self.graph, self.agent.node.pos, self.prey.node.pos, self.predator.node.pos, self.timestep)

    def create_gif(self):
        directory='viz'
        images = os.listdir(directory)
        filtered_images=[file for file in images if file.endswith('.png')]
        with imageio.get_writer(directory+'/'+self.agent.__class__.__name__+'_viz.gif', mode='I') as writer:
            for filename in filtered_images:
                image = imageio.imread(directory+'/'+filename)
                writer.append_data(image)
            for filename in filtered_images:
                filepath=os.path.join(directory, filename)
                os.remove(filepath)

    def run(self):
        '''
        Update graph at every timestep
        '''
        while(self.running()):
            # self.game_viz()
            self.timestep += 1
            self.agent.move()
            if not self.running():
                break
            self.prey.move()
            if not self.running(): # in case the prey moves into the agent's node
                break
            self.predator.move()
        # self.game_viz()
        # self.create_gif()
        return self.victory, self.timestep

def run_game(graph, agent):
    game = Game(agent, graph)
    return game.run()

if __name__ == "__main__":
    a = perf_counter()
    num_agents = 4
    iterations=100
    win = np.zeros(num_agents)
    loss2 = np.zeros(num_agents)
    agent_caught = np.zeros(num_agents)
    with open('./utable/utable_graph_1.pkl','rb') as f:
        q = pk.load(f)

    with open('./graphs/graph_1.pkl','rb') as f:
        graph = pk.load(f)
    for _ in range(iterations):
        victories = []
        agents = [UstarAgent(utable=q), VAgent(utable=q), UpartialAgent(utable=q), VpartialAgent(utable=q)]
        correct_prey_guess={agent:0 for agent in agents}
        correct_predator_guess={agent:0 for agent in agents}
        graph.get_random_positions()
        for agent in agents:
            graph_copy = copy.deepcopy(graph)
            v, timesteps = run_game(graph_copy, agent)
            victories.append(v)
            prey_guess_rate=agent.correct_prey_guess/timesteps
            predator_guess_rate=agent.correct_predator_guess/timesteps
            correct_prey_guess[agent]+=prey_guess_rate
            correct_predator_guess[agent]+=predator_guess_rate

        for i,victory in enumerate(victories):
            if False not in victory:
                win[i] += 1
            elif victory[1] == False:
                loss2[i] += 1
            elif victory[0] == False:
                agent_caught[i] += 1

    for w, l2, death, agent in zip(win, loss2, agent_caught, agents):
        print()
        print(f"Agent: {agent.__class__.__name__}")
        print(f"win percentage: {(w/iterations)*100}")
        print(f"loss from timeout: {(l2/iterations)*100}")
        print(f"agent caught by predator: {(death/iterations)*100}")
        print(f"Accuracy of prey guess over timesteps: {(correct_prey_guess[agent]/iterations)*100}")
        print(f"Accuracy of predator guess over timesteps: {(correct_predator_guess[agent]/iterations)*100}")
    print()
    b = perf_counter()

    print(f"time taken:{b-a}")