from env import Graph, Node
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import global_variables as g_v

def graph_edges(graph):
# def main():
    vertices=[]
    for node in graph.graph_nodes:
        for neighbor in node.neighbors:
            temp=[node.pos]
            temp.append(neighbor.pos)
            if temp[::-1] not in vertices:
                vertices.append(temp)
    vertices=np.array(vertices)
    df = pd.DataFrame(vertices, columns=['Node 1', 'Node 2'])
    dist=[]
    for _, node in df.iterrows():
        difference = node[0] - node[1]
        if abs(difference)<=24:
            difference = abs(difference)
        distance = difference % g_v.Number_of_nodes 
        dist.append(distance)
    df['Distance'] = dist
    return df

def graph_viz(graph, agent_pos, prey_pos, predator_pos, timestep):
    dataframe = graph_edges(graph)
    G = nx.Graph()
    for i in graph.graph_nodes:
        G.add_node(i.pos)
    for _, nodes in dataframe.iterrows():
        G.add_edge(nodes[0], nodes[1], distance=nodes[2])
    colors = []
    for node in G:
        if node==agent_pos:
            colors.append('blue')
        elif node==prey_pos:
            colors.append('green')
        elif node == predator_pos:
            colors.append('red')
        else:
            colors.append('black')
    plt.figure(figsize=(8, 8))
    plt.title(f'Timestep: {timestep}', size = 15)
    nx.draw_circular(G, edge_color ='black', node_color = colors, node_size = 80, font_size=8, font_color='white', with_labels=False)
    for i in range(5):
        plt.savefig(f'viz/{timestep}_{i}.png')
    # plt.show()

