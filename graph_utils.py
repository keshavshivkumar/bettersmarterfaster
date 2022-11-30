from queue import Queue
from math import inf
from secrets import choice

def get_bfs_path(came_from: dict, end_node):
    cur_node = end_node
    path = [cur_node]
    while cur_node != None:
        path.append(came_from[cur_node])
        cur_node = came_from[cur_node]
    return path[:-2]

def bfs(start_node, end_node=None):
    q = Queue()
    q.put(start_node)
    came_from = dict()
    came_from[start_node] = None

    while not q.empty():
        current_node = q.get()

        if end_node is not None:
            if current_node==end_node:
                break
            else:
                pass
        else:
            if current_node.agent:
                break

        for next in current_node.neighbors:
            if next not in came_from:
                q.put(next)
                came_from[next] = current_node
    
    return get_bfs_path(came_from, current_node)

def agent_bfs(start_node, prey = None, pred = None):
    q = Queue()
    q.put(start_node)
    came_from = dict()
    came_from[start_node] = None

    prey_node = pred_node = None
    while not q.empty():
        current_node = q.get()
        if prey is not None:
            if current_node == prey:
                prey_node = current_node    
        elif current_node.prey:
            prey_node=current_node

        if pred is not None:
            if current_node == pred:
                pred_node = current_node
        elif current_node.predator:
            pred_node=current_node

        if prey_node!=None and pred_node !=None:
            break

        for next in current_node.neighbors:
            if next not in came_from:
                q.put(next)
                came_from[next] = current_node
    
    return get_bfs_path(came_from, prey_node), get_bfs_path(came_from, pred_node)

def predicted_prey_move(agent, prey):
    probable_moves=list(prey.neighbors)
    probable_moves.append(prey)
    choices=[]
    # sort according to distance from agent
    for neighbor_prey in probable_moves:
        path = bfs(agent, neighbor_prey)
        choices.append(path)
    choices.sort(key=lambda x:len(x))
    return choices[-1] # returns largest