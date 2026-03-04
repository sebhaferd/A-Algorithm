import networkx as nx
import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.legend_handler import HandlerPatch


#Sebastian Haferd
#Implementing CBS algorithm

def location(path, t):
    if t<len(path):
        return path[t]
    else:
        return -1
    

def find_conflicts(paths, agents):
    max_time = 0
    for path in paths.values():
        max_time = max(len(path), max_time)
    
    for t in range(max_time):
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                continue
    pass


def cbs(graph, agents, sources, targets):
    #initialize open list
    open_list = []

    #create root node for conflict tree
    root = {"constraints": [], "paths": {}, "cost" : 0}


    #run a star search on each agent
    for i in range(len(agents)):
        agent = agents[i]
        source = sources[i]
        target = targets[i]
        path, cost = astar(graph, source, target, heuristic)
        if path is None:
            return None
        root["paths"][i] = path
        root["cost"]+=cost
    

def heuristic(source, target):
    return abs(target[0] - source[0]) + abs(target[1]-source[1])

def valid_node(grid, r, c):
    if r<0 or c < 0 or r >= len(grid) or c >= len(grid[0]):
        return False
    if grid[r][c] == 0:
        return False
    return True

def get_neighbors(grid, node):
    row, col = node
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    result = []
    for dr, dc in dirs:
        new_row = row+dr
        new_col = col + dc
        if valid_node(grid, new_row, new_col):
            result.append((new_row, new_col))
    return result

def edge_collision(prev_state, next_state):
    p1, p2, p3 = prev_state
    n1, n2, n3 = next_state

    if p1 == n2 and p2 == n1:
        return True
    if p1 == n3 and p3 == n1:
        return True
    if p2 == n3 and p3 == n2:
        return True
    return False

#create path from source to target of each node
def make_path(path, start_state, end_state):
    res = []
    current = end_state
    if end_state not in path and end_state != start_state:
        return None
    while current != start_state:
        res.append(current)
        current = path[current]

    res.append(start_state)
    res.reverse()
    return res

#A* search alrogithm, calculating through distance to each node finding the shortest path
def astar(grid, start_state, end_state, heuristic):
    #Set state as tuple of two nodes
    source1, source2, source3 = start_state 
    (target1, target2, target3) = end_state

    #check if already at goal
    if start_state == end_state:
        return [start_state], 0
    
    #create min heap queue to store nodes in path and h and g values
    open_list = []
    heapq.heappush(open_list, ((heuristic(source1, target1)+heuristic(source2, target2)+heuristic(source3, target3), start_state)))

    #dictionary of g value cost to states
    costs = {}
    costs[start_state] = 0

    #dictionary of regression map for path
    path = {}

    #set for closed list of nodes that have been fully explored
    closed_list = set()

    #loop until every path has been explored or target reached
    while open_list:
        #pop smallest f value from open list 
        f_cost, state = heapq.heappop(open_list)

        #state already explored 
        if state in closed_list:
            continue

        #reached target, return path
        if state == end_state:
            return make_path(path, start_state, state), costs[state]
        
        #mark state as explored
        closed_list.add(state)

        #list of neighbors and weights of each state
        neighbors1 = get_neighbors(grid, state[0]) + [state[0]]
        neighbors2 = get_neighbors(grid, state[1]) +  [state[1]]
        neighbors3 = get_neighbors(grid, state[2]) +  [state[2]]

        #nested loop to avoid collisions between agents
        for n1 in neighbors1:
            for n2 in neighbors2:
                for n3 in neighbors3:
                    next_state = (n1, n2, n3)
                    #check if on same node
                    if n1 == n2 or n1 == n3 or n2 == n3:
                        continue
                    #check if exploring same edge
                    if edge_collision(state, next_state):
                        continue

                    #new g cost
                    temp_cost = costs[state] + 1

                    #check if new path or if more efficient than previously explored path
                    if next_state not in costs or temp_cost < costs[next_state]:
                        costs[next_state] = temp_cost
                        f_cost = temp_cost + heuristic(n1, target1)+heuristic(n2, target2)+heuristic(n3, target3)
                        heapq.heappush(open_list, (f_cost, next_state))
                        path[next_state] = state

    #all paths explored and no solution, distance to goal is infinity
    return None, float('inf')


grid1 = [
[1,1,1,1,1,1,1],
[0,0,0,1,0,0,0],
[0,0,0,1,0,0,0],
[0,0,0,1,0,0,0],
[0,0,0,1,0,0,0],
[0,0,0,1,0,0,0]
]


grid2 = [
[1,1,1,1,1,1,1,1,1,1],
[1,0,1,1,1,1,0,0,1,1],
[0,1,1,1,0,1,1,1,0,1],
[0,1,0,1,1,1,0,1,1,1],
[0,1,1,1,1,0,1,1,1,1],
[0,1,1,0,1,0,1,0,1,1],
[1,1,0,1,0,1,1,0,1,1],
[1,1,0,1,1,1,1,1,1,1],
[1,1,1,1,1,1,0,0,1,1],
[1,1,1,0,1,1,1,1,1,1]
]



def add_legend(ax):

    obstacle = plt.Line2D(
        [0],[0],
        marker="s",
        color="black",
        markerfacecolor="black",
        markersize=10,
        linestyle="None",
        label="Obstacle"
    )

    free = plt.Line2D(
        [0],[0],
        marker="s",
        color="black",
        markerfacecolor="white",
        markersize=10,
        linestyle="None",
        label="Free Space"
    )

    agent = plt.Line2D(
        [0],[0],
        marker="o",
        color="red",
        markerfacecolor="red",
        markersize=10,
        linestyle="None",
        label="Agent"
    )

    target = plt.Line2D(
        [0],[0],
        marker="*",
        color="red",
        markerfacecolor="red",
        markersize=14,
        linestyle="None",
        label="Target"
    )

    ax.legend(
        handles=[obstacle, free, agent, target],
        loc="center left",
        bbox_to_anchor=(1,0.5),
        frameon=False
    )

def draw_grid(ax, grid):

    rows = len(grid)
    cols = len(grid[0])

    ax.set_facecolor("white")  # light background

    for r in range(rows):
        for c in range(cols):

            if grid[r][c] == 0:   # obstacle
                rect = patches.Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor="grey",
                    edgecolor="none"
                )
                ax.add_patch(rect)


def draw_targets(ax, targets):

    colors = ["red", "blue", "green"]
    target_list = []

    for i, (r, c) in enumerate(targets):

        star = ax.scatter(
            c + 0.5,
            r + 0.5,
            marker="*",    
            s=400,             
            color=colors[i],
            linewidths=1
        )

        target_list.append(star)

    return target_list


def create_agents(ax, start_positions):

    colors = ["red", "blue", "green"]
    agents = []

    for i, pos in enumerate(start_positions):
        r, c = pos

        circle = patches.Circle(
            (c + 0.4, r + 0.4),
            0.4,
            facecolor=colors[i],
            linewidth=1
        )

        ax.add_patch(circle)
        agents.append(circle)

    return agents


def animate_agents(grid, path):

    fig, ax = plt.subplots(figsize=(8,6))
    fig.subplots_adjust(right=0.8)

    draw_grid(ax, grid)
    draw_targets(ax, path[-1])
    add_legend(ax)
    agents = create_agents(ax, path[0])

    ax.set_xlim(0, len(grid[0]))
    ax.set_ylim(0, len(grid))
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)

    
    def update(frame):

        state = path[frame]

        for i, (r, c) in enumerate(state):
            agents[i].center = (c + 0.5, r + 0.5)

        return agents

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(path),
        interval=300,
        blit=True
    )
    ani.save("astar1.mp4", writer="ffmpeg", fps=3)
    plt.show()
s1, g1 = ((0, 0), (0, 6), (5, 3) ), ((0, 6), (0, 0), (0,3) )
s2, g2 = ((0,0), (9,0),(0,9)), ((9, 9), (0, 9), (9, 0))


path1, cost1 = astar(grid1, s1, g1, heuristic)
print("Cost of grid 1:", cost1)


# path2, cost2 = astar(grid2, s2, g2, heuristic)

# print("Cost of grid 2:", cost2)


# animate_agents(grid2, path2)

animate_agents(grid1, path1)