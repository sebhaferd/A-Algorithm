import networkx as nx
import heapq
import random

#A* search algorithm using weighted graphs
#graph test


#heuristic for dijkstras algorithm
def heuristic(n, target):
    return 0

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
def astar(graph, start_state, end_state, heuristic):
    #Set state as tuple of two nodes
    source1, source2 = start_state 
    (target1, target2) = end_state

    #check if already at goal
    if start_state == end_state:
        return [start_state], 0
    
    #create min heap queue to store nodes in path and h and g values
    open_list = []
    heapq.heappush(open_list, ((heuristic(source1, target1)+heuristic(source2, target2), start_state)))

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
        neighbors1 = graph.get(state[0], []) + [(state[0], 0)]
        neighbors2 = graph.get(state[1], []) + [(state[1], 0)]

        #nested loop to avoid collisions between agents
        for neighbor1, weight1 in neighbors1:
            for neighbor2, weight2 in neighbors2:
                #check if on same node
                if neighbor1 == neighbor2:
                    continue
                #check if exploring same edge
                if state[0] == neighbor2 and neighbor1 == state[1]:
                    continue

                #set new state as shortest distance of each
                next_state = (neighbor1, neighbor2)

                #total g value of operatoin
                temp_cost = costs[state] + weight1 + weight2

                #check if new path or if more efficient than previously explored path
                if next_state not in costs or temp_cost < costs[next_state]:
                    costs[next_state] = temp_cost
                    f_cost = temp_cost + heuristic(neighbor1, target1)+heuristic(neighbor2, target2)
                    heapq.heappush(open_list, (f_cost, next_state))
                    path[next_state] = state

    #all paths explored and no solution, distance to goal is infinity
    return None, float('inf')

#Creating graphs
#simple graph
def create_graph1():
    graph = {
        'A': [('B', 2), ('C', 5)],
        'B': [('C', 1), ('D', 4)],
        'C': [('D', 1)],
        'D': []
    }
    return graph
#simple graph 2
def create_graph2():
    graph = {
        'A': [('B', 3), ('C', 1)],
        'B': [('D', 2), ('E', 3)],
        'C': [('F', 4)],
        'D': [('G', 1)],
        'E': [('G', 2)],
        'F': [('G', 5)],
        'G': []
    }
    return graph

#unreachable target
def create_graph3():
    graph = {
        'A': [('B', 3), ('C', 1), ('E', 4)], 
        'B': [('E', 2), ('C', 2)],
        'C': [('E', 4)],
        'D': [('G', 2)],
        'E': [], 
        'G': []
    }
    return graph
#longer graph
def create_graph4():
    graph = {
        'A': [('B', 3), ('C', 1), ('D', 5)],
        'B': [('D', 2), ('E', 3), ('G', 12)],
        'C': [('F', 4), ('E', 4)],
        'D': [('G', 5)],
        'E': [('G', 2), ('F', 5)],
        'F': [('G', 3), ('I', 6)],
        'G': [('K', 6), ('H', 3)], 
        'H': [('I', 4), ('K', 5)], 
        'I':[('J', 4)], 
        'J': [('L', 3)], 
        'K':[('L', 2)], 
        'L': []

    }
    return graph

#random generated graphs with random weights from networkx module
g5 = nx.gnp_random_graph(10, 0.3, directed=True)
for node, neighbor in g5.edges():
    g5[node][neighbor]['weight'] = random.randint(1, 10)

g6 = nx.gnp_random_graph(15, 0.3, directed=True)
for node, neighbor in g6.edges():
    g6[node][neighbor]['weight'] = random.randint(1, 10)

g7 = nx.gnp_random_graph(20, 0.3, directed=True)
for node, neighbor in g7.edges():
    g7[node][neighbor]['weight'] = random.randint(1, 10)





#test cases with two agents:

def test_graph1(graph, start_state, end_state):
    my_path, my_cost = astar(graph, start_state, end_state, heuristic)
    print(my_cost)
    return my_cost

g1, g2, g4 = create_graph1(), create_graph2(), create_graph4()
g3 = create_graph3()

start1 = ('A', 'B')
goal1 = ('C', 'D')
start2 = ('A', 'B')
goal2 = ('G', 'E')
start3 = ('B', 'A')
goal3 = ('G', 'D')
start4 = ('A', 'D')
goal4 = ('L', 'G')

test_graph1(g1, start1, goal1)
test_graph1(g2, start1, goal1)
test_graph1(g3, start1, goal1)
test_graph1(g4, start1, goal1)


#single agent test case

# #convert nx graph to adj list
# def convert_to_adj(graph):
#     result = {}
#     for node in graph.nodes():
#         result[node] = []
#         for neighbor in graph.successors(node):
#             weight = graph[node][neighbor]['weight']
#             result[node].append((neighbor, weight))
#     return result

# #convert adjlist to nx format
# def convert_to_nx(graph):
#     nxGraph = nx.DiGraph()
#     for node in graph:
#         for neighbor, weight in graph[node]:
#             nxGraph.add_edge(node, neighbor, weight=weight)
#     return nxGraph

# def test_graph2(graph, source, target):
#     nxGraph = convert_to_nx(graph)
#     try:
#         nx_cost = nx.dijkstra_path_length(nxGraph, source, target)
#     except nx.NetworkXNoPath:
#         nx_cost = float('inf')

#     my_path, my_cost = astar(graph, source, target, heuristic)

#     assert my_cost == nx_cost
#     print(my_cost)
#     return my_cost


# g1, g2, g4 = create_graph1(), create_graph2(), create_graph4()
# g3 = create_graph3()

# test_graph1(g1, 'A', 'D')
# test_graph1(g2, 'A', 'G')
# test_graph1(g3, 'A', 'G')
# test_graph1(g4, 'A', 'L')
# test_graph1(convert_to_adj(g5), 0, 9)
# test_graph1(convert_to_adj(g6), 0, 14)
# test_graph1(convert_to_adj(g7), 0, 19)






