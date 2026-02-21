import networkx as nx
import heapq
import random

#A* search algorithm using weighted graphs
#graph test


#heuristic for dijkstras algorithm
def heuristic(n, target):
    return 0

#create path from source to target of each node
def make_path(path, source, target):
    res = []
    current = target
    if target not in path and target != source:
        return None
    while current != source:
        res.append(current)
        current = path[current]

    res.append(source)
    res.reverse()
    return res
    
#A* search alrogithm, calculating through distance to each node finding the shortest path
def astar(graph, source, target, heuristic):
    #check if at target
    if source == target:
        return [source], 0
    
    #create heap queue to store nodes in path and h and g values
    min_queue = []
    heapq.heappush(min_queue, ((heuristic(source, target)), source))

    costs = {}
    costs[source] = 0
    path = {}
    closed = set()

    while min_queue:
        #pop smallest value from min 
        f_cost, node = heapq.heappop(min_queue)

        #node already explored 
        if node in closed:
            continue

        #reached target, return path
        if node == target:
            return make_path(path, source, target), costs[target]
        
        closed.add(node)
        
        for neighbor, weight in graph.get(node, []):
            temp_cost = costs[node] + weight

            if neighbor not in costs or temp_cost < costs[neighbor]:
                costs[neighbor] = temp_cost
                f = temp_cost + heuristic(neighbor, target)
                heapq.heappush(min_queue, (f, neighbor))
                path[neighbor] = node

    return None, float('inf')



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


#convert nx graph to adj list
def convert_to_adj(graph):
    result = {}
    for node in graph.nodes():
        result[node] = []
        for neighbor in graph.successors(node):
            weight = graph[node][neighbor]['weight']
            result[node].append((neighbor, weight))
    return result

#convert adjlist to nx format
def convert_to_nx(graph):
    nxGraph = nx.DiGraph()
    for node in graph:
        for neighbor, weight in graph[node]:
            nxGraph.add_edge(node, neighbor, weight=weight)
    return nxGraph


def test_graph(graph, source, target):
    nxGraph = convert_to_nx(graph)
    try:
        nx_cost = nx.dijkstra_path_length(nxGraph, source, target)
    except nx.NetworkXNoPath:
        nx_cost = float('inf')

    my_path, my_cost = astar(graph, source, target, heuristic)

    assert my_cost == nx_cost
    print(my_cost)
    return my_cost


g1, g2, g4 = create_graph1(), create_graph2(), create_graph4()
g3 = create_graph3()

test_graph(g1, 'A', 'D')
test_graph(g2, 'A', 'G')
test_graph(g3, 'A', 'G')
test_graph(g4, 'A', 'L')
test_graph(convert_to_adj(g5), 0, 9)
test_graph(convert_to_adj(g6), 0, 14)
test_graph(convert_to_adj(g7), 0, 19)






