import heapq
import matplotlib.pyplot as plt
import numpy as np

#Creating maze with obstacles
def create_maze():
    maze = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    return maze

def get_neighbors(node, maze):
    rows, cols = len(maze), len(maze[0])

    (row, col) = node
    neighbors = []

    directions = [(1, 0), (0, 1), (0, -1), (-1, 0)]
    for drow, dcol in directions:
        noder, nodec = row+drow, col+dcol

        if (0 <= noder < rows) and (0<= col + dcol < cols):
            if maze(noder, nodec) == 0:
                neighbors.append((noder, nodec))
    return neighbors

def get_heuristic(a, b):
    return abs(a[0]-b[0]) +abs(a[1]-b[1])

def a_star(maze, start, goal):
    pass


def reconstruct_path(parent, current):
    path = [current]









def plot_maze(maze, path=None):
    maze_array = np.array(maze)
    
    plt.imshow(maze_array, cmap="gray_r")
    
    if path:
        y_coords = [p[1] for p in path]
        x_coords = [p[0] for p in path]
        plt.plot(y_coords, x_coords)
    
    plt.show()


maze = create_maze()
#path = astar(maze, start, goal)

#print("Path:", path)

#plot_maze(maze, path)



def main():
    pass

