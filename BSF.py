from math import dist
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma


class Node:
    def __init__(self, x, y):
        self.height = None
        self.heuristic = None
        self.parent = None
        self.cost = None
        self.x = x
        self.y = y

    def set_cost(self, cost):
        self.cost = cost

    def set_heuristic(self, heuristic):
        self.heuristic = heuristic

    def set_height(self, height):
        self.height = height

    def set_parent(self, parent):
        self.parent = parent


class Grid:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

        grid = []
        for y in range(size_y):
            grid.append([Node(x, y) for x in range(size_x)])

        self.grid = grid

    def get_node(self, x, y):
        return self.grid[y][x]

    def set_heights(self, height_map):
        if len(height_map) != self.size_y or len(height_map[0]) != self.size_x:
            raise Exception('The height map and grid sizes don\'t match')

        for y in range(self.size_y):
            for x in range(self.size_x):
                self.grid[y][x].set_height(height_map[y][x])

    def distance_heuristic(self, x_goal, y_goal):
        for y in range(self.size_y):
            for x in range(self.size_x):
                self.grid[y][x].set_heuristic(dist((x, y), (x_goal, y_goal)))

    def expand(self, node, max_height_delta=0.25):
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)]
        neighbors = []
        x = node.x
        y = node.y

        if node.cost is None:
            raise Exception(f'Node ({x}, {y}) hasn\'t been explored')

        for direction in directions:
            nx = x + direction[0]
            ny = y + direction[1]

            if nx < 0 or ny < 0 or nx >= self.size_x or ny >= self.size_y:
                print(f'{nx}, {ny}')
                continue

            neighbor = self.grid[ny][nx]

            if neighbor.height == -1:
                continue

            if neighbor.parent is None or neighbor.cost > node.cost + dist((x, y), (nx, ny)):
                if abs(node.height - neighbor.height) < max_height_delta:
                    neighbor.parent = node
                    neighbor.cost = node.cost + dist((x, y), (nx, ny))
                    neighbors.append(neighbor)

        return neighbors


def dsf_search(start, goal, grid):

    start_x, start_y = start
    goal_x, goal_y = goal

    reached = np.zeros((grid.size_y, grid.size_x), dtype=bool)
    node = grid.get_node(start_x, start_y)
    node.cost = 0
    frontier = grid.expand(node)

    steps = 0
    while node.x != goal_x or node.y != goal_y:
        steps += 1
        try:
            node = frontier.pop(-1)
        except IndexError:
            print("Couldn't find path")
            break
        reached[node.y][node.x] = True
        frontier += grid.expand(node)

    path_cost = node.cost
    print(f'Found path in {steps} steps \npath length: {path_cost:.2f} \ncart distance: {dist((start_x, start_y), (goal_x, goal_y)):.2f}')

    path_x = []
    path_y = []
    while node.x != start_x or node.y != start_y:
        path_x.append(node.x)
        path_y.append(node.y)
        node = node.parent

    return (path_x, path_y), reached, path_cost


def bsf_search(start, goal, grid):

    start_x, start_y = start
    goal_x, goal_y = goal

    reached = np.zeros((grid.size_y, grid.size_x), dtype=bool)
    node = grid.get_node(start_x, start_y)
    node.cost = 0
    frontier = grid.expand(node)

    steps = 0
    while node.x != goal_x or node.y != goal_y:
        steps += 1
        try:
            node = frontier.pop(0)
        except IndexError:
            print("Couldn't find path")
            break
        reached[node.y][node.x] = True
        frontier += grid.expand(node)

    path_cost = node.cost
    print(f'Found path in {steps} steps \npath length: {path_cost:.2f} \ncart distance: {dist((start_x, start_y), (goal_x, goal_y)):.2f}')

    path_x = []
    path_y = []
    while node.x != start_x or node.y != start_y:
        path_x.append(node.x)
        path_y.append(node.y)
        node = node.parent

    return (path_x, path_y), reached, path_cost


def astar_search(start, goal, grid):

    start_x, start_y = start
    goal_x, goal_y = goal

    reached = np.zeros((grid.size_y, grid.size_x), dtype=bool)
    node = grid.get_node(start_x, start_y)
    node.cost = 0
    frontier = grid.expand(node)

    steps = 0
    while node.x != goal_x or node.y != goal_y:
        steps += 1
        try:
            node = frontier.pop(0)
        except IndexError:
            print("Couldn't find path")
            break
        reached[node.y][node.x] = True
        frontier += grid.expand(node)

    path_cost = node.cost
    print(f'Found path in {steps} steps \npath length: {path_cost:.2f} \ncart distance: {dist((start_x, start_y), (goal_x, goal_y)):.2f}')

    path_x = []
    path_y = []
    while node.x != start_x or node.y != start_y:
        path_x.append(node.x)
        path_y.append(node.y)
        node = node.parent

    return (path_x, path_y), reached, path_cost


mars_map = np.load('mars_map.npy')
yn, xn = mars_map.shape
grid = Grid(xn, yn)
grid.set_heights(mars_map)

scale = 10.0174

# goal_x = round(2850/scale)
# goal_y = size_y - round(6400/scale)

goal = (65, 450)
start = (round(3150/scale), yn - round(6800/scale))

path, reached, length = dsf_search(start, goal, grid)
plt.imshow(mars_map, cmap='viridis', interpolation='none')
plt.imshow(ma.masked_where(reached, mars_map), cmap='gray', interpolation='none')
plt.scatter(goal[0], goal[1], color='red')
plt.scatter(start[0], start[1], color='green')
plt.plot(path[0], path[1], color='black')
plt.show()
