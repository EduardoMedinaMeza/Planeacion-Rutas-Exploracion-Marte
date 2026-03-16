import numpy as np
from math import dist
import matplotlib.pyplot as plt
import numpy.ma as ma



class Node:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.height = None
        self.heuristic = None
        
    def set_height(self, height):
        self.height = height
    def set_heuristic(self, heuristic):
        self.heuristic = heuristic
    
    
class Grid:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        
        grid = []
        for y in range(size_y):
            grid.append([Node(x,y) for x in range(size_x)])
        
        self.grid = grid
    
    def get_node(self,x,y):
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
                self.grid[y][x].set_heuristic(dist((x,y),(x_goal,y_goal)))
    def expand(self, node):
        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        neighbors = []
        x = node.x
        y = node.y
        for direction in directions:
            new_x = x + direction[0]
            new_y = y + direction[1]
            if new_x >= 0 and new_x < self.size_x and new_y >= 0 and new_y < self.size_y:
                neighbors.append(self.grid[new_y][new_x])
        
        return neighbors
mars_map = np.load("mars_map.npy")
size_y, size_x = mars_map.shape


grid = Grid(size_x, size_y)
grid.set_heights(mars_map)


scale = 10.0174
max_diferencia_altura = 0.25


x_inicio, y_inicio = 2850, 6400
x_meta, y_meta = 3150, 6800


c_inicio = round(x_inicio / scale)
r_inicio = size_y - round(y_inicio / scale)

c_meta = round(x_meta / scale)
r_meta = size_y - round(y_meta / scale)


start_node = grid.get_node(c_inicio, r_inicio)
goal_node = grid.get_node(c_meta, r_meta)


grid.distance_heuristic(c_meta, r_meta)

def hill_climbing(start_node, goal_node):
    current_node = start_node
    path = [current_node]
    
    while current_node != goal_node:
        neighbors = grid.expand(current_node)
        next_node = None
        
        for neighbor in neighbors:
            if neighbor.height == -1:
                continue
            
            if abs(neighbor.height - current_node.height) > max_diferencia_altura:
                continue
            
            if next_node is None or neighbor.heuristic < next_node.heuristic:
                next_node = neighbor
        
        if next_node is None or next_node.heuristic >= current_node.heuristic:
            break
        
        current_node = next_node
        path.append(current_node)
    
    return path


# Ejecutar Hill Climbing
path = hill_climbing(start_node, goal_node)
print(f"Camino encontrado con {len(path)} pasos")

# Preparar datos para visualización
reached = np.zeros((size_y, size_x), dtype=bool)
path_x = []
path_y = []

for node in path:
    path_x.append(node.x)
    path_y.append(node.y)
    reached[node.y, node.x] = True

# Calcular distancia recorrida
distancia_total = 0
for i in range(len(path)-1):
    dx = abs(path[i+1].x - path[i].x)
    dy = abs(path[i+1].y - path[i].y)
    if dx != 0 and dy != 0:
        distancia_total += np.sqrt(2) * scale  # diagonal
    else:
        distancia_total += 1 * scale  # cardinal
print(f"Distancia total recorrida: {distancia_total:.2f} metros")

# Visualización
plt.figure(figsize=(12, 10))

# Mostrar mapa de alturas
plt.imshow(mars_map, cmap='magma', interpolation='none', alpha=0.7)
plt.imshow(ma.masked_where(reached, mars_map), cmap='gray', alpha=0.3)

# Marcar inicio y meta
plt.scatter(c_meta, r_meta, color='red', s=100, label='Meta', edgecolors='white')
plt.scatter(c_inicio, r_inicio, color='green', s=100, label='Inicio', edgecolors='white')

# Dibujar camino
plt.plot(path_x, path_y, 'cyan', linewidth=2, label='Camino Hill Climbing')

plt.colorbar(label='Altura (metros)')
plt.title(f'Hill Climbing - Distancia: {distancia_total:.2f} metros')
plt.xlabel('Columna (píxeles)')
plt.ylabel('Renglón (píxeles)')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

# Mostrar información adicional
print(f"Coordenadas de inicio (píxeles): ({c_inicio}, {r_inicio})")
print(f"Coordenadas de meta (píxeles): ({c_meta}, {r_meta})")
print(f"Altura en inicio: {start_node.height:.4f} m")
print(f"Altura en meta: {goal_node.height:.4f} m")