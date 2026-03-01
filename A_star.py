import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import matplotlib.pyplot as plt


mars_map = np.load('mars_map.npy')
nr, nc = mars_map.shape  
print(f"Dimensiones del mapa: {nr} filas x {nc} columnas")


escala = 10.0174  
max_diferencia_altura = 0.25  


x_inicio = 2850  
y_inicio = 6400  
x_meta = 3150    
y_meta = 6800    

print(f"\nCoordenadas en metros:")
print(f"Inicio: ({x_inicio}, {y_inicio})")
print(f"Meta: ({x_meta}, {y_meta})")



r_inicio = nr - round(y_inicio / escala)
c_inicio = round(x_inicio / escala)

r_meta = nr - round(y_meta / escala)
c_meta = round(x_meta / escala)

print(f"\nCoordenadas en píxeles:")
print(f"Inicio: (renglón={r_inicio}, columna={c_inicio})")
print(f"Meta: (renglón={r_meta}, columna={c_meta})")


assert 0 <= r_inicio < nr, f"Renglón inicio {r_inicio} fuera de límites (0-{nr-1})"
assert 0 <= c_inicio < nc, f"Columna inicio {c_inicio} fuera de límites (0-{nc-1})"
assert 0 <= r_meta < nr, f"Renglón meta {r_meta} fuera de límites (0-{nr-1})"
assert 0 <= c_meta < nc, f"Columna meta {c_meta} fuera de límites (0-{nc-1})"


altura_inicio = mars_map[r_inicio, c_inicio]
altura_meta = mars_map[r_meta, c_meta]

print(f"\nAlturas:")
print(f"Altura en inicio: {altura_inicio:.4f} m")
print(f"Altura en meta: {altura_meta:.4f} m")

if altura_inicio == -1:
    print("¡ADVERTENCIA! El punto de inicio está en un pixel no válido (altura=-1)")
if altura_meta == -1:
    print("¡ADVERTENCIA! El punto de meta está en un pixel no válido (altura=-1)")


cost_matrix = np.zeros((nr, nc), dtype=int)


cost_matrix[mars_map == -1] = 1

print(f"\nEstadísticas del mapa:")
print(f"Píxeles totales: {nr * nc}")
print(f"Píxeles no válidos (-1): {np.sum(mars_map == -1)}")
print(f"Píxeles válidos: {np.sum(mars_map != -1)}")


def es_movimiento_valido(altura_actual, altura_siguiente):
    """Verifica si el movimiento es válido según la diferencia de altura"""
    if altura_siguiente == -1:  # Pixel no válido
        return False
    diferencia = abs(altura_siguiente - altura_actual)
    return diferencia < max_diferencia_altura


class MarsGrid:
    def __init__(self, matrix, altura_matrix, max_diff):
        self.matrix = matrix
        self.alturas = altura_matrix
        self.max_diff = max_diff
        self.rows, self.cols = matrix.shape
        
    def node_walkable(self, x, y):
        """Verifica si un nodo es caminable"""
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return False
        # Primero verificar si no es obstáculo por valor -1
        if self.matrix[y, x] == 1:
            return False
        return True
    
    def can_move(self, from_x, from_y, to_x, to_y):
        """Verifica si se puede mover de un nodo a otro"""
        if not self.node_walkable(to_x, to_y):
            return False
        
        altura_actual = self.alturas[from_y, from_x]
        altura_siguiente = self.alturas[to_y, to_x]
        
        return es_movimiento_valido(altura_actual, altura_siguiente)


mars_grid = MarsGrid(cost_matrix, mars_map, max_diferencia_altura)


grid = Grid(matrix=cost_matrix)


start_node = grid.node(c_inicio, r_inicio)
end_node = grid.node(c_meta, r_meta)

print(f"\nNodos de inicio y meta:")
print(f"Inicio: ({start_node.x}, {start_node.y}) - Caminable: {start_node.walkable}")
print(f"Meta: ({end_node.x}, {end_node.y}) - Caminable: {end_node.walkable}")


finder = AStarFinder(diagonal_movement=DiagonalMovement.always)


def find_path_with_height_constraint():
    """Versión personalizada de A* que respeta las restricciones de altura"""
    
    import heapq
    
  
    open_set = []
    counter = 0
    heapq.heappush(open_set, (0, counter, (c_inicio, r_inicio)))
    
    
    came_from = {}
    g_score = {(c_inicio, r_inicio): 0}
    f_score = {(c_inicio, r_inicio): abs(c_meta - c_inicio) + abs(r_meta - r_inicio)}
    
 
    open_set_hash = {(c_inicio, r_inicio)}
    
   
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),  
        (1, 1), (1, -1), (-1, 1), (-1, -1)  ]
    
    iterations = 0
    
    while open_set:
        iterations += 1
        current = heapq.heappop(open_set)[2]
        open_set_hash.remove(current)
        
        cx, cy = current
        
        
        if current == (c_meta, r_meta):
           
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append((c_inicio, r_inicio))
            path.reverse()
            
            print(f"Iteraciones: {iterations}")
            return path
        
       
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            

            if nx < 0 or nx >= nc or ny < 0 or ny >= nr:
                continue
            
           
            if not mars_grid.can_move(cx, cy, nx, ny):
                continue
            
           
            if dx != 0 and dy != 0:
                move_cost = 1.414  
            else:
                move_cost = 1.0  
            
            tentative_g_score = g_score[current] + move_cost
            
            if (nx, ny) not in g_score or tentative_g_score < g_score[(nx, ny)]:
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = tentative_g_score
                f_score[(nx, ny)] = tentative_g_score + (abs(nx - c_meta) + abs(ny - r_meta))
                
                if (nx, ny) not in open_set_hash:
                    counter += 1
                    heapq.heappush(open_set, (f_score[(nx, ny)], counter, (nx, ny)))
                    open_set_hash.add((nx, ny))
    
    return None  


print("\nBuscando camino con A* (respetando diferencias de altura)...")
path = find_path_with_height_constraint()

if path:
    print(f"¡Camino encontrado! Longitud: {len(path)} pasos")
    
    
    distancia_total = 0
    alturas_recorrido = []
    
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        
        
        if x1 != x2 and y1 != y2:
            dist_pixeles = np.sqrt(2) 
        else:
            dist_pixeles = 1.0  
        
       
        dist_metros = dist_pixeles * escala
        distancia_total += dist_metros
        
        
        alturas_recorrido.append(mars_map[y1, x1])
    
    alturas_recorrido.append(mars_map[path[-1][1], path[-1][0]])
    
    print(f"\nResultados del camino:")
    print(f"Número de pasos: {len(path)}")
    print(f"Distancia total recorrida: {distancia_total:.2f} metros")
    print(f"Distancia en línea recta: {np.sqrt((x_meta-x_inicio)**2 + (y_meta-y_inicio)**2):.2f} metros")
    print(f"Altura mínima en el camino: {min(alturas_recorrido):.4f} m")
    print(f"Altura máxima en el camino: {max(alturas_recorrido):.4f} m")
    print(f"Diferencia de altura total: {max(alturas_recorrido) - min(alturas_recorrido):.4f} m")
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    im1 = ax1.imshow(mars_map, cmap='terrain', origin='upper')
    ax1.set_title('Mapa de Alturas de Marte\n(con camino A*)')
    ax1.set_xlabel('Columna (píxeles)')
    ax1.set_ylabel('Renglón (píxeles)')
    plt.colorbar(im1, ax=ax1, label='Altura (metros)')
    
   
    ax1.plot(c_inicio, r_inicio, 'go', markersize=8, label='Inicio')
    ax1.plot(c_meta, r_meta, 'ro', markersize=8, label='Meta')
    
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax1.plot(path_x, path_y, 'y-', linewidth=2, label='Camino A*')
    ax1.legend()
    
   
    obstacle_map = np.zeros_like(mars_map)
    obstacle_map[mars_map == -1] = 1  
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if not mars_grid.can_move(x1, y1, x2, y2):
            obstacle_map[y2, x2] = 2  
    
    im2 = ax2.imshow(obstacle_map, cmap='RdYlGn_r', origin='upper', 
                     vmin=0, vmax=2)
    ax2.set_title('Mapa de Obstáculos\n(Verde=Transitable, Rojo=Obstáculo)')
    ax2.set_xlabel('Columna (píxeles)')
    ax2.set_ylabel('Renglón (píxeles)')
    

    ax2.plot(c_inicio, r_inicio, 'go', markersize=8, label='Inicio')
    ax2.plot(c_meta, r_meta, 'ro', markersize=8, label='Meta')
    
    
    ax2.plot(path_x, path_y, 'b-', linewidth=2, label='Camino A*')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    
    path_array = np.array(path)
    np.save('camino_a_star.npy', path_array)
    print("\nCamino guardado en 'camino_a_star.npy'")
    
   
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(alturas_recorrido)), alturas_recorrido, 'b-', linewidth=2)
    plt.fill_between(range(len(alturas_recorrido)), alturas_recorrido, alpha=0.3)
    plt.xlabel('Paso en el camino')
    plt.ylabel('Altura (metros)')
    plt.title('Perfil de Alturas a lo largo del Camino')
    plt.grid(True, alpha=0.3)
    

    plt.plot(0, alturas_recorrido[0], 'go', markersize=10, label='Inicio')
    plt.plot(len(alturas_recorrido)-1, alturas_recorrido[-1], 'ro', markersize=10, label='Meta')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
else:
    print("No se encontró un camino que respete las restricciones de altura.")
    print("Probando con umbral de altura más permisivo...")
    
    
    max_diferencia_altura_temp = 0.5
    print(f"Usando diferencia máxima de {max_diferencia_altura_temp} metros")
    
    
    mars_grid.max_diff = max_diferencia_altura_temp
    
 
    path = find_path_with_height_constraint()
    
    if path:
        print("¡Camino encontrado con umbral más permisivo!")
    else:
        print("Todavía no se encuentra camino. Los puntos pueden estar aislados.")