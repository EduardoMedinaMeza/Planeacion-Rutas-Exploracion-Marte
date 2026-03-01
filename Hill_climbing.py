import numpy as np
import matplotlib.pyplot as plt
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid

mars_map = np.load('mars_map.npy')
nr, nc = mars_map.shape
print(f"Dimensiones del mapa: {nr} filas x {nc} columnas")


escala = 10.0174
max_diferencia_altura = 0.3

x_inicio = 2850
y_inicio = 6400
x_meta = 3150
y_meta = 6800

print(f"\nCoordenadas en metros:")
print(f"Inicio: ({x_inicio}, {y_inicio})")
print(f"Meta: ({x_meta}, {y_meta})")

r_inicio = nr - round(y_inicio/escala)
c_inicio = round(x_inicio/escala)


r_meta = nr - round(y_meta/escala)
c_meta = round(x_meta/escala)

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


