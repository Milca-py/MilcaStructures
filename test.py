import numpy as np
from core.system import SystemMilcaModel
from frontend.widgets.UIdisplay import create_plot_window

# --------------------------------------------------
# 1. Definición del modelo y secciones
# --------------------------------------------------

model = SystemMilcaModel()

model.add_material(
    name="concreto",
    modulus_elasticity=2.1e6,
    poisson_ratio=0.2
)

model.add_rectangular_section(
    name="seccion1",
    material_name="concreto",
    base=0.3,
    height=0.5
)
model.add_rectangular_section(
    name="seccion2",
    material_name="concreto",
    base=0.5,
    height=0.5
)
model.add_rectangular_section(
    name="seccion3",
    material_name="concreto",
    base=0.6,
    height=0.6
)

# --------------------------------------------------
# 2. Definición de nodos
# --------------------------------------------------
nodes = {
    1: (0, 0),
    2: (0, 5),
    3: (0, 8.5),
    4: (0, 12),
    5: (7, 0),
    6: (7, 5),
    7: (7, 8.5),
    8: (7, 12),
    # 9: (3.5, 15)  # Desactivado en este ejemplo
}

for key, value in nodes.items():
    model.add_node(key, value)

# --------------------------------------------------
# 3. Definición de elementos
# --------------------------------------------------
elements = {
    1: (1, 2, "seccion3"),
    2: (2, 3, "seccion3"),
    3: (3, 4, "seccion3"),
    4: (5, 6, "seccion2"),
    5: (6, 7, "seccion2"),
    6: (7, 8, "seccion2"),
    7: (2, 6, "seccion1"),
    8: (3, 7, "seccion1"),
    9: (4, 8, "seccion1"),
    # 10: (4, 9, "seccion1"),
    # 11: (8, 9, "seccion1")
}

for key, value in elements.items():
    model.add_element(key, *value)

# --------------------------------------------------
# 4. Restricciones y cargas
# --------------------------------------------------
model.add_restraint(1, (True, True, True))
model.add_restraint(5, (True, True, True))

model.add_load_pattern(name="Live Load")
model.add_point_load(2, "Live Load", "GLOBAL", 5, 0, 0)
model.add_point_load(3, "Live Load", "GLOBAL", 10, 0, 0)
model.add_point_load(4, "Live Load", "GLOBAL", 20, 0, 0)

model.add_distributed_load(7, "Live Load", "LOCAL", -5, -5)
model.add_distributed_load(8, "Live Load", "LOCAL", -2, -6)
model.add_distributed_load(9, "Live Load", "LOCAL", -4, -3)

# --------------------------------------------------
# 5. Resolución del modelo
# --------------------------------------------------
model.solve()

# --------------------------------------------------
# 6. Mostrar la estructura (opcional)
# --------------------------------------------------
# model.show_structure(show=False)
# También podrías usar los diagramas por defecto:
# model.plotter.show_diagrams(type="shear_force", show=False)
# model.plotter.show_deformed(escala=10, show=False)

# --------------------------------------------------
# 7. Función corregida para graficar la deformada de cada elemento
# --------------------------------------------------
def plot_element_deformed(element, ax, scale):
    """
    Grafica la forma deformada de 'element' sobre el eje 'ax',
    asumiendo que 'element.deflection' es la deflexión local en
    la dirección y local (sin desplazamiento axial).
    """
    # Coordenadas originales de los nodos
    vertice_i = element.node_i.vertex.coordinates  # (x_i, y_i)
    vertice_j = element.node_j.vertex.coordinates  # (x_j, y_j)
    
    # Desplazamientos globales en x,y del nodo i (se asume que el 3er dof es rotación)
    u_i = np.array(element.node_i.desplacement[:2])  # (u_xi, u_yi)
    u_j = np.array(element.node_j.desplacement[:2])  # (u_xj, u_yj)
    
    # Array con la deflexión transversal local a lo largo del elemento
    array_deflexion = element.deflection
    
    # Longitud y ángulo del elemento respecto al eje X global (en radianes)
    length = element.length
    angle = element.angle_x  # Se asume que ya está en radianes
    
    # Número de puntos en la discretización de la deflexión
    n_points = len(array_deflexion)
    
    # Coordenada local x a lo largo del elemento (0 a L)
    x_local = np.linspace(0, length, n_points)
    
    # Deflexión local (dirección y local)
    y_local = np.array(array_deflexion) * scale
    
    # Transformación a coordenadas globales
    #   X_global = x_local*cos(angle) - y_local*sin(angle)
    #   Y_global = x_local*sin(angle) + y_local*cos(angle)
    X_global = x_local * np.cos(angle) - y_local * np.sin(angle)
    Y_global = x_local * np.sin(angle) + y_local * np.cos(angle)
    
    # Se traslada la curva para que inicie en la posición global del nodo i (incluyendo su desplazamiento)
    origin = np.array(vertice_i) + u_i  # (x_i + u_xi, y_i + u_yi)
    X_global += origin[0]
    Y_global += origin[1]
    
    # Graficar la forma deformada del elemento en rojo
    ax.plot(X_global, Y_global, 'r-')

# def plot_deformed_element(element, ax, scale):
#     # Obtener datos del elemento
#     vertice_i = np.array(element.node_i.vertex.coordinates)
#     vertice_j = np.array(element.node_j.vertex.coordinates)
#     u_i = np.array(element.node_i.desplacement[:2])
#     u_j = np.array(element.node_j.desplacement[:2])
#     deflexiones = np.array(element.deflection)
#     angle = element.angle_x
#     L = element.length

#     # Crear coordenadas locales
#     n_points = len(deflexiones)
#     s = np.linspace(0, L, n_points)
    
#     # Matriz de rotación
#     c = np.cos(angle)
#     s_ang = np.sin(angle)
#     R = np.array([[c, -s_ang], 
#                  [s_ang, c]])
    
#     # Posiciones originales del elemento
#     original_points = np.column_stack((
#         vertice_i[0] + s * c,
#         vertice_i[1] + s * s_ang
#     ))

#     # Desplazamientos rígidos interpolados
#     weight_i = 1 - s/L
#     weight_j = s/L
#     rigid_disp = np.outer(weight_i, u_i) + np.outer(weight_j, u_j)

#     # Deformaciones locales transformadas y escaladas
#     local_def = np.column_stack((np.zeros_like(deflexiones), deflexiones * scale))
#     global_def = np.dot(local_def, R.T)

#     # Posición final deformada
#     deformed_points = original_points + rigid_disp + global_def

#     # Graficar
#     ax.plot(deformed_points[:, 0], deformed_points[:, 1], 
#             'r-', lw=1.5)  # Elemento deformado



import numpy as np
import matplotlib.pyplot as plt

def plot_deformed_element(element, ax, scale):
    """
    Grafica la forma deformada del elemento.
    
    Parámetros:
    - element: objeto que representa el elemento estructural.
    - ax: objeto de matplotlib para graficar.
    - scale: factor de escala para amplificar la deformación.
    """
    # Obtener coordenadas iniciales de los nodos
    vertice_i = np.array(element.node_i.vertex.coordinates)  # (x_i, y_i)
    vertice_j = np.array(element.node_j.vertex.coordinates)  # (x_j, y_j)

    # Obtener desplazamientos de los nodos (se asume [ux, uy, θ])
    u_i = np.array(element.node_i.desplacement[:2])  # (u_xi, u_yi)
    u_j = np.array(element.node_j.desplacement[:2])  # (u_xj, u_yj)

    # Deflexión local en la dirección y local
    deflexiones = np.array(element.deflection) * scale

    # Longitud del elemento y su ángulo de inclinación
    L = element.length
    angle = element.angle_x  # Se asume que ya está en radianes

    # Discretización de la longitud del elemento
    n_points = len(deflexiones)
    s = np.linspace(0, L, n_points)  # Coordenada local x

    # Matriz de rotación para transformar coordenadas locales a globales
    c = np.cos(angle)
    s_ang = np.sin(angle)
    R = np.array([[c, -s_ang], [s_ang, c]])  # Matriz de rotación

    # Coordenadas locales iniciales (posición en el eje del elemento)
    x_local = s  # El eje x local va de 0 a L
    y_local = deflexiones  # Deflexión en la dirección y local

    # Convertir las coordenadas deformadas a globales
    x_global = x_local * c - y_local * s_ang
    y_global = x_local * s_ang + y_local * c

    # Aplicar desplazamiento rígido interpolado
    weight_i = 1 - s / L
    weight_j = s / L
    rigid_disp_x = weight_i * u_i[0] + weight_j * u_j[0]
    rigid_disp_y = weight_i * u_i[1] + weight_j * u_j[1]

    # Aplicar desplazamientos globales
    x_final = vertice_i[0] + x_global + rigid_disp_x
    y_final = vertice_i[1] + y_global + rigid_disp_y

    # Graficar la deformada del elemento en rojo
    ax.plot(x_final, y_final, 'r-', lw=1.5)







# --------------------------------------------------
# 8. Graficar la deformada de cada elemento con la función corregida
# --------------------------------------------------
ax = model.plotter.axes[0]
scale = 40

# for element in model.element_map.values():
#     plot_element_deformed(element, ax, scale)


for element in model.element_map.values():
    plot_deformed_element(element, ax, scale)

# --------------------------------------------------
# 9. Mostrar la ventana con la figura
# --------------------------------------------------
root = create_plot_window(model.plotter.fig)
root.mainloop()
