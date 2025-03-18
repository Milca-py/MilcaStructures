import numpy as np
from typing import Tuple, TYPE_CHECKING
from milcapy.utils import rotation_matrix

if TYPE_CHECKING:
    from milcapy.elements.element import Element


def integration_coefficients(element: "Element") -> np.ndarray:
    """
    Calcula los coeficientes de integración para un elemento.
    
    Args:
        element: Elemento estructural
        
    Returns:
        Array con los coeficientes de integración (C1, C2, C3, C4)
    """
    L = element.length                               # Longitud del elemento
    q_i = element.distributed_load.q_i               # Intensidad de la carga distribuida en el nodo i
    q_j = element.distributed_load.q_j               # Intensidad de la carga distribuida en el nodo j
    E = element.section.material.modulus_elasticity  # Módulo de elasticidad
    I = element.section.moment_of_inertia            # Inercia
    phi = element.shear_angle                        # Aporte por cortante
    u_i = element.displacement[1]                    # Desplazamiento en el GDL 2 nodo i
    u_j = element.displacement[4]                    # Desplazamiento en el GDL 5 nodo j
    theta_i = element.displacement[2]                # Ángulo de rotación en el GDL 3 nodo i
    theta_j = element.displacement[5]                # Ángulo de rotación en el GDL 6 nodo j
    
    A = -(q_j - q_i) / L
    B = -q_i
    
    M = np.array([
        [0,                     0,          0, 1],
        [0,                     0,          1, 0],
        [L**3 * (2 - phi) / 12, L**2 / 2,   L, 1],
        [L**2 / 2,              L,          1, 0],
    ])
    
    N = np.array([
        E * I * u_i,
        E * I * theta_i,
        E * I * u_j + A * L**5 * (0.6 - phi) / 72 + B * L**4 * (1 - phi) / 24,
        E * I * theta_j + A * L**4 / 24 + B * L**3 / 6,
    ])
    
    C = np.linalg.solve(M, N)
    
    return C


def axial_force(element: "Element", factor: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la fuerza axial a lo largo del elemento.
    
    Args:
        element: Elemento estructural
        factor: Factor de escala
        n: Número de puntos para discretizar
        
    Returns:
        Tupla de arrays con las coordenadas x y los valores de fuerza axial
    """
    axial_i = element.internal_forces[0]
    axial_j = element.internal_forces[3]
    
    length = element.length
    
    A = (axial_j + axial_i) / element.length
    B = axial_i
    
    x = np.linspace(0, length, n)
    N = - A*x + B
    
    return x, N * factor


def shear_force(element: "Element", factor: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la fuerza cortante a lo largo del elemento.
    
    Args:
        element: Elemento estructural
        factor: Factor de escala
        n: Número de puntos para discretizar
        
    Returns:
        Tupla de arrays con las coordenadas x y los valores de fuerza cortante
    """
    q_i = element.distributed_load.q_i
    q_j = element.distributed_load.q_j
    
    shear_i = element.internal_forces[1]
    
    length = element.length
    
    A = (q_j - q_i) / length
    B = q_i
    
    x = np.linspace(0, length, n)
    V = A * x**2 / 2 + B * x + shear_i
    
    return x, V * factor


def bending_moment(element: "Element", factor: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula el momento flector a lo largo del elemento.
    
    Args:
        element: Elemento estructural
        factor: Factor de escala
        n: Número de puntos para discretizar
        
    Returns:
        Tupla de arrays con las coordenadas x y los valores de momento flector
    """
    q_i = element.distributed_load.q_i
    q_j = element.distributed_load.q_j
    
    shear_i = element.internal_forces[1]
    moment_i = element.internal_forces[2]
    
    length = element.length
    
    A = (q_j - q_i) / length
    B = q_i
    
    x = np.linspace(0, length, n)
    M = -(A * x**3 / 6 + B * x**2 / 2 + shear_i * x - moment_i)
    
    return x, M * factor


def slope(element: "Element", factor: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la pendiente de la deformada a lo largo del elemento.
    
    Args:
        element: Elemento estructural
        factor: Factor de escala
        n: Número de puntos para discretizar
        
    Returns:
        Tupla de arrays con las coordenadas x y los valores de pendiente
    """
    q_i = element.distributed_load.q_i
    q_j = element.distributed_load.q_j
    
    length = element.length
    
    A = (q_j - q_i) / length
    B = q_i
    
    C1 = element.integration_coefficients[0]
    C2 = element.integration_coefficients[1]
    C3 = element.integration_coefficients[2]
    
    E = element.section.material.modulus_elasticity
    I = element.section.moment_of_inertia
    
    x = np.linspace(0, length, n)
    theta = (A * x**4 / 24 + B * x**3 / 6 + C1 * x**2 / 2 + C2 * x + C3) / (E * I)
    
    return x, theta * factor


def deflection(element: "Element", factor: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la deflexión a lo largo del elemento.
    
    Args:
        element: Elemento estructural
        factor: Factor de escala
        n: Número de puntos para discretizar
        
    Returns:
        Tupla de arrays con las coordenadas x y los valores de deflexión
    """
    q_i = element.distributed_load.q_i
    q_j = element.distributed_load.q_j
    
    L = element.length
    
    A = (q_j - q_i) / L
    B = q_i
    
    C1 = element.integration_coefficients[0]
    C2 = element.integration_coefficients[1]
    C3 = element.integration_coefficients[2]
    C4 = element.integration_coefficients[3]
    
    shear_angle = element.shear_angle  # aporte por cortante (phi)
    E = element.section.material.modulus_elasticity
    I = element.section.moment_of_inertia
    
    x = np.linspace(0, L, n)
    term1 = A * L**2 * x**3 * (0.6 * (x / L)**2 - shear_angle) / 72
    term2 = B * L**2 * x**2 * ((x / L)**2 - shear_angle) / 24
    term3 = C1 * x * L**2 * (2 * (x / L)**2 - shear_angle) / 12
    term4 = C2 * x**2 / 2 + C3 * x + C4
    
    u = (term1 + term2 + term3 + term4) / (E * I)
    return x, u * factor


def deformed(element: "Element", factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la forma deformada del elemento.
    
    Args:
        element: Elemento estructural
        factor: Factor de escala
        
    Returns:
        Tupla de arrays con las coordenadas x e y de la forma deformada
    """
    Lo = element.length
    # Cálculo de longitud deformada
    vertice_i = np.array(element.node_i.vertex.coordinates)  # (x_i, y_i)
    vertice_j = np.array(element.node_j.vertex.coordinates)  # (x_j, y_j)
    vertices = np.array([vertice_i, vertice_j]) - vertice_i
    vertices_local = np.dot(vertices, rotation_matrix(element.angle_x))

    # Obtener desplazamientos de los nodos (se asume [ux, uy, θ])
    u_i = np.array(element.node_i.displacement[:2])  # (u_xi, u_yi)
    u_j = np.array(element.node_j.displacement[:2])  # (u_xj, u_yj)

    displacement = np.array([u_i, u_j])
    desp_local = np.dot(displacement, rotation_matrix(element.angle_x)) * factor  # [ [ui, vi], [uj, vj] ]
    defor_local = vertices_local + desp_local  # [ [xdi, ydi], [xdj, ydj] ]

    # Agregamos deflexiones
    deflection = element.deflection * factor
    x = np.linspace(0, Lo, len(deflection))

    # Corrección de la deformada por deformación axial
    Lf = defor_local[1, 0] - defor_local[0, 0]
    x = x * Lf / Lo + desp_local[0, 0]

    # Rotar el vector de deflexiones
    deformada_local = np.column_stack((x, deflection))
    deformada_global = np.dot(deformada_local, rotation_matrix(element.angle_x).T) + vertice_i

    x_val = deformada_global[:, 0]
    y_val = deformada_global[:, 1]
    
    return x_val, y_val


def rigid_deformed(element: "Element", factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la forma deformada rígida del elemento.
    
    Args:
        element: Elemento estructural
        factor: Factor de escala
        
    Returns:
        Tupla de arrays con las coordenadas x e y de la forma deformada rígida
    """
    # Cálculo de longitud deformada
    vertice_i = np.array(element.node_i.vertex.coordinates)  # (x_i, y_i)
    vertice_j = np.array(element.node_j.vertex.coordinates)  # (x_j, y_j)


    # Obtener desplazamientos de los nodos
    u_i = np.array(element.node_i.displacement[:2])  # (u_xi, u_yi)
    u_j = np.array(element.node_j.displacement[:2])  # (u_xj, u_yj)

    # Dibujar la deformada rígida
    desp_rigid_i = vertice_i + u_i * factor
    desp_rigid_j = vertice_j + u_j * factor

    x_val = np.array([desp_rigid_i[0], desp_rigid_j[0]])
    y_val = np.array([desp_rigid_i[1], desp_rigid_j[1]])
    return x_val, y_val
