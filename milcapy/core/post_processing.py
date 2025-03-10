from typing import List, Tuple, Dict, Union, TYPE_CHECKING
from utils import rotation_matrix
import numpy as np

if TYPE_CHECKING:
    from core.system import SystemMilcaModel
    from core.element import Element
    from utils import ElementType

class PostProcessingOptions:
    def __init__(self, factor: float, n: int):
        self.factor = factor
        self.n = n

class PostProcessing:
    def __init__(self, system: "SystemMilcaModel",
                options: "PostProcessingOptions" = PostProcessingOptions(factor=1, n=100)
                ) -> None:
        self.system = system
        self.results = self.system.results
        self.options = options
        
        # Diccionarios de resultados
        self.integration_coefficients_elements = {}
        self.values_axial_force_elements = {}
        self.values_shear_force_elements = {}
        self.values_bending_moment_elements = {}
        self.values_slope_elements = {}
        self.values_deflection_elements = {}
        self.values_deformed_elements = {}
        
        # Calcular resultados para cada elemento
        self.process_all_elements()
    
    def process_all_elements(self) -> None:
        """Calcula todos los resultados para cada elemento."""
        factor = self.options.factor
        n = self.options.n
        
        for element in self.system.element_map.values():
            # Calcular coeficientes de integración
            self.integration_coefficients_elements[element.id] = integration_coefficients(element)
            element.integration_coefficients = self.integration_coefficients_elements[element.id]
            
            # Calcular fuerzas y momentos
            self.values_axial_force_elements[element.id] = values_axial_force(element, factor, n)
            element.axial_force = self.values_axial_force_elements[element.id][1]
            
            self.values_shear_force_elements[element.id] = values_shear_force(element, factor, n)
            element.shear_force = self.values_shear_force_elements[element.id][1]
            
            self.values_bending_moment_elements[element.id] = values_bending_moment(element, factor, n)
            element.bending_moment = self.values_bending_moment_elements[element.id][1]
            
            
            # Calcular deformaciones
            self.values_slope_elements[element.id] = values_slope(element, factor, n)
            element.slope = self.values_slope_elements[element.id][1]
            
            self.values_deflection_elements[element.id] = values_deflection(element, factor, n)
            element.deflection = self.values_deflection_elements[element.id][1]
            
            self.values_deformed_elements[element.id] = values_deformed(element, factor)
            element.deformed_shape = self.values_deformed_elements[element.id][1]
    
    def update_with_new_options(self, new_options: "PostProcessingOptions") -> None:
        """Actualiza las opciones y recalcula todos los resultados."""
        self.options = new_options
        self.process_all_elements()
        self.assign_results_to_elements()


def integration_coefficients(element: "Element") -> np.ndarray:
    "Coeficientes de integracion."
    L = element.length                              # Longitud del elemento.
    q_i = element.distributed_load.q_i              # Intensidad de la carga distribuida en el nodo i.
    q_j = element.distributed_load.q_j              # Intensidad de la carga distribuida en el nodo j.
    E = element.section.material.modulus_elasticity # Modulo de elasticidad.
    I = element.section.moment_of_inertia           # Inercia.
    phi = element.shear_angle                       # Aporte por cortante.
    u_i = element.desplacement[1]                   # Desplazamiento en el GDL 2 nodo i.
    u_j = element.desplacement[4]                   # Desplazamiento en el GDL 5 nodo j.
    theta_i = element.desplacement[2]               # Angulo de rotacion en el GDL 3 nodo i.
    theta_j = element.desplacement[5]               # Angulo de rotacion en el GDL 6 nodo j.
    
    A = -(q_j - q_i) / L
    B = -q_i
    
    M = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [L**3*(2-phi)/12, L**2/2, L, 1],
        [L**2/2, L, 1, 0],
    ])
    
    N = np.array([
        E*I*u_i,
        E*I*theta_i,
        E*I*u_j + A*L**5*(0.6-phi)/72 + B*L**4*(1-phi)/24,
        E*I*theta_j + A*L**4/24 + B*L**3/6,
    ])
    
    C = np.linalg.solve(M, N)
    
    return C

def values_axial_force(
    element: "Element", factor: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    axial_i = element.internal_forces[0]
    axial_j = element.internal_forces[3]
    
    length = element.length
    
    A = (axial_j + axial_i) / element.length
    B = axial_i
    
    x = np.linspace(0, length, n)
    N = - A*x + B
    
    return x, N*factor

def values_shear_force(
    element: "Element", factor: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    q_i = element.distributed_load.q_i
    q_j = element.distributed_load.q_j
    
    shear_i = element.internal_forces[1]
    
    length = element.length
    
    A = (q_j - q_i) / length
    B = q_i
    
    x = np.linspace(0, length, n)
    V = A*x**2/2 + B*x + shear_i
    
    return x, V*factor

def values_bending_moment(
    element: "Element", factor: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    q_i = element.distributed_load.q_i
    q_j = element.distributed_load.q_j
    
    shear_i = element.internal_forces[1]
    moment_i = element.internal_forces[2]
    
    length = element.length
    
    A = (q_j - q_i) / length
    B = q_i
    
    x = np.linspace(0, length, n)
    M = - ( A*x**3/6 + B*x**2/2 + shear_i*x - moment_i )
    
    return x, M*factor

def values_slope(
    element: "Element", factor: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    q_i = element.distributed_load.q_i
    q_j = element.distributed_load.q_j
    
    length = element.length
    
    A = -(q_j - q_i) / length
    B = -q_i
    
    C1 = element.integration_coefficients[0]
    C2 = element.integration_coefficients[1]
    C3 = element.integration_coefficients[2]
    
    E = element.section.material.modulus_elasticity
    I = element.section.moment_of_inertia
    
    x = np.linspace(0, length, n)
    theta = (- A*x**4/24 - B*x**3/6 + C1*x**2/2 + C2*x + C3) / (E*I)
    
    return x, theta*factor

def values_deflection(
    element: "Element", factor: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    q_i = element.distributed_load.q_i
    q_j = element.distributed_load.q_j
    
    L = element.length
    
    A = -(q_j - q_i) / L
    B = -q_i
    
    C1 = element.integration_coefficients[0]
    C2 = element.integration_coefficients[1]
    C3 = element.integration_coefficients[2]
    C4 = element.integration_coefficients[3]
    
    shear_angle = element.shear_angle # aporte por cortante (phi)
    E = element.section.material.modulus_elasticity
    I = element.section.moment_of_inertia
    
    x = np.linspace(0, L, n)
    u = (- A*L**2*x**3*(0.6*(x/L)**2 - shear_angle) / 72 - B*L**2*x**2*((x/L)**2 - shear_angle) / 24 + C1*x*L**2*(2*(x/L)**2 - shear_angle) / 12 + C2*x**2/2 + C3*x + C4) / (E*I)
    return x, u*factor


def values_deformed(element: "Element", factor: int) -> Tuple[np.ndarray, np.ndarray]:
    Lo = element.length
    # calculo de longitud deformada
    vertice_i = np.array(element.node_i.vertex.coordinates)  # (x_i, y_i)
    vertice_j = np.array(element.node_j.vertex.coordinates)  # (x_j, y_j)
    vertices = np.array([vertice_i, vertice_j]) - vertice_i
    vertices_local = np.dot(vertices, rotation_matrix(element.angle_x))

    # Obtener desplazamientos de los nodos (se asume [ux, uy, θ])
    u_i = np.array(element.node_i.desplacement[:2]) # (u_xi, u_yi)
    u_j = np.array(element.node_j.desplacement[:2]) # (u_xj, u_yj)

    desplacement = np.array([u_i, u_j])
    desp_local = np.dot(desplacement, rotation_matrix(element.angle_x)) * factor # [ [ui, vi], [uj, vj] ]
    defor_local = vertices_local + desp_local # [ [xdi, ydi], [xdj, ydj] ]

    # AGREGAMOS DEFLEXIONES
    deflection = element.deflection * factor
    x = np.linspace(0, Lo, len(deflection))

    # correccion de la deformada por deformacion axial
    Lf = defor_local[1, 0] - defor_local[0, 0]
    x = x * Lf / Lo + desp_local[0, 0]

    #  2. ROTAR EL VECTOR DE DEFLEXIONES
    deformada_local = np.column_stack((x, deflection))
    deformada_global = np.dot(deformada_local, rotation_matrix(element.angle_x).T) + vertice_i

    x_val = deformada_global[:, 0]
    y_val = deformada_global[:, 1]
    
    return x_val, y_val

def values_rigid_deformed(element: "Element", factor: int) -> Tuple[np.ndarray, np.ndarray]:
    # calculo de longitud deformada
    vertice_i = np.array(element.node_i.vertex.coordinates)  # (x_i, y_i)
    vertice_j = np.array(element.node_j.vertex.coordinates)  # (x_j, y_j)
    vertices = np.array([vertice_i, vertice_j]) - vertice_i
    vertices_local = np.dot(vertices, rotation_matrix(element.angle_x))

    # Obtener desplazamientos de los nodos (se asume [ux, uy, θ])
    u_i = np.array(element.node_i.desplacement[:2]) # (u_xi, u_yi)
    u_j = np.array(element.node_j.desplacement[:2]) # (u_xj, u_yj)
    desplacement = np.array([u_i, u_j])
    desp_local = np.dot(desplacement, rotation_matrix(element.angle_x)) * factor # [ [ui, vi], [uj, vj] ]
    defor_local = vertices_local + desp_local # [ [xdi, ydi], [xdj, ydj] ]

    # 5. DIBUJAR LA DEFORMADA RIGIDA
    desp_regida_local = np.array([defor_local[0], defor_local[1]])
    desp_regida_global = np.dot(desp_regida_local, rotation_matrix(element.angle_x).T) + vertice_i

    x_val = desp_regida_global[:, 0]
    y_val = desp_regida_global[:, 1]

    return x_val, y_val