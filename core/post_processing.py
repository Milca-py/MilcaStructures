from typing import List, Tuple, Dict, Union, TYPE_CHECKING
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
                options: "PostProcessingOptions" = PostProcessingOptions(factor=1, n=10)
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
    
    A = (q_j - q_i) / L
    B = q_i
    
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
    N = A*x + B
    
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
    
    A = (q_j - q_i) / length
    B = q_i
    
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
    
    A = (q_j - q_i) / L
    B = q_i
    
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

def values_deformed(element: "Element", factor: int) -> None:
    """obtiene los puntos de la deformada de un elemento."""
    xu1 = element.desplacement[0]
    yu1 = element.desplacement[1]
    
    xo1 = element.node_i.vertex.x 
    yo1 = element.node_i.vertex.y 
    
    n = len(element.deflection)
    L = element.length
    
    def Tetha(element: "Element") -> float:
        xi = element.node_i.vertex.x
        yi = element.node_i.vertex.y
        xf = element.node_j.vertex.x
        yf = element.node_j.vertex.y
        if yf - yi == 0 and xf - xi < 0:
            tetha = -np.pi
        else:
            if xf - xi == 0:
                tetha = np.pi / 2*np.sign(yf - yi)
            elif xf - xi < 0:
                tetha = np.arctan((yf - yi) / (xf - xi)) + np.pi
            else:
                tetha = np.arctan((yf - yi) / (xf - xi))
        return tetha
    
    tetha = Tetha(element)
    x = np.linspace(0, L, n)
    
    x_val = xo1 + x * np.cos(tetha) + xu1 * np.cos(tetha) * factor - element.deflection * np.sin(tetha) * factor
    y_val = yo1 + x * np.sin(tetha) + yu1 * np.sin(tetha) * factor + element.deflection * np.cos(tetha) * factor
    
    return x_val, y_val


# # def values_deformed(element: "Element", factor: int) -> None:
#     """obtiene los puntos de la deformada de un elemento."""
#     ux1 = element.desplacement[0] * factor
#     uy1 = -element.desplacement[1] * factor
#     ux2 = element.desplacement[3] * factor
#     uy2 = -element.desplacement[4] * factor
    
#     x1 = element.node_i.vertex.x + ux1
#     y1 = element.node_i.vertex.y + uy1
#     x2 = element.node_j.vertex.x + ux2
#     y2 = element.node_j.vertex.y + uy2
    
#     # if element.type == ElementType.FRAME:
#     assert element.deflection is not None
#     n = len(element.deflection)
#     x_val = np.linspace(x1, x2, n)
#     y_val = np.linspace(y1, y2, n)
    
#     x_val = x_val + element.deflection * np.sin(element.angle_x) * factor
#     y_val = y_val + element.deflection * -np.cos(element.angle_x) * factor
    
#     # else:
#     #     x_val = np.array([x1, x2])
#     #     y_val = np.array([y1, y2])
#     return x_val, y_val






