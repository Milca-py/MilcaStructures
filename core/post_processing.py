from typing import List, Tuple, Dict, Union, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from core.system import SystemMilcaModel
    from core.element import Element


class PostProcessing:
    def __init__(self, system: "SystemMilcaModel"):
        self.system = system
        self.results = self.system.results
        
        
        # siempre se debe llamar a la funcion all_to_element
        self.all_to_element()
    
    @property
    def integration_coefficients_elements(self) -> Dict[int, np.ndarray]:
        "Coeficientes de integracion de los elementos."
        integration_coefficients_elements = {}
        for element in self.system.element_map.values():
            integration_coefficients_elements[element.id] = integration_coefficients(
                element.length,
                element.distributed_load.q_i,
                element.distributed_load.q_j,
                element.section.material.modulus_elasticity,
                element.section.moment_of_inertia,
                element.shear_angle,
                self.results.global_desplacements_elements[element.id][1],
                self.results.global_desplacements_elements[element.id][4],
                self.results.global_desplacements_elements[element.id][2],
                self.results.global_desplacements_elements[element.id][5]
            )
        return integration_coefficients_elements

    def all_to_element(self) -> Dict[int, np.ndarray]:
        "asigna los resultados a cada elemento."
        for element in self.system.element_map.values():
            element.integration_coefficients = self.integration_coefficients_elements[element.id]
















def integration_coefficients(
    L: float,              # Longitud del elemento.
    q_i: float,            # Intensidad de la carga distribuida en el nodo i.
    q_j: float,            # Intensidad de la carga distribuida en el nodo j.
    E: float,              # Modulo de elasticidad.
    I: float,              # Inercia.
    phi: float,            # Aporte por cortante.
    u_i: float,             # Desplazamiento en el nodo i.
    u_j: float,             # Desplazamiento en el nodo j.
    theta_i: float,        # Angulo de rotacion en el nodo i.
    theta_j: float,        # Angulo de rotacion en el nodo
    ) -> np.ndarray:
    "Coeficientes de integracion."
    A = (q_j - q_i) / L
    B = q_i
    
    M = np.array([
        
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [L**3*(2-phi)/12, L**2/2, L, 1],
        [L**2/2, L, 1, 0],
        
    ])
    
    N = np.array([
        
        [E*I*u_i],
        [E*I*theta_i],
        [E*I*u_j + A*L**5*(0.6-phi)/72 + B*L**4*(1-phi)/24],
        [E*I*theta_j + A*L**4/24 + B*L**3/6],
        
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
    B = -axial_i
    
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
    V = - A*x**2/2 - B*x + shear_i
    
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
    M = - A*x**3/6 - B*x**2/2 + shear_i*x - moment_i
    
    return x, M*factor

def values_spin(
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
    
    return x, theta


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
