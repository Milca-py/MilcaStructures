import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from milcapy.elements.member import Member


def integration_coefficients(member: "Member") -> np.ndarray:
    """
    Calcula los coeficientes de integración para un elemento.

    Args:
        member: Elemento estructural

    Returns:
        Array con los coeficientes de integración (C1, C2, C3, C4)
    """
    L = member.length()                             # Longitud del elemento
    q_i = member.distributed_load.q_i               # Intensidad de la carga distribuida en el nodo i
    q_j = member.distributed_load.q_j               # Intensidad de la carga distribuida en el nodo j
    E = member.section.E()                          # Módulo de elasticidad
    I = member.section.I()                          # Inercia
    phi = member.phi()                              # Aporte por cortante
    u_i = member.displacement[1]                    # Desplazamiento en el GDL 2 nodo i
    u_j = member.displacement[4]                    # Desplazamiento en el GDL 5 nodo j
    theta_i = member.displacement[2]                # Ángulo de rotación en el GDL 3 nodo i
    theta_j = member.displacement[5]                # Ángulo de rotación en el GDL 6 nodo j

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


def axial_force(member: "Member", factor: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la fuerza axial a lo largo del elemento.

    Args:
        member: Elemento estructural
        factor: Factor de escala
        n: Número de puntos para discretizar

    Returns:
        Tupla de arrays con las coordenadas x y los valores de fuerza axial
    """
    axial_i = member.internal_forces[0]
    axial_j = member.internal_forces[3]

    length = member.length()

    A = (axial_j + axial_i) / length
    B = axial_i

    x = np.linspace(0, length, n)
    N = - A*x + B

    return x, N * factor


def shear_force(member: "Member", factor: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la fuerza cortante a lo largo del elemento.

    Args:
        member: Elemento estructural
        factor: Factor de escala
        n: Número de puntos para discretizar

    Returns:
        Tupla de arrays con las coordenadas x y los valores de fuerza cortante
    """
    q_i = member.distributed_load.q_i
    q_j = member.distributed_load.q_j

    shear_i = member.internal_forces[1]

    length = member.length()

    A = (q_j - q_i) / length
    B = q_i

    x = np.linspace(0, length, n)
    V = A * x**2 / 2 + B * x + shear_i

    return x, V * factor


def bending_moment(member: "Member", factor: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula el momento flector a lo largo del elemento.

    Args:
        member: Elemento estructural
        factor: Factor de escala
        n: Número de puntos para discretizar

    Returns:
        Tupla de arrays con las coordenadas x y los valores de momento flector
    """
    q_i = member.distributed_load.q_i
    q_j = member.distributed_load.q_j

    shear_i = member.internal_forces[1]
    moment_i = member.internal_forces[2]

    length = member.length()

    A = (q_j - q_i) / length
    B = q_i

    x = np.linspace(0, length, n)
    M = -(A * x**3 / 6 + B * x**2 / 2 + shear_i * x - moment_i)

    return x, M * factor


def slope(member: "Member", factor: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la pendiente de la deformada a lo largo del elemento.

    Args:
        member: Elemento estructural
        factor: Factor de escala
        n: Número de puntos para discretizar

    Returns:
        Tupla de arrays con las coordenadas x y los valores de pendiente
    """
    q_i = member.distributed_load.q_i
    q_j = member.distributed_load.q_j

    length = member.length()

    A = (q_j - q_i) / length
    B = q_i

    # Coeficientes de integración
    COEF = integration_coefficients(member)
    C1 = COEF[0]
    C2 = COEF[1]
    C3 = COEF[2]

    E = member.section.E()
    I = member.section.I()

    x = np.linspace(0, length, n)
    theta = (A * x**4 / 24 + B * x**3 / 6 + C1 * x**2 / 2 + C2 * x + C3) / (E * I)

    return theta * factor


def deflection(member: "Member", factor: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la deflexión a lo largo del elemento.

    Args:
        member: Elemento estructural
        factor: Factor de escala
        n: Número de puntos para discretizar

    Returns:
        Tupla de arrays con las coordenadas x y los valores de deflexión
    """
    q_i = member.distributed_load.q_i
    q_j = member.distributed_load.q_j

    L = member.length()

    A = (q_j - q_i) / L
    B = q_i

    C1 = member.integration_coefficients[0]
    C2 = member.integration_coefficients[1]
    C3 = member.integration_coefficients[2]
    C4 = member.integration_coefficients[3]

    shear_angle = member.shear_angle  # aporte por cortante (phi)
    E = member.section.E()
    I = member.section.I()

    x = np.linspace(0, L, n)
    term1 = A * L**2 * x**3 * (0.6 * (x / L)**2 - shear_angle) / 72
    term2 = B * L**2 * x**2 * ((x / L)**2 - shear_angle) / 24
    term3 = C1 * x * L**2 * (2 * (x / L)**2 - shear_angle) / 12
    term4 = C2 * x**2 / 2 + C3 * x + C4

    u = (term1 + term2 + term3 + term4) / (E * I)
    return x, u * factor



def deformed_shape(member: "Member", results: dict, escale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula las coordenada de la deformada en sistema local"""
    L = member.length()
    le = member.le()
    la = member.la or 0
    lb = member.lb or 0

    vtx_local = np.array([
        [la, 0],
        [le + la, 0]
    ])

    disp_local = np.array([
        [results["displacements"][0], results["displacements"][1]],
        [results["displacements"][3], results["displacements"][4]]
    ]) * escale + vtx_local

    flecha = results["deflections"] * escale
    x = np.linspace(0, le, len(flecha))

    # Corrección de la deformada por deformación axial
    Lf = disp_local[1, 0] - disp_local[0, 0]
    x_val = x * Lf / le + disp_local[0, 0]

    #? AJUSTE Y OBTENSION DE LOS PUNTOS DE LOS EXTREMOS
    H = member.H()
    Hinv = np.linalg.pinv(H)
    u = results["displacements"]
    urig = Hinv @ u * escale
    extremos =  np.array([
        [urig[0], urig[1]],
        [urig[3] + L, urig[4]]
    ])

    flecha = np.concatenate(([extremos[0][1]], flecha, [extremos[1][1]]))
    x_val = np.concatenate(([extremos[0][0]], x_val, [extremos[1][0]]))
    return x_val, flecha