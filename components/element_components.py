import numpy as np

def local_stiffness_matrix(
    E: float,  # Módulo de Young
    I: float,  # Momento de inercia
    A: float,  # Área de la sección
    L: float,  # Longitud del elemento
    v: float,  # Coeficiente de Poisson
    k: float,  # Coeficiente de Timoshenko
    shear_effect: bool = True  # Efectos por cortante
) -> np.ndarray:
    """Matriz de rigidez local de una viga.

    Args:
        E (float): Módulo de elasticidad.
        I (float): Momento de inercia.
        A (float): Área transversal.
        L (float): Longitud del elemento.
        v (float): Coef. de Poisson.
        κ (float): Factor de Timoshenko.
        shear_effect (bool): Si es True, incluye cortante.

    Returns:
        np.ndarray: Matriz de rigidez 6x6.
    """
    G = E / (2 * (1 + v))  # Módulo de corte
    φ = (12 * E * I) / (L**2 * A * k * G) if shear_effect else 0  # Corrección por cortante

    k11 = E * A / L
    k22 = 12 * E * I / (L**3 * (1 + φ))
    k23 = 6 * E * I / (L**2 * (1 + φ))
    k33 = (4 + φ) * E * I / (L * (1 + φ))
    k66 = (2 - φ) * E * I / (L * (1 + φ))

    return np.array([
        [ k11,   0,    0,   -k11,   0,    0   ],
        [  0,   k22,  k23,    0,   -k22,  k23 ],
        [  0,   k23,  k33,    0,   -k23,  k66 ],
        [-k11,   0,    0,    k11,   0,    0   ],
        [  0,  -k22, -k23,    0,    k22, -k23 ],
        [  0,   k23,  k66,    0,   -k23,  k33 ]
    ])



def transformation_matrix(
    angle: float
) -> np.ndarray:
    """Matriz de transformación para un ángulo dado.

    Args:
        angle (float): Ángulo de rotación en radianes.

    Returns:
        np.ndarray: Matriz de transformación.
    """
    s, c = np.sin(angle), np.cos(angle)
    return np.array([
        [ c,  s, 0,  0, 0, 0 ],
        [-s,  c, 0,  0, 0, 0 ],
        [ 0,  0, 1,  0, 0, 0 ],
        [ 0,  0, 0,  c, s, 0 ],
        [ 0,  0, 0, -s, c, 0 ],
        [ 0,  0, 0,  0, 0, 1 ]
    ])


def trapezoidal_load_vector(q_i: float, q_j: float, L: float) -> np.ndarray:
    """Vector de fuerzas nodales equivalentes para una carga trapezoidal.

    Args:
        q_i (float): Intensidad de carga en el nodo inicial.
        q_j (float): Intensidad de carga en el nodo final.
        L (float): Longitud del elemento.

    Returns:
        np.ndarray: Vector de fuerzas nodales [Fxi, Fyi, Mi, Fxj, Fyj, Mj].
    """

    F_i = (7 * q_i + 3 * q_j) * L / 20
    M_i = (q_i / 20 + q_j / 30) * L**2 
    F_j = (3 * q_i + 7 * q_j) * L / 20
    M_j = -((q_i / 30 + q_j / 20) * L**2 )

    return np.array([0, F_i, M_i, 0, F_j, M_j])


def axial_linear_force(p_i: float, p_j: float, L: float) -> np.ndarray:
    """Vector de fuerzas nodales equivalentes para una carga axial lineal.

    Args:
        p_i (float): Carga axial en el nodo inicial.
        p_j (float): Carga axial en el nodo final.
        L (float): Longitud del elemento.

    Returns:
        np.ndarray: Vector de fuerzas nodales [Fxi, Fyi, Mi, Fxj, Fyj, Mj].
    """
    F_i = -(2 * p_i + p_j) * L / 6
    F_j = -(p_i + 2 * p_j) * L / 6

    return np.array([F_i, 0, 0, F_j, 0, 0])


def moment_linear_force(m_i: float, m_j: float, L: float) -> np.ndarray:
    """Vector de fuerzas nodales equivalentes para una carga de momento lineal.

    Args:
        m_i (float): Momento en el nodo inicial.
        m_j (float): Momento en el nodo final.
        L (float): Longitud del elemento.

    Returns:
        np.ndarray: Vector de fuerzas nodales [Fxi, Fyi, Mi, Fxj, Fyj, Mj].
    """
    F_i = (m_i + m_j) / 2
    M_i = (m_i - m_j) * L / 12

    return np.array([0, F_i, M_i, 0, F_i, -M_i])