import numpy as np
from functools import lru_cache

CACHE_COPIES = None

@lru_cache(maxsize=CACHE_COPIES)
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


@lru_cache(maxsize=CACHE_COPIES)
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


@lru_cache(maxsize=CACHE_COPIES)
def local_load_vector(
    L: float,  # Longitud del elemento
    phi: float,  # Aporte de cortante
    q_i: float,  # intensidad de carga transversal en el nodo i
    q_j: float,  # intensidad de carga transversal en el nodo j
    p_i: float,  # intensidad de carga axial en el nodo i
    p_j: float,  # intensidad de carga axial en el nodo j
) -> np.ndarray:
    """
    Calcula el vector de carga local para un elemento de longitud L
    con un coeficiente de Timoshenko phi y cargas distribuidas q_i y q_j.
    
    Args:
        L (float): Longitud del elemento.
        phi (float): Aporte de cortante.
        q_i (float): Carga distribuida en el nodo inicial.
        q_j (float): Carga distribuida en el nodo final.
        p_i (float): Carga axial en el nodo inicial.
        p_j (float): Carga axial en el nodo final.
    
    Returns:
        np.ndarray: Vector de carga local.
    """
    A = (q_j - q_i) / L
    B = q_i

    M = np.array([
        [L**2/2, L],
        [L**3 * (2 - phi) / 12, L**2 / 2]
    ])

    N = np.array([
        A * L**4 / 24 + B * L**3 / 6,
        A * L**5 * (0.6 - phi) / 72 + B * L**4 * (1 - phi) / 24
    ])

    C = np.linalg.solve(M, N)  # Más eficiente que inv(M) @ N

    Q = np.array([
        (2 * p_i + p_j) * L / 6,
        C[0],
        -C[1],
        (p_i + 2 * p_j) * L / 6,
        -(-A * L**2 / 2 - B * L + C[0]),
        (-A * L**3 / 6 - B * L**2 / 2 + C[0] * L + C[1]),
    ])
    
    return Q


# otra alternativa
def load_vector(L, q_i, q_j):
    import numpy as np
    f = np.array([
        0,
        L/20 * (7*q_i + 3*q_j),
        L**2 * (q_i/20 + q_j/30),
        0,
        L/20 * (3*q_i + 7*q_j),
        -L**2 * (q_i/30 + q_j/20),
    ])
    return f