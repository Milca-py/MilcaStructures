import numpy as np
from functools import lru_cache

CACHE_COPIES = None

# ==========================================================================================================
# ===================================   MATRICES DE RIGIDEZ   ==============================================
# ==========================================================================================================

@lru_cache(maxsize=CACHE_COPIES)
def local_stiffness_matrix(
    E: float,  # Módulo de Young
    I: float,  # Momento de inercia
    A: float,  # Área de la sección
    L: float,  # Longitud del elemento
    le: float,  # Longitud de la parte flexible
    phi: float,  # Aporte de cortante
) -> np.ndarray:
    """Matriz de rigidez local de una viga.

    Args:
        E (float): Módulo de elasticidad.
        I (float): Momento de inercia.
        A (float): Área transversal.
        L (float): Longitud del elemento.
        phi (float): Aporte de cortante.

    Returns:
        np.ndarray: Matriz de rigidez 6x6.
    """
    k11 = E * A / L
    k22 = 12 * E * I / (le**3 * (1 + phi))
    k23 = 6 * E * I / (le**2 * (1 + phi))
    k33 = (4 + phi) * E * I / (le * (1 + phi))
    k66 = (2 - phi) * E * I / (le * (1 + phi))

    return np.array([
        [ k11,   0,    0,   -k11,   0,    0   ],
        [  0,   k22,  k23,    0,   -k22,  k23 ],
        [  0,   k23,  k33,    0,   -k23,  k66 ],
        [-k11,   0,    0,    k11,   0,    0   ],
        [  0,  -k22, -k23,    0,    k22, -k23 ],
        [  0,   k23,  k66,    0,   -k23,  k33 ]
    ], dtype=np.float64)

# ==========================================================================================================
# ===================================   MATRICES DE TRANFORMACION   ========================================
# ==========================================================================================================

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
    ], dtype=np.float64)

@lru_cache(maxsize=CACHE_COPIES)
def length_offset_transformation_matrix(la: float, lb: float) -> np.ndarray:
    """
    Calcula la matriz de tranformacion para miembros con brazos rigidos.

    La matriz de tranformacion se utiliza para transformar las cargas
    distribuidas en los brazos a los extremos de la barra.

    Parameters
    ----------
    la : float
        Longitud del brazo izquierdo.
    lb : float
        Longitud del brazo derecho.

    Returns
    -------
    np.ndarray
        Matriz de tranformacion 6x6.
    """
    return np.array([
        [1, 0, 0,  0, 0,   0],
        [0, 1, la, 0, 0,   0],
        [0, 0, 1,  0, 0,   0],
        [0, 0, 0,  1, 0,   0],
        [0, 0, 0,  0, 1, -lb],
        [0, 0, 0,  0, 0,   1]
    ], dtype=np.float64)


# ==========================================================================================================
# ===================================   VECTORES DE CARGAS   ===============================================
# ==========================================================================================================

@lru_cache(maxsize=CACHE_COPIES)
def q_phi(L, phi, qi, qj, pi, pj):
    """
    Calcula el vector de carga para un miembro de longitud L con deformaciones de corte
    y cargas distribuidas qi y qj.

    Args:
        qi (float): Intensidad de carga distribuida transversal en el nodo inicial.
        qj (float): Intensidad de carga distribuida transversal en el nodo final.
        pi (float): Intensidad de carga distribuida axial en el nodo inicial.
        pj (float): Intensidad de carga distribuida axial en el nodo final.
        L (float): Longitud del miembro.
        phi (float): Aporte de cortante.

    Returns:
        np.ndarray: Vector de carga.
    """
    a = (qj - qi) / L
    b = qi

    N1 = (2 * pi + pj) * L / 6
    V1 = b * L / 2 + a * L**2 / 60 * ((10 * phi + 9) / (phi + 1))
    M1 = b * L**2 / 12 + a * L**3 / 120 * ((5 * phi + 4) / (phi + 1))
    N2 = (pi + 2 * pj) * L / 6
    V2 = b * L / 2 + a * L**2 / 60 * ((20 * phi + 21) / (phi + 1))
    M2 = -(b * L**2 / 12 + a * L**3 / 120 * ((5 * phi + 6) / (phi + 1)))
    return np.array([N1, V1, M1, N2, V2, M2])

def length_offset_q(L, phi, qi, qj, pi, pj, la, lb, qla, qlb):
    """
    Calcula el vector de cargas considerando brazos rigidos.

    Primero se calculan las cargas distribuidas en los extremos de los brazos
    rigidos y luego se suman las cargas distribuidas en la parte flexible y
    en los extremos de los brazos rigidos.

    Args:
        L (float): Longitud del miembro.
        phi (float): Aporte de cortante.
        qi (float): Intensidad de carga distribuida transversal en el nodo inicial.
        qj (float): Intensidad de carga distribuida transversal en el nodo final.
        la (float): Longitud del brazo izquierdo.
        lb (float): Longitud del brazo derecho.

    Returns:
        np.ndarray: Vector de carga.
    """
    a = (qj - qi) / L
    b = qi

    c = (pj - pi) / L
    d = pi

    qa = a*la + b
    qb = a*(L -lb) + b

    pa = c*la + d
    pb = c*(L -lb) + d

    q = q_phi(L - la - lb, phi, qa, qb, pa, pb)   # Con deformaciones por corte (parte flexible)
    if qla:
        qla = q_phi(la, 0, qi, qa, pa, pb)            # Sin deformaciones por corte (parte rígida inicial)
    else:
        qla = np.zeros(6)
    if qlb:
        qlb = q_phi(lb, 0, qb, qj, pa, pb)            # Sin deformaciones por corte (parte rígida final)
    else:
        qlb = np.zeros(6)

    q_contacto = np.concatenate([qla[3:], qlb[:3]])   # Empotramiento en los contactos
    q_extremos = np.concatenate([qla[:3], qlb[3:]])   # Empotramiento en los extremos

    H = length_offset_transformation_matrix(la, lb)
    Q = np.dot(H.T, q + q_contacto) + q_extremos

    return Q
