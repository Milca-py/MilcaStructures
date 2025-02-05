import numpy as np

def local_stiffness_matrix(
    modulus_elasticity: float,
    moment_of_inertia: float,
    area: float,
    length: float,
    poisson_ratio: float,
    timoshenko_coefficient: float,
    def_shear: bool = False
) -> np.ndarray:
    """Matriz de rigidez elemental para un elemento de viga.

    Args:
        modulus_elasticity (float): Módulo de Young.
        moment_of_inertia (float): Momento de inercia.
        area (float): Área de la sección transversal.
        length (float): Longitud del elemento.
        poisson_ratio (float): Coeficiente de Poisson.
        timoshenko_coefficient (float): Factor de corrección por cortante.
        def_shear (bool, opcional): Considerar deformaciones por cortante.

    Returns:
        np.ndarray: Matriz de rigidez elemental.
    """
    shear_modulus = modulus_elasticity / (2 * (1 + poisson_ratio))  # Cálculo de G
    shear_deflection = 12 * modulus_elasticity * moment_of_inertia / (length**2 * area * timoshenko_coefficient * shear_modulus) if def_shear else 0

    k11 = modulus_elasticity * area / length
    k22 = 12 * modulus_elasticity * moment_of_inertia / (length**3 * (1 + shear_deflection))
    k23 = 6 * modulus_elasticity * moment_of_inertia / (length**2 * (1 + shear_deflection))
    k33 = (4 + shear_deflection) * modulus_elasticity * moment_of_inertia / (length * (1 + shear_deflection))
    k66 = (2 - shear_deflection) * modulus_elasticity * moment_of_inertia / (length * (1 + shear_deflection))

    return np.array([
        [ k11,   0,    0,   -k11,   0,    0 ],
        [  0,   k22,  k23,    0,   -k22,  k23 ],
        [  0,   k23,  k33,    0,   -k23,  k66 ],
        [-k11,   0,    0,    k11,   0,    0 ],
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


def trapezoidal_load_vector(
    load_start: float,
    load_end: float,
    length: float
) -> np.ndarray:
    """Vector de cargas equivalentes para carga trapezoidal en un elemento.

    Args:
        load_start (float): Carga en el nodo inicial.
        load_end (float): Carga en el nodo final.
        length (float): Longitud del elemento.

    Returns:
        np.ndarray: Vector de fuerzas nodales equivalentes.
    """
    Fi = 7 / 20 * load_start * length + 3 / 20 * load_end * length
    Mi = 1 / 20 * load_start * length**2 + 1 / 30 * load_end * length**2
    Fj = 3 / 20 * load_start * length + 7 / 20 * load_end * length
    Mj = -(1 / 30 * load_start * length**2 + 1 / 20 * load_end * length**2)
    
    return np.array([0, Fi, Mi, 0, Fj, Mj])
