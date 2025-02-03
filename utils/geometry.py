from typing import  Any, Sequence, Tuple
import numpy as np


def find_nearest(array: np.ndarray, value: float) -> Tuple[float, int]:
    """Encuentra el valor más cercano en un arreglo.

    Args:
        array (np.ndarray): Arreglo en el cual buscar.
        value (float): Valor a buscar.

    Returns:
        Tuple[float, int]: Valor más cercano e índice correspondiente.
    """
    index: int = int(np.abs(array - value).argmin())
    return array[index], index

def integrate_array(y: np.ndarray, dx: float) -> np.ndarray:
    """Integra un arreglo usando la regla trapezoidal.

    Args:
        y (np.ndarray): Arreglo a integrar.
        dx (float): Tamaño del paso.

    Returns:
        np.ndarray: Arreglo integrado.
    """
    return np.cumsum(y) * dx

class MatrixException(Exception):
    """Excepción personalizada para errores en analisis matricial."""

    def __init__(self, type_: str, message: str):
        super().__init__(message)
        self.type = type_
        self.message = message

def arg_to_list(arg: Any, n: int) -> list:
    """Convierte un argumento en una lista de longitud n.

    Args:
        arg (Any): Argumento a convertir.
        n (int): Longitud de la lista.

    Returns:
        list: Lista de longitud n.
    """
    if isinstance(arg, Sequence) and not isinstance(arg, str):
        if len(arg) == n:
            return list(arg)
        if len(arg) == 1:
            return [arg[0] for _ in range(n)]
    return [arg for _ in range(n)]

def rotation_matrix(angle: float) -> np.ndarray:
    """Crea una matriz de rotación 2x2.

    Args:
        angle (float): Ángulo en radianes.

    Returns:
        np.ndarray: Matriz de rotación 2x2.
    """
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, -s], [s, c]])

def rotate_xy(a: np.ndarray, angle: float) -> np.ndarray:
    """Rota una matriz 2D alrededor del origen.

    Args:
        a (np.ndarray): Matriz a rotar.
        angle (float): Ángulo en radianes.

    Returns:
        np.ndarray: Matriz rotada.
    """
    b = a - a[0]  # Centrar en el origen
    b = np.dot(b, rotation_matrix(angle))  # Aplicar rotación
    return b + a[0]  # Descentrar

def converge(lhs: float, rhs: float) -> float:
    """Calcula el factor de convergencia.

    Args:
        lhs (float): Lado izquierdo de la ecuación.
        rhs (float): Lado derecho de la ecuación.

    Returns:
        float: Factor de convergencia.
    """
    lhs, rhs = abs(lhs), abs(rhs)
    if rhs == 0 or lhs == 0:
        return 1.0  # Evita división por cero
    div = max(lhs, rhs) / min(lhs, rhs) * 2
    return (rhs / lhs - 1) / div + 1

def angle_x_axis(delta_x: float, delta_y: float) -> float:
    """Calcula el ángulo de un elemento respecto al eje x global usando NumPy.

    Args:
        delta_x (float): Longitud en dirección x.
        delta_y (float): Longitud en dirección y.

    Returns:
        float: Ángulo en radianes.
    """
    angle = np.arccos(delta_x / np.hypot(delta_x, delta_y))
    return 2 * np.pi - angle if delta_y < 0 else angle
