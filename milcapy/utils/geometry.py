import numpy as np


class Vertex:
    def __init__(
        self,
        x: float,
        y: float,
    ):
        self.coordinates: np.ndarray = np.array([x, y], dtype=np.float32)
        self.x: float = float(self.coordinates[0])
        self.y: float = float(self.coordinates[1])

    def distance_to(self, other: "Vertex") -> float:
        """Calcular la distancia entre dos vértices

        Args:
            other (Vertex): Otro vértice

        Returns:
            float: Distancia entre los vértices
        """
        return np.linalg.norm(self.coordinates - other.coordinates)

    def __str__(self) -> str:
        return f"Vertex({self.x}, {self.y})"


class MatrixException(Exception):
    """Excepción personalizada para errores en análisis matricial."""

    def __init__(self, type_: str, message: str):
        super().__init__(message)
        self.type = type_
        self.message = message


def rotation_matrix(angle: float) -> np.ndarray:
    """Crea una matriz de rotación 2x2.

    Args:
        angle (float): Ángulo en radianes.

    Returns:
        np.ndarray: Matriz de rotación 2x2.
    """
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, -s], [s, c]])


def rotate_xy(a: np.ndarray, angle: float, x: float, y: float) -> np.ndarray:
    """Rota una matriz 2D alrededor del origen, en sentido antihorario.

    Args:
        a (np.ndarray): Matriz a rotar.
        angle (float): Ángulo en radianes.

    Returns:
        np.ndarray: Matriz rotada.
    """
    b = a - np.array([x, y])  # Centrar en el origen
    b = np.dot(b, rotation_matrix(angle).T)  # Aplicar rotación
    return b + np.array([x, y])  # Descentrar


def traslate_xy(a: np.ndarray, x: float, y: float) -> np.ndarray:
    """Traslada una matriz 2D.

    Args:
        a (np.ndarray): Matriz a trasladar.
        x (float): Traslación en x.
        y (float): Traslación en y.

    Returns:
        np.ndarray: Matriz trasladada.
    """
    return a + np.array([x, y])


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