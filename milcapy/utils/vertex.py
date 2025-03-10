from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Sequence, Union

if TYPE_CHECKING:
    from utils.custom_types import NumberLike, VertexLike


class Vertex:
    """
    Punto de utilidad en 2D.
    """

    def __init__(
        self,
        x: Union["VertexLike", "NumberLike"],
        y: Union["NumberLike", None] = None,
    ):
        """Crear un objeto Vertex

        Args:
            x (Union[VertexLike, NumberLike]): Coordenada X o un objeto Vertex, o un objeto que
                puede ser convertido a un Vertex.
            y (Union[NumberLike, None], optional): Coordenada Y. Por defecto es None.
        """
        if isinstance(x, Vertex):
            self.coordinates: np.ndarray = np.array(
                x.coordinates, dtype=np.float32)
        elif (
            isinstance(x, (Sequence, np.ndarray))
            and len(x) == 2
            and isinstance(x[0], (float, int, np.number))
            and isinstance(x[1], (float, int, np.number))
        ):
            self.coordinates = np.array([x[0], x[1]], dtype=np.float32)
        elif isinstance(x, (float, int, np.number)) and isinstance(
            y, (float, int, np.number)
        ):
            self.coordinates = np.array([x, y], dtype=np.float32)
        else:
            raise TypeError(
                "Los puntos deben ser convertibles a un objeto Vertex: (x, y) o [x, y] o np.array([x, y]) o Vertex(x, y)"
            )

    @property
    def x(self) -> float:
        """Coordenada X

        Returns:
            float: Coordenada X
        """
        return float(self.coordinates[0])

    @property
    def y(self) -> float:
        """Coordenada Y

        Returns:
            float: Coordenada Y
        """
        return float(self.coordinates[1])

    def modulus(self) -> float:
        """Magnitud del vector desde el origen hasta el Vertex

        Returns:
            float: Magnitud del vector desde el origen hasta el Vertex
        """
        return float(np.sqrt(np.sum(self.coordinates**2)))
    
    def unit(self) -> Vertex:
        """Vector unitario desde el origen hasta el Vertex

        Returns:
            Vertex: Vector unitario desde el origen hasta el Vertex
        """
        return (1 / self.modulus) * self

    def __add__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Sumar dos objetos Vertex

        Args:
            other (Union[VertexLike, NumberLike]): Vertex a sumar

        Returns:
            Vertex: Suma de los dos objetos Vertex
        """
        other = det_coordinates(other)
        return Vertex(self.coordinates + other)

    def __radd__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Sumar dos objetos Vertex

        Args:
            other (Union[VertexLike, NumberLike]): Vertex a sumar

        Returns:
            Vertex: Suma de los dos objetos Vertex
        """
        return self.__add__(other)

    def __sub__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Restar dos objetos Vertex

        Args:
            other (Union[VertexLike, NumberLike]): Vertex a restar

        Returns:
            Vertex: Diferencia de los dos objetos Vertex
        """
        other = det_coordinates(other)
        return Vertex(self.coordinates - other)

    def __rsub__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Restar dos objetos Vertex

        Args:
            other (Union[VertexLike, NumberLike]): Vertex a restar

        Returns:
            Vertex: Diferencia de los dos objetos Vertex
        """
        return self.__sub__(other)

    def __mul__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Multiplicar dos objetos Vertex

        Args:
            other (Union[VertexLike, NumberLike]): Vertex a multiplicar

        Returns:
            Vertex: Producto de los dos objetos Vertex
        """
        other = det_coordinates(other)
        return Vertex(self.coordinates * other)

    def __rmul__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Multiplicar dos objetos Vertex

        Args:
            other (Union[VertexLike, NumberLike]): Vertex a multiplicar

        Returns:
            Vertex: Producto de los dos objetos Vertex
        """
        return self.__mul__(other)

    def __truediv__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Dividir dos objetos Vertex

        Args:
            other (Union[VertexLike, NumberLike]): Vertex a dividir

        Returns:
            Vertex: Cociente de los dos objetos Vertex
        """
        other = det_coordinates(other)
        return Vertex(self.coordinates / other)

    def __neg__(self) -> Vertex:
        """Negación de un objeto Vertex

        Returns:
            Vertex: Negación de un objeto Vertex
        """
        return Vertex(-self.coordinates)

    def __pos__(self) -> Vertex:
        """Suma unaria de un objeto Vertex

        Returns:
            Vertex: Suma unaria de un objeto Vertex
        """
        return Vertex(+self.coordinates)

    def __pow__(self, power: float) -> Vertex:
        """Elevar cada componente del Vertex a una potencia

        Args:
            power (float): Potencia a elevar

        Returns:
            Vertex: Vertex con cada componente elevado a la potencia
        """
        if not isinstance(power, (int, float)):
            raise TypeError("La potencia debe ser un escalar (int o float)")
        return Vertex(self.coordinates**power)

    def __abs__(self) -> float:
        """Valor absoluto de un objeto Vertex

        Returns:
            float: Valor absoluto de un objeto Vertex
        """
        return Vertex(np.abs(self.coordinates))

    def __matmul__(self, other: Vertex) -> float:
        """Producto punto usando el operador @

        Args:
            other (Vertex): Vertex a multiplicar

        Returns:
            float: Producto punto entre los dos objetos Vertex
        """
        return np.dot(self.coordinates, other.coordinates)

    def __eq__(self, other: object) -> bool:
        """Verificar si dos objetos Vertex son iguales

        Args:
            other (object): Objeto a comparar

        Raises:
            NotImplementedError: Si el objeto no es un objeto Vertex o una tupla o lista de longitud 2

        Returns:
            bool: True si los dos objetos Vertex son iguales
        """
        if isinstance(other, Vertex):
            return self.x == other.x and self.y == other.y
        if (
            isinstance(other, (np.ndarray, Sequence))
            and len(other) == 2
            and isinstance(other[0], (int, float))
            and isinstance(other[1], (int, float))
        ):
            return self.x == other[0] and self.y == other[1]
        return NotImplemented

    def __str__(self) -> str:
        """Representación en cadena del objeto Vertex

        Returns:
            str: Representación en cadena del objeto Vertex
        """
        return f"Vertex({self.x}, {self.y})"

    def __repr__(self) -> str:
        """Representación en cadena del objeto Vertex

        Returns:
            str: Representación en cadena del objeto Vertex
        """
        return f"Vertex({self.x}, {self.y})"


def vertex_range(v1: Vertex, v2: Vertex, n: int) -> list:
    """Crear una lista de n + 1 objetos Vertex entre dos objetos Vertex

    Args:
        v1 (Vertex): Vertex inicial
        v2 (Vertex): Vertex final
        n (int): Número de objetos Vertex a crear

    Returns:
        list: Lista de n + 1 objetos Vertex entre v1 y v2
    """
    dv = v2 - v1
    return [v1 + dv * i / n for i in range(n + 1)]


def det_coordinates(point: Union["VertexLike", "NumberLike"]) -> np.ndarray:
    """Convertir un punto a coordenadas

    Args:
        point (Union[VertexLike, NumberLike]): Punto a convertir

    Raises:
        TypeError: Si el punto no es convertible a un objeto Vertex

    Returns:
        np.ndarray: Coordenadas del punto
    """
    if isinstance(point, Vertex):
        return point.coordinates
    if (
        isinstance(point, (np.ndarray, Sequence))
        and len(point) == 2
        and isinstance(point[0], (float, int, np.number))
        and isinstance(point[1], (float, int, np.number))
    ):
        return np.asarray(point)
    if isinstance(point, (float, int, np.number)):
        return np.array([point, point])
    raise TypeError(
        "Los puntos deben ser convertibles a un objeto Vertex: (x, y) o [x, y] o np.array([x, y]) o Vertex(x, y)"
    )
