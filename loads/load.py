from typing import Optional, Dict, Union
import numpy as np


class PointLoad:
    """
    Representa una carga puntual en 2D con fuerzas en X, Y y momento en Z.
    """

    def __init__(
        self,
        fx: float = 0.0,
        fy: float = 0.0,
        mz: float = 0.0,
        id: Optional[int] = None,
    ) -> None:
        """Inicializa un objeto de carga puntual.

        Args:
            fx (float, opcional): Fuerza en dirección X. Default es 0.0.
            fy (float, opcional): Fuerza en dirección Y. Default es 0.0.
            mz (float, opcional): Momento alrededor del eje Z. Default es 0.0.
            id (Optional[int], opcional): Identificador de la carga. Default es None.
        """
        self.fx = float(fx)
        self.fy = float(fy)
        self.mz = float(mz)
        self.id = id

    @property
    def components(self) -> np.ndarray:
        """Devuelve los componentes de la carga como un array de NumPy."""
        return np.array([self.fx, self.fy, self.mz], dtype=np.float64)

    def __repr__(self) -> str:
        return f"PointLoad(fx={self.fx}, fy={self.fy}, mz={self.mz}, id={self.id})"

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: "PointLoad") -> "PointLoad":
        if not isinstance(other, PointLoad):
            return NotImplemented
        return PointLoad(
            fx=self.fx + other.fx,
            fy=self.fy + other.fy,
            mz=self.mz + other.mz,
            id=self.id or other.id,
        )

    def __sub__(self, other: "PointLoad") -> "PointLoad":
        if not isinstance(other, PointLoad):
            return NotImplemented
        return PointLoad(
            fx=self.fx - other.fx,
            fy=self.fy - other.fy,
            mz=self.mz - other.mz,
            id=self.id or other.id,
        )

    def __mul__(self, other: Union[float, int]) -> "PointLoad":
        """Multiplicación por un escalar."""
        if not isinstance(other, (float, int)):
            return NotImplemented
        return PointLoad(
            fx=self.fx * other,
            fy=self.fy * other,
            mz=self.mz * other,
            id=self.id,
        )

    def __rmul__(self, other: Union[float, int]) -> "PointLoad":
        """Multiplicación con escalar (permite 2 * carga)."""
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, int]) -> "PointLoad":
        """División por un escalar."""
        if not isinstance(other, (float, int)):
            return NotImplemented
        if other == 0:
            raise ZeroDivisionError("No se puede dividir por cero.")
        return PointLoad(
            fx=self.fx / other,
            fy=self.fy / other,
            mz=self.mz / other,
            id=self.id,
        )

    def __neg__(self) -> "PointLoad":
        """Carga invertida (negativa)."""
        return PointLoad(-self.fx, -self.fy, -self.mz, id=self.id)

    def __eq__(self, other: object) -> bool:
        """Compara si dos cargas son iguales."""
        if not isinstance(other, PointLoad):
            return NotImplemented
        return np.allclose(self.components, other.components)
    
    def __pos__(self) -> "PointLoad":
        """Carga positiva."""
        return self

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convierte la carga en un diccionario."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "mz": self.mz,
            "id": self.id,
        }


class DistributedLoad:
    """Clase que representa una carga distribuida en un elemento estructural."""

    __slots__ = ("q_i", "q_j", "id")

    def __init__(self, q_i: float = 0.0, q_j: float = 0.0, id: Optional[int] = None) -> None:
        """Inicializa un objeto de carga distribuida.

        Args:
            q_i (float, opcional): Carga en el nodo inicial. Default es 0.0.
            q_j (float, opcional): Carga en el nodo final. Default es 0.0.
            id (Optional[int], opcional): Identificador de la carga. Default es None.
        """
        if not isinstance(q_i, (int, float)) or not isinstance(q_j, (int, float)):
            raise TypeError("Las cargas q_i y q_j deben ser números reales.")

        self.q_i: float = float(q_i)
        self.q_j: float = float(q_j)
        self.id: Optional[int] = id

    @property
    def components(self) -> np.ndarray:
        """Devuelve los componentes de la carga como un array de NumPy."""
        return np.array([self.q_i, self.q_j], dtype=np.float64)

    def __repr__(self) -> str:
        return f"DistributedLoad(q_i={self.q_i}, q_j={self.q_j}, id={self.id})"

    def __str__(self) -> str:
        return self.__repr__()

    def __mul__(self, other: Union[float, int]) -> "DistributedLoad":
        """Multiplicación por un escalar."""
        if isinstance(other, (int, float)):
            return DistributedLoad(self.q_i * other, self.q_j * other, self.id)
        return NotImplemented

    __rmul__ = __mul__  # Evita duplicar código para multiplicación conmutativa.

    def __truediv__(self, other: Union[float, int]) -> "DistributedLoad":
        """División por un escalar."""
        if not isinstance(other, (int, float)):
            return NotImplemented
        if other == 0:
            raise ZeroDivisionError("No se puede dividir por cero.")
        return DistributedLoad(self.q_i / other, self.q_j / other, self.id)

    def __neg__(self) -> "DistributedLoad":
        """Carga invertida (negativa)."""
        return DistributedLoad(-self.q_i, -self.q_j, self.id)

    def __eq__(self, other: object) -> bool:
        """Compara si dos cargas son iguales."""
        if not isinstance(other, DistributedLoad):
            return NotImplemented
        return np.allclose(self.components, other.components)

    def to_dict(self) -> Dict[str, Union[int, float, None]]:
        """Convierte la carga en un diccionario."""
        return {"q_i": self.q_i, "q_j": self.q_j, "id": self.id}
