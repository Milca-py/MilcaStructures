from typing import TYPE_CHECKING, Dict, Optional
from milcapy.loads.load import PointLoad
import numpy as np

if TYPE_CHECKING:
    from milcapy.utils.types import Restraints
    from milcapy.utils.geometry import Vertex


class Node:
    """Representa un nodo en el modelo estructural."""

    def __init__(self, id: int, vertex: "Vertex") -> None:
        """
        Inicializa un nodo con su identificador y posición.

        Args:
            id (int): Identificador único del nodo.
            vertex (Vertex): Coordenadas (x, y) del nodo.
        """
        self.id: int = id
        self.vertex: "Vertex" = vertex
        self.restraints: "Restraints" = (
            False, False, False)  # Restricciones del nodo

        # Cálculo de los índices de grados de libertad
        self.dofs: np.ndarray = np.array([
            self.id * 3 - 2,  # DOF en x
            self.id * 3 - 1,  # DOF en y
            self.id * 3       # DOF en theta
        ], dtype=int)

        # Cargas aplicadas al nodo
        self.loads: Dict[str, PointLoad] = {}   # {pattern_name: PointLoad}

        # Patrón de carga actual
        self.current_load_pattern: Optional[str] = None

    def load_vector(self) -> np.ndarray:
        """Vector de cargas aplicadas al nodo para el patrón de carga actual en sistema global."""
        load = self.loads.get(self.current_load_pattern, PointLoad())

        return load.components

    def set_current_load_pattern(self, load_pattern_name: str) -> None:
        """Establece el patrón de carga actual del nodo."""
        self.current_load_pattern = load_pattern_name

    def set_restraints(self, restraints: "Restraints") -> None:
        """Establece las restricciones del nodo."""
        self.restraints = restraints

    def set_load(self, load: "PointLoad") -> None:
        """Establece una carga puntual en el nodo."""
        self.loads[self.current_load_pattern] = load
