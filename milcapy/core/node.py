from typing import TYPE_CHECKING, Dict, Optional
import numpy as np

if TYPE_CHECKING:
    from milcapy.utils.types import Restraints
    from milcapy.utils.geometry import Vertex
    from milcapy.loads.load import PointLoad


class Node:
    """
    Representa un nodo en el modelo estructural.

    Atributos:
        id (int): Identificador único del nodo.
        vertex (Vertex): Coordenadas (x, y) del nodo.
        restraints (Restraints): Restricciones del nodo [ux, uy, theta].
        dof (np.ndarray): Grados de libertad del nodo.
        loads (Dict[str, PointLoad]): Cargas aplicadas al nodo.
        displacement (np.ndarray): Desplazamientos calculados del nodo.
        reaction (np.ndarray): Reacciones calculadas en el nodo.
    """

    def __init__(self, id: int, vertex: "Vertex") -> None:
        """
        Inicializa un nodo con su identificador y posición.

        Args:
            id (int): Identificador único del nodo.
            vertex (Vertex): Coordenadas (x, y) del nodo.
        """
        self.id: int = id
        self.vertex: "Vertex" = vertex
        self.restraints: "Restraints" = (False, False, False)  # Restricciones del nodo

        # Cálculo de los índices de grados de libertad
        self.dof: np.ndarray = np.array([
            self.id * 3 - 2,  # DOF en x
            self.id * 3 - 1,  # DOF en y
            self.id * 3       # DOF en theta (corregido: era -0)
        ], dtype=int)

        # Cargas aplicadas al nodo
        self.loads: Dict[str, PointLoad] = {}   # {pattern_name: PointLoad}

        # Patrón de carga actual
        self.current_load_pattern: Optional[str] = None

    def set_restraints(self, restraints: "Restraints") -> None:
        """
        Asigna restricciones al nodo.

        Args:
            restraints (Restraints): Restricciones [ux, uy, theta].
        """
        self.restraints = restraints 

    def set_load(self, load: PointLoad) -> None:
        """
        Suma una carga puntual a las fuerzas existentes en el nodo.

        Args:
            load (PointLoad): Carga puntual aplicada al nodo.
        """
        self.loads[self.current_load_pattern] = load

    def __str__(self) -> str:
        """
        Representación en cadena del nodo.
        
        Returns:
            str: Descripción del nodo.
        """
        return f"Node {self.id}: {self.vertex}, Restraints: {self.restraints}, Loads: {self.loads}"