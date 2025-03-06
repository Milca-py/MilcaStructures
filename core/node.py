from typing import TYPE_CHECKING, Optional
import numpy as np
from loads.load import PointLoad

if TYPE_CHECKING:
    from utils.custom_types import Restraints
    from utils.vertex import Vertex

class Node:
    """
    Representa un nodo en el modelo estructural.

    Atributos:
        id (int): Identificador único del nodo.
        vertex (Vertex): Coordenadas (x, y) del nodo.
        restraints (Restraints): Restricciones del nodo [ux, uy, theta].
        dof (np.ndarray): Grados de libertad del nodo.
        forces (PointLoad): Fuerzas aplicadas en el nodo [fx, fy, mz].
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

        self.dof: np.ndarray = np.array([
            self.id * 3 - 2,  # DOF en x
            self.id * 3 - 1,  # DOF en y
            self.id * 3 - 0   # DOF en theta
        ], dtype=int)

        self.forces: PointLoad = PointLoad()  # Carga inicializada en 0
        

        # resultados
        self.desplacement: Optional[np.ndarray] = None
        self.reaction: Optional[np.ndarray] = None
        
    def add_restraints(self, restraints: "Restraints") -> None:
        """
        Asigna restricciones al nodo.

        Args:
            restraints (Restraints): Restricciones [ux, uy, theta].
        """
        self.restraints = restraints 

    def add_forces(self, forces: PointLoad) -> None:
        """
        Suma una carga puntual a las fuerzas existentes en el nodo.

        Args:
            forces (PointLoad): Carga puntual aplicada al nodo.
        """
        self.forces += forces

    def __str__(self) -> str:
        return f"Node {self.id}: {self.vertex}, Restraints: {self.restraints}, Forces: {self.forces}"
