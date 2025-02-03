from typing import TYPE_CHECKING, Optional
import numpy as np
from loads.load import PointLoad
from utils.vertex import Vertex

if TYPE_CHECKING:
    from utils.custom_types import Restraints

class Node:
    """
    Representa un nodo en el modelo estructural.

    Atributos:
        id (int): Identificador único del nodo.
        vertex (Vertex): Coordenadas (x, y) del nodo.
        restraints (Optional[Restraints]): Restricciones del nodo [ux, uy, theta].
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
        self.restraints: Optional["Restraints"] = None  # Restricciones del nodo

        self.dof: np.ndarray = np.array([
            self.id * 3 - 2,  # DOF en x
            self.id * 3 - 1,  # DOF en y
            self.id * 3       # DOF en theta
        ], dtype=int)

        self.forces: PointLoad = PointLoad()  # Carga inicializada en 0

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
        if not isinstance(forces, PointLoad):
            raise TypeError("Las fuerzas deben ser una instancia de PointLoad.")

        self.forces += forces

    def __str__(self) -> str:
        return f"Node {self.id}: {self.vertex}, Restraints: {self.restraints}, Forces: {self.forces}"
    
    
    

