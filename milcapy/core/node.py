from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
from milcapy.loads.load import PointLoad

if TYPE_CHECKING:
    from milcapy.utils.custom_types import Restraints
    from milcapy.utils.vertex import Vertex


class Node:
    """
    Representa un nodo en el modelo estructural.

    Atributos:
        id (int): Identificador único del nodo.
        vertex (Vertex): Coordenadas (x, y) del nodo.
        restraints (Restraints): Restricciones del nodo [ux, uy, theta].
        dof (np.ndarray): Grados de libertad del nodo.
        forces (PointLoad): Fuerzas aplicadas en el nodo [fx, fy, mz].
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
        self.forces: PointLoad = PointLoad()  # Carga inicializada en 0
        
        # Resultados del análisis
        self.displacement: Optional[np.ndarray] = None  # Corregido "desplacement" a "displacement"
        self.reaction: Optional[np.ndarray] = None
    
    @property
    def is_restrained(self) -> bool:
        """
        Verifica si el nodo tiene alguna restricción.

        Returns:
            bool: True si el nodo tiene al menos una restricción, False en caso contrario.
        """
        return any(self.restraints)
    
    @property
    def restrained_dof(self) -> Tuple[int, ...]:
        """
        Obtiene los índices de los grados de libertad restringidos.

        Returns:
            Tuple[int, ...]: Índices de los grados de libertad restringidos.
        """
        return tuple(dof for dof, restrained in zip(self.dof, self.restraints) if restrained)

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
    
    def clear_forces(self) -> None:
        """
        Elimina todas las fuerzas aplicadas al nodo.
        """
        self.forces = PointLoad()
    
    def reset_results(self) -> None:
        """
        Reinicia los resultados de análisis (desplazamientos y reacciones).
        """
        self.displacement = None
        self.reaction = None

    def __str__(self) -> str:
        """
        Representación en cadena del nodo.
        
        Returns:
            str: Descripción del nodo.
        """
        return f"Node {self.id}: {self.vertex}, Restraints: {self.restraints}, Forces: {self.forces}"
    
    def __repr__(self) -> str:
        """
        Representación formal del nodo para depuración.
        
        Returns:
            str: Representación formal del nodo.
        """
        return f"Node(id={self.id}, vertex={self.vertex})"