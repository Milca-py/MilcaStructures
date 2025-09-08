from milcapy.core.node import Node
from milcapy.section.section import Section
import numpy as np

class TrussElement:
    """
    Elemento de Armadura 3dof por nodo.
    """
    def __init__(
        self,
        id: int,
        node_i: Node,
        node_j: Node,
        section: Section,
    ) -> None:
        """
        Inicializa el elemento de armadura 3dof por nodo.

        Args:
            id (int): Identificador del elemento.
            node_i (Node): Primer nodo.
            node_j (Node): Segundo nodo.
            section (Section): Sección del elemento tipo area (shell).
        """
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.section = section
        self.E = section.E()
        self.A = section.A()
        self.dofs = np.concatenate((node_i.dofs[:2], node_j.dofs[:2]))
        self.length = np.linalg.norm(self.node_j.vertex.coordinates - self.node_i.vertex.coordinates)

        self.Tlg = self.transform_matrix()

        self.kl = self.E*self.A/self.length*np.array([[1, -1], [-1, 1]])

        self.Ki = self.Tlg.T @ self.kl @ self.Tlg

        self.forces = None
        self.displacements = None

    def transform_matrix(self) -> np.ndarray:
        """
        Matriz de transformación.
        """
        L = self.length
        cx = (self.node_j.vertex.x - self.node_i.vertex.x) / L
        cy = (self.node_j.vertex.y - self.node_i.vertex.y) / L
        T = np.array([
            [cx, cy, 0, 0],
            [0, 0, cx, cy],
        ])
        return T

    def global_stiffness_matrix(self) -> np.ndarray:
        """
        Matriz de rigidez global.
        """
        return self.Ki