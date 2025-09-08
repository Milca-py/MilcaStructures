from milcapy.elements.MQ6 import MembraneQuad6
from milcapy.elements.MQ6I import MembraneQuad6I
from milcapy.core.node import Node
from milcapy.section.section import Section
import numpy as np

class MembraneQuad6IMod(MembraneQuad6):
    def __init__(
        self,
        id: int,
        node1: Node,
        node2: Node,
        node3: Node,
        node4: Node,
        section: Section,
    ):
        super().__init__(id, node1, node2, node3, node4, section)

    def K_global(self) -> np.ndarray:
        """
        Matriz de rigidez global.
        """
        MQ6I = MembraneQuad6I(self.id, self.node1, self.node2, self.node3, self.node4, self.section) # tiene rigidez de desplazamientos
        Krot = super().global_stiffness_matrix()
        Kdisp = MQ6I.global_stiffness_matrix()
        # Krot = self.Ki()
        # Kdisp = MQ6I.Ki()
        # T = self.get_transformation_matrix()
        dofDisp = [0, 1,      3, 4,     6, 7,     9, 10]
        Krot[np.ix_(dofDisp,dofDisp)] = Kdisp
        K = Krot
        # K_GLOBAL = T @ K @ T.T
        return K

    def force_vector(self) -> np.ndarray:
        """
        Vector de fuerzas nodales.
        """
        pass
