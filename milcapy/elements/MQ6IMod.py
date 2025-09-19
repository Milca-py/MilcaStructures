import numpy as np
from typing import Union
from milcapy.core.node import Node
from milcapy.section.section import ShellSection
from milcapy.utils.types import ConstitutiveModelType
from milcapy.elements.quad4 import MembraneQuad4
from milcapy.elements.MQ6   import MembraneQuad6
from milcapy.elements.MQ6I  import MembraneQuad6I
from milcapy.elements.quad8 import MembraneQuad8
from milcapy.utils.types    import MembraneQuadElementType, IntegrationType


class MembraneQuad6IMod(MembraneQuad6):
    def __init__(
        self,
        id: int,
        node1: Node,
        node2: Node,
        node3: Node,
        node4: Node,
        section: ShellSection,
        state: Union[ConstitutiveModelType, str] = ConstitutiveModelType.PLANE_STRESS,
        ele_type: Union[MembraneQuadElementType, str] = MembraneQuadElementType.MQ6,
    ):
        super().__init__(id, node1, node2, node3, node4, section, state=state)

    def K_global(self) -> np.ndarray:
        """
        Matriz de rigidez global.
        """

        if self.ele_type == MembraneQuadElementType.MQ4:
            MQ4 = MembraneQuad4(self.id, self.node1, self.node2, self.node3, self.node4, self.section, state=self.state) # tiene rigidez de desplazamientos
            Kdisp = MQ4.global_stiffness_matrix()
        elif self.ele_type == MembraneQuadElementType.MQ6:
            return super().global_stiffness_matrix()
        elif self.ele_type == MembraneQuadElementType.MQ6I:
            MQ6I = MembraneQuad6I(self.id, self.node1, self.node2, self.node3, self.node4, self.section, state=self.state) # tiene rigidez de desplazamientos
            Kdisp = MQ6I.global_stiffness_matrix()
        elif self.ele_type == MembraneQuadElementType.MQ8Reduced:
            MQ8 = MembraneQuad8(self.id, self.node1, self.node2, self.node3, self.node4, self.section, state=self.state, integration_type=IntegrationType.REDUCED) # tiene rigidez de desplazamientos
            Kdisp = MQ8.global_stiffness_matrix()
        elif self.ele_type == MembraneQuadElementType.MQ8Complete:
            MQ8 = MembraneQuad8(self.id, self.node1, self.node2, self.node3, self.node4, self.section, state=self.state, integration_type=IntegrationType.COMPLETE) # tiene rigidez de desplazamientos
            Kdisp = MQ8.global_stiffness_matrix()

        Krot = super().global_stiffness_matrix()
        # Krot = self.Ki()
        # Kdisp = MQ6I.Ki()
        # T = self.get_transformation_matrix()
        dofDisp = [0, 1,      3, 4,     6, 7,     9, 10]
        # INSERTAMOS LOS K DISP EN LOS DOFDISP DE KROT
        Krot[np.ix_(dofDisp,dofDisp)] = Kdisp
        K = Krot
        # K_GLOBAL = T @ K @ T.T
        return K

    def force_vector(self) -> np.ndarray:
        """
        Vector de fuerzas nodales.
        """
        pass
