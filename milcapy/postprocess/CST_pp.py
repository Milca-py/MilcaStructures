from milcapy.elements.CST import MembraneTriangle
from typing import Optional
import numpy as np

class PP_CST(MembraneTriangle):
    """
    Un postprocesador para el elemento "Constant Straint Triangle" (CST)

    Permite obtener las deformaciones y tensiones en cada nodo del elemento CST

    Attributes:
        displacements (Optional[np.ndarray]): array con los valores de desplazamiento en cada nodo [ux1, uy1, ux2, uy2, ux3, uy3]
        coordinates (np.ndarray): array con las coordenadas de cada nodo [x1, y1, x2, y2, x3, y3]
    """
    def __init__(self):
        self.displacements: Optional[np.ndarray] = None

    def strains(self):
        """
        Calcula las deformaciones en cada nodo del elemento CST

        Returns:
            np.ndarray: array con las deformaciones en cada nodo
        """
        return self.B @ self.displacements

    def stresses(self):
        """
        Calcula las tensiones en cada nodo del elemento CST

        Returns:
            np.ndarray: array con las tensiones en cada nodo
        """
        return self.D @ self.strains()

    def process_builder(self, cst: MembraneTriangle, displacements: np.ndarray):
        """
        Crea un nuevo objeto PP_CST y asigna los valores de desplazamiento y coordenadas

        Args:
            cst (MembraneTriangle): objeto CST del que se va a obtener la informacion
            displacements (np.ndarray): array con los valores de desplazamiento en cada nodo
        """
        super().__init__(0, cst.node1, cst.node2, cst.node3, cst.section, cst.state)
        self.displacements = displacements
        self.coordinates = np.concatenate(([cst.node1.vertex.coordinates, cst.node2.vertex.coordinates, cst.node3.vertex.coordinates]))

    def deformed_shape(self) -> np.ndarray:
        """
        Calcula la forma deformada del elemento CST

        Returns:
            np.ndarray: array con las coordenadas de la forma deformada
        """
        return self.coordinates + self.displacements