from milcapy.core.node import Node
from milcapy.section.section import Section
import numpy as np
from milcapy.loads.load import DistributedLoad
from typing import Dict, Optional
from milcapy.utils.geometry import angle_x_axis

class TrussElement:
    """
    Elemento de armadura 3dof por nodo.
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

        self.distributed_load: Dict[str, DistributedLoad] = {}  # {pattern_name: DistributedLoad}
        self.displacements = None

        self.current_load_pattern: Optional[str] = None
        self.la = None; self.lb = None

    def length(self) -> float:
        """
        Longitud del miembro.
        """
        return (self.node_i.vertex.distance_to(self.node_j.vertex))

    def angle_x(self) -> float:
        """
        Ángulo del armadura respecto al eje X del sistema global.
        """
        return angle_x_axis(
            self.node_j.vertex.x - self.node_i.vertex.x,
            self.node_j.vertex.y - self.node_i.vertex.y
        )


    def transformation_matrix(self) -> np.ndarray:
        """
        Matriz de transformación del sistema local a global.
        """
        L = self.length()
        cx = (self.node_j.vertex.x - self.node_i.vertex.x) / L
        cy = (self.node_j.vertex.y - self.node_i.vertex.y) / L
        T = np.array([
            [cx, cy, 0, 0],
            [0, 0, cx, cy],
        ])
        return T

    def k_local(self) -> np.ndarray:
        """
        Matriz de rigidez local del elemento.
        """
        return self.E*self.A/self.length()*np.array([[1, -1], [-1, 1]])

    def global_stiffness_matrix(self) -> np.ndarray:
        """
        Matriz de rigidez global del elemento.
        """
        return self.transformation_matrix().T @ self.k_local() @ self.transformation_matrix()

    def global_load_vector(self) -> np.ndarray:
        """
        Vector de cargas global del elemento.
        """
        return self.transformation_matrix().T @ self.q_local()

    def set_current_load_pattern(self, load_pattern_name: str) -> None:
        """
        Establece el patrón de carga actual del miembro.
        """
        self.current_load_pattern = load_pattern_name

    def set_distributed_load(self, load: DistributedLoad) -> None:
        """
        Asigna una carga distribuida al miembro para el patrón de carga actual.
        """
        if self.current_load_pattern is None:
            raise ValueError("Debe establecer un patrón de carga actual antes de asignar una carga distribuida.")
        self.distributed_load[self.current_load_pattern] = load

    def get_distributed_load(self, load_pattern_name: str) -> Optional["DistributedLoad"]:
        """
        Obtiene la carga distribuida para el patrón de carga actual.
        """
        load = self.distributed_load.get(load_pattern_name, None)
        if load is None:
            return DistributedLoad()
        return load

    def local_stiffness_matrix(self) -> np.ndarray:
        """
        Matriz de rigidez local del elemento.
        """
        return self.k_local()

    def q_local(self) -> np.ndarray:
        """
        Vector de cargas distribuidas equivalentes en sistema local.
        """
        load = self.get_distributed_load(self.current_load_pattern)
        vector = np.array([
            2*load.p_i + load.p_j,
            2*load.p_j + load.p_i,
        ])
        return vector

    def global_load_vector(self) -> np.ndarray:
        """
        Vector de cargas global del elemento.
        """
        return self.transformation_matrix().T @ self.q_local()