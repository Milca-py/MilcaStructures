import numpy as np
from typing import TYPE_CHECKING, Optional, Dict
from milcapy.utils.geometry import angle_x_axis
from milcapy.loads.load import DistributedLoad
from milcapy.utils.types import BeamTheoriesType
from milcapy.utils.element import (
    local_stiffness_matrix,
    transformation_matrix,
    length_offset_transformation_matrix,
    length_offset_q,
    q_phi
)

if TYPE_CHECKING:
    from milcapy.core.node import Node
    from milcapy.utils.types import MemberType
    from milcapy.section.section import Section


class Member:
    """Clase que representa un miembro estructural."""

    def __init__(
        self,
        id: int,
        node_i: "Node",
        node_j: "Node",
        section: "Section",
        member_type: "MemberType",
        beam_theory: "BeamTheoriesType",
    ) -> None:
        """Inicializa un miembro estructural."""

        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.section = section
        self.member_type = member_type
        self.beam_theory = beam_theory

        # Mapa de grados de libertad
        self.dofs: np.ndarray = np.concatenate([node_i.dofs, node_j.dofs]) # [dofs node_i, dofs node_j]

        # Cargas distribuidas
        self.distributed_load: Dict[str, DistributedLoad] = {}  # {pattern_name: DistributedLoad}

        # Patrón de carga actual
        self.current_load_pattern: Optional[str] = None

        # Topicos
        self.la: Optional[float] = None    # Longitud del brazo rigido inicial
        self.lb: Optional[float] = None    # Longitud del brazo rigido final
        self.qla: Optional[bool] = None    # Indica si hay cargas en el brazo rigido inicial
        self.qlb: Optional[bool] = None    # Indica si hay cargas en el brazo rigido final

    def length(self) -> float:
        """Longitud del miembro."""
        return (self.node_i.vertex.distance_to(self.node_j.vertex))

    def angle_x(self) -> float:
        """Ángulo del miembro respecto al eje X del sistema global."""
        return angle_x_axis(
            self.node_j.vertex.x - self.node_i.vertex.x,
            self.node_j.vertex.y - self.node_i.vertex.y
        )

    def phi(self) -> float:
        """Ángulo de corte para efectos de deformación por cortante (parámetro phi).

        phi = (12 * E * I) / (L**2 * A * k * G)
        """
        if self.beam_theory == BeamTheoriesType.EULER_BERNOULLI:
            return 0

        elif self.beam_theory == BeamTheoriesType.TIMOSHENKO:
            E = self.section.E()
            I = self.section.I()
            L = self.length() - (self.la or 0) - (self.lb or 0)
            A = self.section.A()
            k = self.section.k()
            G = self.section.G()

            return (12 * E * I) / (L**2 * A * k * G)

    def set_current_load_pattern(self, load_pattern_name: str) -> None:
        """Establece el patrón de carga actual del miembro."""
        self.current_load_pattern = load_pattern_name

    def set_distributed_load(self, load: DistributedLoad) -> None:
        """Asigna una carga distribuida al miembro para el patrón de carga actual.

        Args:
            load (DistributedLoad): Carga distribuida a aplicar.
        """
        if self.current_load_pattern is None:
            raise ValueError("Debe establecer un patrón de carga actual antes de asignar una carga distribuida.")
        self.distributed_load[self.current_load_pattern] = load

    def get_distributed_load(self, load_pattern_name: str) -> Optional["DistributedLoad"]:
        """Obtiene la carga distribuida para el patrón de carga actual.

        Returns:
            Optional["DistributedLoad"]: Carga distribuida.
        """
        load = self.distributed_load.get(load_pattern_name, None)
        if load is None:
            return DistributedLoad()
        return load

    def transformation_matrix(self) -> np.ndarray:
        """Calcula la matriz de transformación del miembro.

        Returns:
            np.ndarray: Matriz de transformación.
        """
        return transformation_matrix(
            angle=self.angle_x()
        )

    def length_offset_transformation_matrix(self) -> np.ndarray:
        """Matriz de transformación para miembros con brazos rígidos (identidad si no hay brazos)."""
        if self.la is None and self.lb is None:
            return np.eye(6)  # Matriz identidad 6x6
        H = length_offset_transformation_matrix(self.la or 0, self.lb or 0)
        return H

    def local_stiffness_matrix(self) -> np.ndarray:
        """Calcula la matriz de rigidez local del miembro.

        Returns:
            np.ndarray: Matriz de rigidez en local.
        """
        if self.la is None and self.lb is None:
            k = local_stiffness_matrix(
                E=self.section.E(),
                I=self.section.I(),
                A=self.section.A(),
                L=self.length(),
                le=self.length(),
                phi=self.phi(),
            )
            return k
        else:
            H = self.length_offset_transformation_matrix()
            k = local_stiffness_matrix(
                E=self.section.E(),
                I=self.section.I(),
                A=self.section.A(),
                L=self.length(),
                le=self.length() - (self.la or 0) - (self.lb or 0),
                phi=self.phi(),
            )
            return H.T @ k @ H

    def flexible_stiffness_matrix(self) -> np.ndarray:
        """Calcula la matriz de rigidez de la parte flexible del miembro."""
        k = local_stiffness_matrix(
            E=self.section.E(),
            I=self.section.I(),
            A=self.section.A(),
            L=self.length(),
            le=self.length() - (self.la or 0) - (self.lb or 0),
            phi=self.phi(),
        )
        return k

    def local_load_vector(self) -> np.ndarray:
        """Calcula el vector de cargas distribuidas equivalentes en sistema local."""
        load = self.get_distributed_load(self.current_load_pattern)
        if self.la is None and self.lb is None:
            q = q_phi(
                L=self.length(),
                phi=self.phi(),
                qi=load.q_i,
                qj=load.q_j,
                pi=load.p_i,
                pj=load.p_j
            )
            return q
        else:
            q = length_offset_q(
                self.length(),
                self.phi(),
                load.q_i,
                load.q_j,
                load.p_i,
                load.p_j,
                self.la,
                self.lb,
                self.qla,
                self.qlb,
            )
            return q

    def global_stiffness_matrix(self) -> np.ndarray:
        """Calcula la matriz de rigidez global del miembro.

        Returns:
            np.ndarray: Matriz de rigidez global.
        """
        T = self.transformation_matrix()
        ke = self.local_stiffness_matrix()

        return T.T @ ke @ T

    def global_load_vector(self) -> np.ndarray:
        """Calcula el vector de fuerzas global del miembro para el patron de carga actual.

        Returns:
            np.ndarray: Vector de fuerzas global.
        """
        if self.current_load_pattern is None:
            raise ValueError("Debe establecer un patrón de carga actual antes de calcular el vector de fuerzas global.")
        T = self.transformation_matrix()
        q = self.local_load_vector()

        return T.T @ q
