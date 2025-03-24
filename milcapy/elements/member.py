import numpy as np
from typing import TYPE_CHECKING, Optional, Dict
from milcapy.utils.geometry import angle_x_axis
from milcapy.loads.load import DistributedLoad
from milcapy.utils.types import BeamTheoriesType
from milcapy.utils.element import (
    local_stiffness_matrix,
    transformation_matrix,
    local_load_vector
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
            L = self.length()
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

    def local_stiffness_matrix(self) -> np.ndarray:
        """Calcula la matriz de rigidez local del miembro.

        Returns:
            np.ndarray: Matriz de rigidez en local.
        """
        return local_stiffness_matrix(
            E=self.section.E(),
            I=self.section.I(),
            A=self.section.A(),
            L=self.length(),
            phi=self.phi(),
        )

    def local_load_vector(self) -> np.ndarray:
        """Calcula el vector de fuerzas equivalentes debido a la carga distribuida para el patron de carga actual.

        Returns:
            np.ndarray: Vector de fuerzas equivalentes en el sistema local.
        """
        load = self.get_distributed_load(self.current_load_pattern)
        return local_load_vector(
            L=self.length(),
            phi=self.phi(),
            q_i=load.q_i,
            q_j=load.q_j,
            p_i=load.p_i,
            p_j=load.p_j
        )

    def global_stiffness_matrix(self) -> np.ndarray:
        """Calcula la matriz de rigidez global del miembro.

        Returns:
            np.ndarray: Matriz de rigidez global.
        """
        return (
            self.transformation_matrix().T @ self.local_stiffness_matrix() @ self.transformation_matrix()
        )

    def global_load_vector(self) -> np.ndarray:
        """Calcula el vector de fuerzas global del miembro para el patron de carga actual.

        Returns:
            np.ndarray: Vector de fuerzas global.
        """
        if self.current_load_pattern is None:
            raise ValueError("Debe establecer un patrón de carga actual antes de calcular el vector de fuerzas global.")
        return self.transformation_matrix().T @ self.local_load_vector()
