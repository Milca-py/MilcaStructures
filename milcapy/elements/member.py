import numpy as np
from typing import TYPE_CHECKING, Optional, Dict
from milcapy.utils.geometry import angle_x_axis
from milcapy.loads.load import DistributedLoad
from milcapy.utils.element import (
    local_stiffness_matrix,
    transformation_matrix,
    local_load_vector
)

if TYPE_CHECKING:
    from milcapy.core.node import Node
    from milcapy.utils.types import ElementType
    from milcapy.section.section import Section


class Member:
    """Clase que representa un miembro estructural."""

    def __init__(
        self,
        id: int,
        node_i: "Node",
        node_j: "Node",
        section: "Section",
        type: "ElementType",
    ) -> None:
        """Inicializa un miembro estructural."""

        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.section = section
        self.type = type

        # Mapa de grados de libertad
        self.dof: np.ndarray = np.concatenate([node_i.dof, node_j.dof]) # [dofs node_i, dofs node_j]

        # Cargas distribuidas
        self.distributed_load: Dict[str, DistributedLoad] = {}  # {pattern_name: DistributedLoad}

        # Patrón de carga actual
        self.current_load_pattern: Optional[str] = None

    @property
    def length(self) -> float:
        """Longitud del miembro."""
        return (self.node_i.vertex.distance_to(self.node_j.vertex))

    @property
    def angle_x(self) -> float:
        """Ángulo del miembro respecto al eje X del sistema global."""
        return angle_x_axis(
            self.node_j.vertex.x - self.node_i.vertex.x,
            self.node_j.vertex.y - self.node_i.vertex.y
        )

    @property
    def phi(self) -> float:
        """Ángulo de corte para efectos de deformación por cortante (parámetro phi).

        phi = (12 * E * I) / (L**2 * A * k * G)
        """
        E = self.section.material.modulus_elasticity
        I = self.section.moment_of_inertia
        L = self.length
        A = self.section.area
        k = self.section.timoshenko_coefficient
        G = self.section.material.shear_modulus

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

    def transformation_matrix(self) -> np.ndarray:
        """Compila la matriz de transformación del miembro.

        Returns:
            np.ndarray: Matriz de transformación.
        """
        return transformation_matrix(
            angle=self.angle_x
        )

    def local_stiffness_matrix(self) -> np.ndarray:
        """Compila la matriz de rigidez local del miembro.

        Returns:
            np.ndarray: Matriz de rigidez en local.
        """
        return local_stiffness_matrix(
            E=self.section.material.modulus_elasticity,
            I=self.section.moment_of_inertia,
            A=self.section.area,
            L=self.length,
            phi=self.phi,
        )

    def local_load_vector(self) -> np.ndarray:
        """Determina el vector de fuerzas equivalentes debido a la carga distribuida.

        Returns:
            np.ndarray: Vector de fuerzas equivalentes en el sistema local.
        """
        return local_load_vector(
            L=self.length,
            phi=self.phi,
            q_i=self.distributed_load[self.current_load_pattern].q_i,
            q_j=self.distributed_load[self.current_load_pattern].q_j,
            p_i=self.distributed_load[self.current_load_pattern].p_i,
            p_j=self.distributed_load[self.current_load_pattern].p_j
        )

    def global_stiffness_matrix(self) -> np.ndarray:
        """Compila la matriz de rigidez global del miembro.

        Returns:
            np.ndarray: Matriz de rigidez global.
        """
        return (
            self.transformation_matrix().T @ self.local_stiffness_matrix() @ self.transformation_matrix()
        )

    def global_load_vector(self) -> np.ndarray:
        """Compila el vector de fuerzas global del miembro.

        Returns:
            np.ndarray: Vector de fuerzas global.
        """
        return self.transformation_matrix().T @ self.local_load_vector()
