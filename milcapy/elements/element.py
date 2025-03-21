import numpy as np
from typing import TYPE_CHECKING, Optiona
from milcapy.utils.element import (
    local_stiffness_matrix,
    transformation_matrix,
    local_load_vector
)

if TYPE_CHECKING:
    from milcapy.core.node import Node
    from milcapy.utils.types import ElementType
    from milcapy.section.section import Section


class Element:
    """Clase que representa un elemento estructural."""

    def __init__(
        self,
        id: int,
        node_i: "Node",
        node_j: "Node",
        section: "Section",
        type: "ElementType",
    ) -> None:
        """Inicializa un elemento estructural."""
        
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.section = section
        self.type = type
        
        # Mapa de grados de libertad
        self.dof_map: np.ndarray = np.concatenate([node_i.dof, node_j.dof])

        # Matrices y vectores
        self.transformation_matrix: Optional[np.ndarray] = None
        self.local_stiffness_matrix: Optional[np.ndarray] = None
        self.global_stiffness_matrix: Optional[np.ndarray] = None

        self.local_load_vector: Dict[str, np.ndarray] = {}
        self.global_load_vector: Dict[str, np.ndarray] = {}

        # Cargas distribuidas en sistema de coordenadas locales
        self.distributed_load: Dict[str, DistributedLoad] = {}

        # Patrón de carga actual
        self.current_load_pattern: Optional[str] = None

    @property
    def length(self) -> float:
        """Longitud del elemento."""
        return (self.node_i.vertex - self.node_j.vertex).modulus()

    @property
    def angle_x(self) -> float:
        """Ángulo del elemento respecto al eje X global."""
        return angle_x_axis(
            self.node_j.vertex.x - self.node_i.vertex.x,
            self.node_j.vertex.y - self.node_i.vertex.y
        )
    
    @property
    def shear_angle(self) -> float:
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
        """Establece el patrón de carga actual del elemento."""
        self.current_load_pattern = load_pattern_name

    def add_distributed_load(self, load: DistributedLoad) -> None:
        """Asigna una carga distribuida al elemento.

        Args:
            load (DistributedLoad): Carga distribuida a aplicar.
        """
        if self.current_load_pattern is None:
            raise ValueError("Debe establecer un patrón de carga actual antes de asignar una carga distribuida.")
        self.distributed_load[self.current_load_pattern] += load

    def compile_transformation_matrix(self) -> None:
        """Compila la matriz de transformación del elemento."""
        self.transformation_matrix = transformation_matrix(
            angle=self.angle_x
        )

    def compile_local_stiffness_matrix(self) -> None:
        """Compila la matriz de rigidez local del elemento."""
        self.local_stiffness_matrix = local_stiffness_matrix(
            E=self.section.material.modulus_elasticity,
            I=self.section.moment_of_inertia,
            A=self.section.area,
            L=self.length,
            phi=self.shear_angle,
        )

    def compile_local_load_vector(self) -> None:
        """Determina el vector de fuerzas equivalentes debido a la carga distribuida."""
        self.local_load_vector = local_load_vector(
            L=self.length,
            phi=self.shear_angle,
            q_i=self.distributed_load.q_i,
            q_j=self.distributed_load.q_j,
            p_i=self.distributed_load.p_i,
            p_j=self.distributed_load.p_j
        )

    def compile_global_stiffness_matrix(self) -> None:
        """Compila la matriz de rigidez global del elemento."""
        if self.transformation_matrix is None or self.local_stiffness_matrix is None:
            raise ValueError("Debe compilar primero las matrices de transformación y rigidez local.")
        
        self.global_stiffness_matrix = (
            self.transformation_matrix.T @ self.local_stiffness_matrix @ self.transformation_matrix
        )

    def compile_global_load_vector(self) -> None:
        """Compila el vector de fuerzas global del elemento."""
        if self.transformation_matrix is None or self.local_load_vector is None:
            raise ValueError("Debe compilar primero las matrices de transformación y el vector de cargas locales.")
        
        self.global_load_vector = self.transformation_matrix.T @ self.local_load_vector