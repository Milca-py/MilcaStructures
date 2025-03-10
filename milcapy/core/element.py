import numpy as np
from typing import TYPE_CHECKING, Optional
from milcapy.utils.geometry import angle_x_axis
from milcapy.loads.load import DistributedLoad
from milcapy.components.element_components import (
    local_stiffness_matrix,
    transformation_matrix,
    local_load_vector
)

if TYPE_CHECKING:
    from milcapy.core.node import Node
    from milcapy.utils.custom_types import ElementType
    from milcapy.core.section import Section


class Element:
    """Clase que representa un elemento estructural."""

    def __init__(
        self,
        id: int,
        type: "ElementType",
        node_i: "Node",
        node_j: "Node",
        section: "Section"
    ) -> None:
        """Inicializa un elemento estructural.

        Args:
            id (int): Identificador del elemento.
            type (ElementType): Tipo de elemento (viga, columna, etc.).
            node_i (Node): Nodo inicial.
            node_j (Node): Nodo final.
            section (Section): Sección transversal del elemento.
        """
        self.id = id
        self.type = type
        self.node_i = node_i
        self.node_j = node_j
        self.section = section
        
        # Mapa de grados de libertad
        self.dof_map: np.ndarray = np.concatenate([node_i.dof, node_j.dof])

        # Matrices y vectores
        self.local_stiffness_matrix: Optional[np.ndarray] = None
        self.transformation_matrix: Optional[np.ndarray] = None
        self.local_load_vector: Optional[np.ndarray] = None
        self.global_stiffness_matrix: Optional[np.ndarray] = None
        self.global_load_vector: Optional[np.ndarray] = None

        # Cargas distribuidas en sistema de coordenadas locales
        self.distributed_load: DistributedLoad = DistributedLoad()

        # Resultados
        self.displacement: Optional[np.ndarray] = None
        self.internal_forces: Optional[np.ndarray] = None
        
        # Resultados del postprocesamiento
        self.integration_coefficients: Optional[np.ndarray] = None
        self.axial_force: Optional[np.ndarray] = None
        self.shear_force: Optional[np.ndarray] = None
        self.bending_moment: Optional[np.ndarray] = None
        self.deflection: Optional[np.ndarray] = None
        self.slope: Optional[np.ndarray] = None
        self.deformed_shape: Optional[np.ndarray] = None

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

    def add_distributed_load(self, load: DistributedLoad) -> None:
        """Asigna una carga distribuida al elemento.

        Args:
            load (DistributedLoad): Carga distribuida a aplicar.
        """
        self.distributed_load += load

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
            v=self.section.material.poisson_ratio,
            k=self.section.timoshenko_coefficient,
            shear_effect=True,
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

    def compile(self) -> None:
        """Compila todas las matrices y vectores necesarios para el análisis."""
        self.compile_transformation_matrix()
        self.compile_local_stiffness_matrix()
        self.compile_local_load_vector()
        self.compile_global_stiffness_matrix()
        self.compile_global_load_vector()