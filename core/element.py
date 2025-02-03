import numpy as np
from typing import TYPE_CHECKING, Dict, Optional
from utils.geometry import angle_x_axis
from loads.load import DistributedLoad
from components.element_components import (
    local_stiffness_matrix,
    transformation_matrix,
    trapezoidal_load_vector
)

if TYPE_CHECKING:
    from core.node import Node
    from utils.custom_types import ElementType
    from core.section import Section


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

        self._dof_map: np.ndarray = np.concatenate([node_i.dof, node_j.dof])

        self._stiffness_matrix: Optional[np.ndarray] = None
        self._transformation_matrix: Optional[np.ndarray] = None
        self._load_vector: Optional[np.ndarray] = None

        self.stiffness_matrix_global: Optional[np.ndarray] = None
        self.load_vector_global: Optional[np.ndarray] = None

        self.distributed_load: DistributedLoad = DistributedLoad()

    @property
    def length(self) -> float:
        """Longitud del elemento."""
        return (self.node_i.vertex - self.node_j.vertex).modulus

    @property
    def angle_x_axis(self) -> float:
        """Ángulo del elemento respecto al eje X global."""
        return angle_x_axis(
            self.node_j.vertex.x - self.node_i.vertex.x,
            self.node_j.vertex.y - self.node_i.vertex.y
        )

    def add_distributed_load(self, load: DistributedLoad) -> None:
        """Asigna una carga distribuida al elemento.

        Args:
            load (DistributedLoad): Carga distribuida a aplicar.
        """
        self.distributed_load = load

    def compile_stiffness_matrix(self) -> None:
        """Compila la matriz de rigidez local del elemento."""
        self._stiffness_matrix = local_stiffness_matrix(
            modulus_elasticity=self.section.material.modulus_elasticity,
            moment_of_inertia=self.section.moment_of_inertia,
            area=self.section.area,
            length=self.length,
            poisson_ratio=self.section.material.poisson_ratio,
            timoshenko_coefficient=self.section.timoshenko_coefficient,
            def_shear=True,
        )

    def compile_transformation_matrix(self) -> None:
        """Compila la matriz de transformación del elemento."""
        self._transformation_matrix = transformation_matrix(
            angle=self.angle_x_axis
        )

    def compile_load_vector(self) -> None:
        """Determina el vector de fuerzas equivalentes debido a la carga distribuida."""
        self._load_vector = trapezoidal_load_vector(
            load_start=self.distributed_load.q_i,
            load_end=self.distributed_load.q_j,
            length=self.length
        )

    def compile_stiffness_matrix_global(self) -> None:
        """Compila la matriz de rigidez global del elemento."""
        if self._transformation_matrix is None or self._stiffness_matrix is None:
            raise ValueError("Debe compilar primero las matrices de transformación y rigidez local.")
        self.stiffness_matrix_global = (
            self._transformation_matrix.T @ self._stiffness_matrix @ self._transformation_matrix
        )

    def compile_load_vector_global(self) -> None:
        """Compila el vector de fuerzas global del elemento."""
        if self._transformation_matrix is None or self._load_vector is None:
            raise ValueError("Debe compilar primero las matrices de transformación y el vector de cargas locales.")
        self.load_vector_global = self._transformation_matrix.T @ self._load_vector
