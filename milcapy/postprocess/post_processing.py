from dataclasses import dataclass
from typing import TYPE_CHECKING
from milcapy.postprocess.internal_forces import (
    integration_coefficients,
    axial_force,
    shear_force,
    bending_moment,
    slope,
    deflection,
    deformed,
    rigid_deformed
)

if TYPE_CHECKING:
    from milcapy.elements.system import SystemMilcaModel
    from milcapy.postprocess.results import Results


@dataclass
class PostProcessingOptions:
    """Opciones para el post-procesamiento de resultados estructurales."""

    factor: float  # Factor de escala para la visualización de resultados
    n: int         # Número de puntos para discretizar los elementos


class PostProcessing:
    """Clase para el post-procesamiento de resultados estructurales."""

    def __init__(
        self,
        system: "SystemMilcaModel",
        results: "Results",
        options: "PostProcessingOptions"
    ) -> None:
        """
        Inicializa el post-procesamiento para un sistema estructural.

        Args:
            system: Sistema estructural analizado
            options: Opciones de post-procesamiento
        """
        self.system = system
        self.results = results
        self.options = options

    def process_all_elements(self) -> None:
        """Calcula todos los resultados para cada elemento."""
        factor = self.options.factor
        n = self.options.n

        for element in self.system.element_map.values():
            # Calcular coeficientes de integración
            self.results.integration_coefficients_elements[element.id] = integration_coefficients(
                element)
            element.integration_coefficients = self.results.integration_coefficients_elements[
                element.id]

            # Calcular fuerzas y momentos
            self.results.values_axial_force_elements[element.id] = axial_force(
                element, factor, n)
            element.axial_force = self.results.values_axial_force_elements[element.id][1]

            self.results.values_shear_force_elements[element.id] = shear_force(
                element, factor, n)
            element.shear_force = self.results.values_shear_force_elements[element.id][1]

            self.results.values_bending_moment_elements[element.id] = bending_moment(
                element, factor, n)
            element.bending_moment = self.results.values_bending_moment_elements[element.id][1]

            # Calcular deformaciones
            self.results.values_slope_elements[element.id] = slope(
                element, factor, n)
            element.slope = self.results.values_slope_elements[element.id][1]

            self.results.values_deflection_elements[element.id] = deflection(
                element, factor, n)
            element.deflection = self.results.values_deflection_elements[element.id][1]

            self.results.values_deformed_elements[element.id] = deformed(
                element, factor)
            element.deformed_shape = self.results.values_deformed_elements[element.id]

            self.results.values_rigid_deformed_elements[element.id] = rigid_deformed(
                element, factor)
            element.rigid_deformed_shape = self.results.values_rigid_deformed_elements[
                element.id]
