from typing import TYPE_CHECKING
from milcapy.elements.internal_forces import (
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
    from milcapy.elements.element import Element


class PostProcessingOptions:
    """Opciones para el post-procesamiento de resultados estructurales."""
    
    def __init__(self, factor: float, n: int) -> None:
        """
        Inicializa las opciones de post-procesamiento.
        
        Args:
            factor: Factor de escala para la visualización de resultados
            n: Número de puntos para discretizar los elementos
        """
        self.factor = factor
        self.n = n


class PostProcessing:
    """Clase para el post-procesamiento de resultados estructurales."""
    
    def __init__(
        self, 
        system: "SystemMilcaModel",
        options: "PostProcessingOptions" = None
    ) -> None:
        """
        Inicializa el post-procesamiento para un sistema estructural.
        
        Args:
            system: Sistema estructural analizado
            options: Opciones de post-procesamiento
        """
        self.system = system
        self.results = self.system.results
        self.options = options or PostProcessingOptions(factor=1, n=100)
        
        # Diccionarios de resultados
        self.integration_coefficients_elements = {}
        self.values_axial_force_elements = {}
        self.values_shear_force_elements = {}
        self.values_bending_moment_elements = {}
        self.values_slope_elements = {}
        self.values_deflection_elements = {}
        self.values_deformed_elements = {}
        
        # Calcular resultados para cada elemento
        self.process_all_elements()
    
    def process_all_elements(self) -> None:
        """Calcula todos los resultados para cada elemento."""
        factor = self.options.factor
        n = self.options.n
        
        for element in self.system.element_map.values():
            # Calcular coeficientes de integración
            self.integration_coefficients_elements[element.id] = integration_coefficients(element)
            element.integration_coefficients = self.integration_coefficients_elements[element.id]
            
            # Calcular fuerzas y momentos
            self.values_axial_force_elements[element.id] = axial_force(element, factor, n)
            element.axial_force = self.values_axial_force_elements[element.id][1]
            
            self.values_shear_force_elements[element.id] = shear_force(element, factor, n)
            element.shear_force = self.values_shear_force_elements[element.id][1]
            
            self.values_bending_moment_elements[element.id] = bending_moment(element, factor, n)
            element.bending_moment = self.values_bending_moment_elements[element.id][1]
            
            # Calcular deformaciones
            self.values_slope_elements[element.id] = slope(element, factor, n)
            element.slope = self.values_slope_elements[element.id][1]
            
            self.values_deflection_elements[element.id] = deflection(element, factor, n)
            element.deflection = self.values_deflection_elements[element.id][1]
            
            self.values_deformed_elements[element.id] = deformed(element, factor)
            element.deformed_shape = self.values_deformed_elements[element.id][1]
    
    def update_with_new_options(self, new_options: "PostProcessingOptions") -> None:
        """
        Actualiza las opciones y recalcula todos los resultados.
        
        Args:
            new_options: Nuevas opciones de post-procesamiento
        """
        self.options = new_options
        self.process_all_elements()



