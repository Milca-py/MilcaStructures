from abc import ABC, abstractmethod

class AnalysisOptions(ABC):
    """
    Clase base abstracta para opciones de análisis estructural.
    Define atributos comunes y métodos generales.
    """

    def __init__(
        self,
        include_shear_deformations: bool,
        shear_coefficient_method: str,
    ) -> None:
        """
        Inicializa las opciones de análisis estructural.

        Args:
            include_shear_deformations (bool): Indica si se consideran deformaciones por cortante.
            shear_coefficient_method (str): Método para el coeficiente de Timoshenko. 
                                            Opciones: "Timoshenko", "Cowper".
        """
        self.include_shear_deformations = include_shear_deformations
        self.shear_coefficient_method = shear_coefficient_method

    @property
    @abstractmethod
    def analysis_type(self) -> str:
        """Atributo para definir el tipo de análisis."""
        pass

    def validate(self) -> bool:
        """
        Valida la configuración del análisis estructural.

        Returns:
            bool: True si la configuración es válida, False en caso contrario.
        """
        valid_methods = {"Timoshenko", "Cowper"}

        if self.shear_coefficient_method not in valid_methods:
            print(
                f"Error: Método de coeficiente no válido: {self.shear_coefficient_method}. Opciones válidas: {valid_methods}")
            return False

        return True


class LinearStaticOptions(AnalysisOptions):
    """Opciones específicas para análisis estático lineal."""

    def __init__(
        self,
        include_shear_deformations: bool = True,
        shear_coefficient_method: str = "Timoshenko" # v=0
    ) -> None:
        super().__init__(include_shear_deformations, shear_coefficient_method)

    @property
    def analysis_type(self) -> str:
        """Retorna el tipo de análisis."""
        return "linear_static"
