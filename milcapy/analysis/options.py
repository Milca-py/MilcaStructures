class AnalysisOptions:
    """Define opciones de análisis."""

    def __init__(self) -> None:
        self.include_shear_deformations = True
        self._shear_coefficient_method = "TIMOSHENKO"

    @property
    def shear_coefficient_method(self) -> str:
        """Método del coeficiente de cortante en mayúsculas."""
        return self._shear_coefficient_method

    @shear_coefficient_method.setter
    def shear_coefficient_method(self, value: str) -> None:
        """Convierte el método en mayúsculas antes de almacenarlo."""
        self._shear_coefficient_method = value.upper()
