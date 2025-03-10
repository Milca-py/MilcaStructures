class Material:
    """
    Clase base para representar un material estructural.
    
    Args:
        name (str): Nombre del material.
        modulus_elasticity (float): Módulo de elasticidad (Pa).
        poisson_ratio (float): Coeficiente de Poisson (-1 < poisson_ratio < 0.5).
        specific_weight (float): Peso específico o densidad (N/m³ o kg/m³ según el sistema de unidades).
    """

    def __init__(
        self,
        name: str,
        modulus_elasticity: float,
        poisson_ratio: float,
        specific_weight: float
        ) -> None:

        if modulus_elasticity <= 0:
            raise ValueError("El módulo de elasticidad (modulus_elasticity) debe ser mayor que 0.")
        if not (0 <= poisson_ratio < 0.5):
            raise ValueError("El coeficiente de Poisson (poisson_ratio) debe estar en el rango (-1, 0.5).")
        if specific_weight < 0:
            raise ValueError("El peso específico o densidad (specific_weight) debe ser mayor o igual que 0.")

        self.name: str = name
        self.modulus_elasticity: float = modulus_elasticity
        self.poisson_ratio: float = poisson_ratio
        self.specific_weight: float = specific_weight

    @property
    def shear_modulus(self) -> float:
        """
        Calcula el módulo de rigidez (G) basado en el módulo de elasticidad y el coeficiente de Poisson.
        """
        return self.modulus_elasticity / (2 * (1 + self.poisson_ratio))

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}: {self.name}, "
                f"E={self.modulus_elasticity:.2f}, v={self.poisson_ratio:.3f}, "
                f"G={self.shear_modulus:.2f}, g={self.specific_weight:.2f}")


class ConcreteMaterial(Material):
    """Clase para representar materiales de concreto."""
    pass


class SteelMaterial(Material):
    """Clase para representar materiales de acero estructural."""
    pass


class GenericMaterial(Material):
    """Clase para representar materiales genéricos sin clasificación específica."""
    pass
