from typing import Optional


class Material:
    """
    Clase base para representar un material estructural.

    Args:
        name (str): Nombre del material.
        modulus_elasticity (float): Módulo de elasticidad (Pa).
        poisson_ratio (float): Coeficiente de Poisson (-1 < v < 0.5).
        specific_weight (float): Peso específico o densidad (N/m³ o kg/m³ según el sistema de unidades).
    """

    def __init__(
        self,
        name: str,
        modulus_elasticity: float,
        poisson_ratio: float,
        specific_weight: float,
        alpha: Optional[float] = 1e-6,
        beta: Optional[float] = 1e-6,
        gamma: Optional[float] = 1e-6
        ) -> None:

        if modulus_elasticity <= 0:
            raise ValueError("El módulo de elasticidad (E) debe ser mayor que 0.")
        if not (0 <= poisson_ratio < 0.5):
            raise ValueError("El coeficiente de Poisson (v) debe estar en el rango (-1, 0.5).")
        if specific_weight < 0:
            raise ValueError("El peso específico o densidad (g) debe ser mayor o igual que 0.")

        self.name: str = name
        self.E: float = modulus_elasticity
        self.v: float = poisson_ratio
        self.g: float = specific_weight
        self.kg: float = 1 # Modificador de peso
        self.alpha: float = alpha # Coeficiente de expansión lineal
        self.beta: float = beta # Coeficiente de expansión superficial
        self.gamma: float = gamma # Coeficiente de expansión volumétrica

    @property
    def G(self) -> float:
        """
        Calcula el módulo de Corte (G) basado en el módulo de elasticidad y el coeficiente de Poisson.
        """
        return self.E / (2 * (1 + self.v))

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}: {self.name}, "
                f"E={self.E:.2f}, v={self.v:.3f}, "
                f"G={self.G:.2f}, g={self.g:.2f}, "
                f"alpha={self.alpha:.6e}, beta={self.beta:.6e}, gamma={self.gamma:.6e}")


class ConcreteMaterial(Material):
    """Clase para representar materiales de concreto."""
    pass


class SteelMaterial(Material):
    """Clase para representar materiales de acero estructural."""
    pass


class GenericMaterial(Material):
    """Clase para representar materiales genéricos sin clasificación específica."""
    pass
