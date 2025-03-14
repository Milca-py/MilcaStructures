from abc import ABC, abstractmethod
from milcapy.elements.material import Material

class Section(ABC):
    """
    Clase base para representar una sección estructural.
    """
    def __init__(
        self,
        name: str,
        material: "Material"
    ):
        """
        Inicializa una sección estructural.

        Args:
            name (str): Nombre de la sección.
            material (Material): Material asociado a la sección.
        """
        self.name = name
        self.material = material

    @abstractmethod
    def area(self) -> float:
        """Área de la sección transversal."""
        pass

    @abstractmethod
    def moment_of_inertia(self) -> float:
        """Momento de inercia de la sección."""
        pass

    @abstractmethod
    def timoshenko_coefficient(self) -> float:
        """Coeficiente de corte o Timoshenko de la sección."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: name={self.name}, material=({self.material.name}), A={self.area:.3f}, I={self.moment_of_inertia:.2f}, k={self.shear_coefficient:.2f}"


class RectangularSection(Section):
    """
    Clase para representar una sección rectangular.
    """
    def __init__(
        self,
        name: str,
        material: "Material",
        base: float,
        height: float
    ):
        """
        Inicializa una sección rectangular.

        Args:
            name (str): Nombre de la sección.
            material (Material): Material asociado a la sección.
            base (float): Base de la sección (en unidades de longitud).
            height (float): Altura de la sección (en unidades de longitud).

        Raises:
            ValueError: Si la base o la altura son menores o iguales a cero.
        """
        if base <= 0 or height <= 0:
            raise ValueError("La base y la altura deben ser mayores que cero.")
        super().__init__(name, material)
        self.base = base
        self.height = height

    @property
    def area(self) -> float:
        """Área de la sección transversal."""
        return self.base * self.height

    @property
    def moment_of_inertia(self) -> float:
        """Momento de inercia de la sección."""
        return self.base * self.height ** 3 / 12

    @property
    def timoshenko_coefficient(self) -> float:
        """Coeficiente de corte de Timoshenko de la sección."""
        return 5/6 #10 * (1 + self.material.poisson_ratio) / (12 + 11 * self.material.poisson_ratio)


class CircularSection(Section):
    """
    Clase para secciones circulares.
    """
    def __init__(
        self,
        name: str,
        material: "Material",
        radius: float
    ):
        """
        Inicializa una sección circular.

        Args:
            name (str): Nombre de la sección.
            material (Material): Material asociado a la sección.
            radius (float): Radio de la sección.

        Raises:
            ValueError: Si el radio es menor o igual a cero.
        """
        super().__init__(name, material)
        if radius <= 0:
            raise ValueError("El radio debe ser positivo.")
        self.radius = radius

    @property
    def area(self) -> float:
        """Área de la sección transversal."""
        from math import pi
        return pi * self.radius ** 2

    @property
    def moment_of_inertia(self) -> float:
        """Momento de inercia de la sección."""
        from math import pi
        return (pi * self.radius ** 4) / 4

    @property
    def timoshenko_coefficient(self) -> float:
        """Coeficiente de corte o Timoshenko de la sección."""
        return 6 * (1 + self.material.poisson_ratio) / (7 + 6 * self.material.poisson_ratio)


class GenericSection(Section):
    """Clase para secciones genéricas con propiedades básicas."""

    def __init__(
        self,
        name: str,
        material: "Material",
        area: float,
        moment_of_inertia: float,
        timoshenko_coefficient: float
    ):
        """
        Inicializa una sección genérica.

        Args:
            name (str): Nombre de la sección.
            material (Material): Material asociado a la sección.
            area (float): Área de la sección.
            moment_of_inertia (float): Momento de inercia de la sección.
            timoshenko_coefficient (float): Coeficiente de corte de la sección.

        Raises:
            ValueError: Si el área, el momento de inercia o el coeficiente de corte son inválidos.
        """
        self._validate_properties(area, moment_of_inertia, timoshenko_coefficient)
        super().__init__(name, material)
        self._area = area
        self._moment_of_inertia = moment_of_inertia
        self._timoshenko_coefficient = timoshenko_coefficient

    @staticmethod
    def _validate_properties(area: float, moment_of_inertia: float, timoshenko_coefficient: float):
        """Valida que las propiedades sean positivas."""
        if area <= 0 or moment_of_inertia <= 0 or timoshenko_coefficient <= 0:
            raise ValueError("El área, el momento de inercia y el coeficiente de corte deben ser mayores que cero.")

    @property
    def area(self) -> float:
        return self._area

    @property
    def moment_of_inertia(self) -> float:
        return self._moment_of_inertia

    @property
    def timoshenko_coefficient(self) -> float:
        return self._timoshenko_coefficient
