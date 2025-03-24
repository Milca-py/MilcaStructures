from abc import ABC, abstractmethod
from milcapy.material.material import Material
from math import pi
from milcapy.utils.types import ShearCoefficientMethodType

class Section(ABC):
    """
    Clase base para representar una sección estructural.
    """
    def __init__(
        self,
        name: str,
        material: Material,
        shear_method: ShearCoefficientMethodType
    ):
        """
        Inicializa una sección estructural.

        Args:
            name (str): Nombre de la sección.
            material (Material): Material asociado a la sección.
            shear_method (ShearCoefficientMethodType): Método de cálculo del coeficiente de corte.
        """
        self.name = name
        self.material = material
        self.shear_method = shear_method

    def v(self) -> float:
        """Coeficiente de Poisson."""
        return self.material.v

    def E(self) -> float:
        """Módulo de elasticidad."""
        return self.material.E

    def g(self) -> float:
        """Peso específico."""
        return self.material.g

    def G(self) -> float:
        """Módulo de corte."""
        return self.material.G

    @abstractmethod
    def A(self) -> float:
        """Área de la sección transversal."""
        pass

    @abstractmethod
    def I(self) -> float:
        """Momento de inercia de la sección."""
        pass

    @abstractmethod
    def k(self) -> float:
        """Coeficiente de corte o Timoshenko de la sección."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: name={self.name}, material=({self.material.name}), A={self.A():.3f}, I={self.I():.3f}, k={self.k():.3f}"


class RectangularSection(Section):
    """
    Clase para representar una sección rectangular.
    """
    def __init__(
        self,
        name: str,
        material: Material,
        base: float,
        height: float,
        shear_method: ShearCoefficientMethodType
    ):
        """
        Inicializa una sección rectangular.

        Args:
            name (str): Nombre de la sección.
            material (Material): Material asociado a la sección.
            base (float): Base de la sección (en unidades de longitud).
            height (float): Altura de la sección (en unidades de longitud).
            shear_method (ShearCoefficientMethodType): Método de cálculo del coeficiente de corte.

        Raises:
            ValueError: Si la base o la altura son menores o iguales a cero.
        """
        if base <= 0 or height <= 0:
            raise ValueError("La base y la altura deben ser mayores que cero.")
        super().__init__(name, material, shear_method)
        self.base = base
        self.height = height
        self.shear_method = shear_method

    def A(self) -> float:
        """Área de la sección transversal."""
        return self.base * self.height

    def I(self) -> float:
        """Momento de inercia de la sección."""
        return (self.base * self.height ** 3) / 12

    def k(self) -> float:
        """
        Coeficiente de corte de Timoshenko de la sección.

        Returns:
            float: Coeficiente de corte

        Raises:
            ValueError: Si el método no es válido
        """
        if self.shear_method == ShearCoefficientMethodType.TIMOSHENKO:
            return 5/6
        elif self.shear_method == ShearCoefficientMethodType.COWPER:
            return 10 * (1 + self.v()) / (12 + 11 * self.v())
        else:
            raise ValueError(f"Método de coeficiente no válido: {self.shear_method}. Opciones válidas: 'TIMOSHENKO', 'COWPER'")


class CircularSection(Section):
    """
    Clase para secciones circulares.
    """
    def __init__(
        self,
        name: str,
        material: Material,
        radius: float,
        shear_method: ShearCoefficientMethodType
    ):
        """
        Inicializa una sección circular.

        Args:
            name (str): Nombre de la sección.
            material (Material): Material asociado a la sección.
            radius (float): Radio de la sección.
            shear_method (ShearCoefficientMethodType): Método de cálculo del coeficiente de corte.
        Raises:
            ValueError: Si el radio es menor o igual a cero.
        """
        if radius <= 0:
            raise ValueError("El radio debe ser positivo.")
        super().__init__(name, material, shear_method)
        self.radius = radius
        self.shear_method = shear_method

    def A(self) -> float:
        """Área de la sección transversal."""
        return pi * self.radius ** 2

    def I(self) -> float:
        """Momento de inercia de la sección."""
        return (pi * self.radius ** 4) / 4

    def k(self) -> float:
        """
        Coeficiente de corte o Timoshenko de la sección.

        Returns:
            float: Coeficiente de corte

        Raises:
            ValueError: Si el método no es válido
        """
        if self.shear_method == ShearCoefficientMethodType.TIMOSHENKO:
            return 6/7
        elif self.shear_method == ShearCoefficientMethodType.COWPER:
            return 10 * (1 + self.v()) / (12 + 11 * self.v())
        else:
            raise ValueError(f"Método de coeficiente no válido: {self.shear_method}. Opciones válidas: 'TIMOSHENKO', 'COWPER'")


class GenericSection(Section):
    """Clase para secciones genéricas con propiedades básicas."""

    def __init__(
        self,
        name: str,
        material: Material,
        area: float,
        inertia: float,
        k_factor: float,
        shear_method: ShearCoefficientMethodType
    ):
        """
        Inicializa una sección genérica.

        Args:
            name (str): Nombre de la sección.
            material (Material): Material asociado a la sección.
            area (float): Área de la sección.
            inertia (float): Momento de inercia de la sección.
            k_factor (float): Coeficiente de corte de la sección.
            shear_method (ShearCoefficientMethodType): Método de cálculo del coeficiente de corte.

        Raises:
            ValueError: Si el área, el momento de inercia o el coeficiente de corte son inválidos.
        """
        if area <= 0 or inertia <= 0 or k_factor <= 0:
            raise ValueError("El área, el momento de inercia y el coeficiente de corte deben ser mayores que cero.")
        super().__init__(name, material, shear_method)
        self._area = area
        self._inertia = inertia
        self._k_factor = k_factor
        self.shear_method = shear_method

    def A(self) -> float:
        """Área de la sección transversal."""
        return self._area

    def I(self) -> float:
        """Momento de inercia de la sección."""
        return self._inertia

    def k(self) -> float:
        """
        Coeficiente de corte de Timoshenko de la sección.

        Returns:
            float: Coeficiente de corte
        """
        return self._k_factor

