from abc import ABC, abstractmethod
from milcapy.material.material import Material
from math import pi
from milcapy.utils.types import ShearCoefficientMethodType

class Section(ABC):
    """
    Clase base para representar una sección estructural.

    Esta clase proporciona los métodos y propiedades comunes para todas las secciones
    estructurales. Las secciones estructurales son objetos que representan una sección
    transversal de un miembro estructural, como una viga o una columna. Las secciones
    estructurales tienen un nombre, un material, un método de cálculo del coeficiente de
    corte y propiedades geométricas como el área de la sección transversal y el momento
    de inercia.

    Attributes:
        name (str): Nombre de la sección.
        material (Material): Material asociado a la sección.
        shear_method (ShearCoefficientMethodType): Método de cálculo del coeficiente de corte.
        kA (float): Modificador de área. Por defecto es 1.
        kAs (float): Modificador de área para cálculo de la rigidez a cortante. Por defecto es 1.
        kI (float): Modificador de momento de inercia. Por defecto es 1.
        kg (float): Modificador de peso. Por defecto es 1.

    Properties:
        kg (float): Modificador de peso.
        v (float): Coeficiente de Poisson.
        E (float): Módulo de elasticidad.
        g (float): Peso específico.
        G (float): Módulo de corte.

    Abstract methods:
        A (float): Área de la sección transversal.
        I (float): Momento de inercia de la sección.
        k (float): Coeficiente de corte o Timoshenko de la sección.

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
        self.kA = 1
        self.kAs = 1
        self.kI = 1
        self.kg = 1

    @property
    def kg(self) -> float:
        """Modificador de peso."""
        return self.material.kg

    @kg.setter
    def kg(self, value: float) -> None:
        self.material.kg = value

    def v(self) -> float:
        """Coeficiente de Poisson."""
        return self.material.v

    def E(self) -> float:
        """Módulo de elasticidad."""
        return self.material.E

    def g(self) -> float:
        """Peso específico."""
        return self.material.g * self.kg

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
        return f"{self.__class__.__name__}: name={self.name}, material=({self.material.name}), A={self.A():.3f}, I={self.I():.3f}, k={self.k():.3f}, kg={self.kg():.3f}"


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
        return self.base * self.height * self.kA

    def I(self) -> float:
        """Momento de inercia de la sección."""
        return (self.base * self.height ** 3) / 12 * self.kI

    def k(self) -> float:
        """
        Coeficiente de corte de Timoshenko de la sección.

        Returns:
            float: Coeficiente de corte

        Raises:
            ValueError: Si el método no es válido
        """
        if self.shear_method == ShearCoefficientMethodType.TIMOSHENKO:
            return 5/6 * self.kAs
        elif self.shear_method == ShearCoefficientMethodType.COWPER:
            return 10 * (1 + self.v()) / (12 + 11 * self.v()) * self.kAs
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
        return pi * self.radius ** 2 * self.kA

    def I(self) -> float:
        """Momento de inercia de la sección."""
        return (pi * self.radius ** 4) / 4 * self.kI

    def k(self) -> float:
        """
        Coeficiente de corte o Timoshenko de la sección.

        Returns:
            float: Coeficiente de corte

        Raises:
            ValueError: Si el método no es válido
        """
        if self.shear_method == ShearCoefficientMethodType.TIMOSHENKO:
            return 6/7 * self.kAs
        elif self.shear_method == ShearCoefficientMethodType.COWPER:
            return 10 * (1 + self.v()) / (12 + 11 * self.v()) * self.kAs
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
    ):
        """
        Inicializa una sección genérica.

        Args:
            name (str): Nombre de la sección.
            material (Material): Material asociado a la sección.
            area (float): Área de la sección.
            inertia (float): Momento de inercia de la sección.
            k_factor (float): Coeficiente de corte de la sección.

        Raises:
            ValueError: Si el área, el momento de inercia o el coeficiente de corte son inválidos.
        """
        if area <= 0 or inertia <= 0 or k_factor <= 0:
            raise ValueError("El área, el momento de inercia y el coeficiente de corte deben ser mayores que cero.")
        super().__init__(name, material, ShearCoefficientMethodType.NONE)
        self._area = area
        self._inertia = inertia
        self._k_factor = k_factor

    def A(self) -> float:
        """Área de la sección transversal."""
        return self._area * self.kA

    def I(self) -> float:
        """Momento de inercia de la sección."""
        return self._inertia * self.kI

    def k(self) -> float:
        """
        Coeficiente de corte de Timoshenko de la sección.

        Returns:
            float: Coeficiente de corte
        """
        return self._k_factor * self.kAs


class ShellSection:
    """Clase para secciones de cascaras"""
    def __init__(
        self,
        name: str,
        material: Material,
        thickness: float,
    ):
        """
        Inicializa una sección de cascaras.

        Args:
            name (str): Nombre de la sección.
            material (Material): Material asociado a la sección.
            thickness (float): Grosor de la sección.

        Raises:
            ValueError: Si el grosor es inválido.
        """
        if thickness <= 0:
            raise ValueError("El grosor debe ser mayor que cero.")
        self.name = name
        self.material = material
        self.t = thickness

    def E(self):
        return self.material.E
    def v(self):
        return self.material.v
    def G(self):
        return self.material.G
    def g(self):
        return self.material.g
    def alpha(self):
        return self.material.alpha
    def beta(self):
        return self.material.beta
    def gamma(self):
        return self.material.gamma