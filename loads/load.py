from typing import Optional, Dict, Union
import numpy as np
from abc import ABC, abstractmethod


class Load(ABC):
    """
    Clase base abstracta que define la interfaz común para diferentes tipos de cargas estructurales.
    
    Esta clase implementa operaciones básicas como comparación, inversión y conversión,
    que son comunes a todos los tipos de cargas.
    
    Attributes:
        id (Optional[int]): Identificador único de la carga. Por defecto es None.
    """

    def __init__(self, id: Optional[int] = None) -> None:
        """
        Inicializa una nueva instancia de carga.

        Args:
            id: Identificador único opcional para la carga.
        """
        self.id = id

    @property
    @abstractmethod
    def components(self) -> np.ndarray:
        """
        Obtiene los componentes de la carga como un array NumPy.
        
        Returns:
            np.ndarray: Array con los componentes de la carga.
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Union[int, float, None]]:
        """
        Convierte la carga a un diccionario serializable.
        
        Returns:
            Dict[str, Union[int, float, None]]: Diccionario con los atributos de la carga.
        """
        pass

    def __eq__(self, other: object) -> bool:
        """
        Compara si dos cargas son aproximadamente iguales.
        
        Args:
            other: Objeto a comparar.
            
        Returns:
            bool: True si las cargas son aproximadamente iguales, False en caso contrario.
        """
        if not isinstance(other, Load):
            return NotImplemented
        return np.allclose(self.components, other.components)

    def __neg__(self) -> 'Load':
        """
        Devuelve una nueva instancia con todos los componentes de carga invertidos.
        
        Returns:
            Load: Nueva instancia con cargas negativas.
        """
        return self.__class__(*(-self.components), id=self.id)

    def __pos__(self) -> 'Load':
        """
        Devuelve la misma instancia sin modificaciones (operador unario +).
        
        Returns:
            Load: La misma instancia sin cambios.
        """
        return self


class PointLoad(Load):
    """
    Representa una carga puntual en un sistema estructural 2D.
    
    Esta clase modela una carga concentrada con:
    - Fuerza de corte perpendicular al eje de la viga (fx)
    - Fuerza axial a lo largo del eje x de la viga (fy)
    - Momento que genera rotación alrededor del eje z (mz)
    
    Attributes:
        fx (float): Fuerza de corte perpendicular al eje de la viga.
        fy (float): Fuerza axial a lo largo del eje x de la viga.
        mz (float): Momento alrededor del eje z.
        id (Optional[int]): Identificador único de la carga.
    """
    
    __slots__ = ('fx', 'fy', 'mz', 'id')
    ZERO = np.zeros(3, dtype=np.float64)

    def __init__(self, fx: float = 0.0, fy: float = 0.0, mz: float = 0.0, 
                 id: Optional[int] = None) -> None:
        """
        Inicializa una nueva carga puntual.
        
        Args:
            fx: Fuerza de corte perpendicular al eje de la viga.
            fy: Fuerza axial a lo largo del eje x de la viga.
            mz: Momento que genera rotación alrededor del eje z.
            id: Identificador único opcional.
        """
        super().__init__(id)
        self.fx = float(fx)
        self.fy = float(fy)
        self.mz = float(mz)

    @property
    def components(self) -> np.ndarray:
        """
        Obtiene los componentes de la carga como un array NumPy.
        
        Returns:
            np.ndarray: Array con [fx, fy, mz].
        """
        return np.array([self.fx, self.fy, self.mz], dtype=np.float64)

    def to_dict(self) -> Dict[str, Union[int, float, None]]:
        """
        Convierte la carga puntual a un diccionario serializable.
        
        Returns:
            Dict[str, Union[int, float, None]]: Diccionario con los componentes de la carga.
        """
        return {"fx": self.fx, "fy": self.fy, "mz": self.mz, "id": self.id}

    def __add__(self, other: "PointLoad") -> "PointLoad":
        """
        Suma dos cargas puntuales.
        
        Args:
            other: Otra carga puntual a sumar.
            
        Returns:
            PointLoad: Nueva carga puntual resultante de la suma.
            
        Raises:
            TypeError: Si other no es una instancia de PointLoad.
        """
        if not isinstance(other, PointLoad):
            return NotImplemented
        return PointLoad(
            self.fx + other.fx,
            self.fy + other.fy,
            self.mz + other.mz,
            self.id or other.id
        )

    def __sub__(self, other: "PointLoad") -> "PointLoad":
        """
        Resta dos cargas puntuales.
        
        Args:
            other: Carga puntual a restar.
            
        Returns:
            PointLoad: Nueva carga puntual resultante de la resta.
            
        Raises:
            TypeError: Si other no es una instancia de PointLoad.
        """
        if not isinstance(other, PointLoad):
            return NotImplemented
        return PointLoad(
            self.fx - other.fx,
            self.fy - other.fy,
            self.mz - other.mz,
            self.id or other.id
        )

    def __mul__(self, scalar: Union[float, int]) -> "PointLoad":
        """
        Multiplica la carga por un escalar.
        
        Args:
            scalar: Factor de escala.
            
        Returns:
            PointLoad: Nueva carga puntual escalada.
            
        Raises:
            TypeError: Si scalar no es un número.
        """
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return PointLoad(
            self.fx * scalar,
            self.fy * scalar,
            self.mz * scalar,
            self.id
        )

    __rmul__ = __mul__

    def __truediv__(self, scalar: Union[float, int]) -> "PointLoad":
        """
        Divide la carga por un escalar.
        
        Args:
            scalar: Divisor.
            
        Returns:
            PointLoad: Nueva carga puntual dividida.
            
        Raises:
            TypeError: Si scalar no es un número.
            ZeroDivisionError: Si scalar es cero.
        """
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("No se puede dividir por cero.")
        return PointLoad(
            self.fx / scalar,
            self.fy / scalar,
            self.mz / scalar,
            self.id
        )


class DistributedLoad(Load):
    """
    Representa una carga distribuida en un elemento estructural 2D.
    
    Esta clase modela cargas distribuidas con valores iniciales (i) y finales (j):
    - q: Carga distribuida perpendicular al eje de la viga (cortante)
    - p: Carga distribuida axial a lo largo del eje x de la viga
    - m: Momento distribuido que genera rotación alrededor del eje z
    
    Attributes:
        q_i (float): Carga de corte inicial, perpendicular al eje de la viga, eje y.
        q_j (float): Carga de corte final, perpendicular al eje de la viga, eje y.
        p_i (float): Carga axial inicial, a lo largo del eje x de la viga.
        p_j (float): Carga axial final, a lo largo del eje x de la viga.
        m_i (float): Momento inicial alrededor del eje z.
        m_j (float): Momento final alrededor del eje z.
        id (Optional[int]): Identificador único de la carga.
    """
    
    __slots__ = ('q_i', 'q_j', 'p_i', 'p_j', 'm_i', 'm_j', 'id')
    ZERO = np.zeros(6, dtype=np.float64)

    def __init__(self, q_i: float = 0.0, q_j: float = 0.0,
                 p_i: float = 0.0, p_j: float = 0.0,
                 m_i: float = 0.0, m_j: float = 0.0,
                 id: Optional[int] = None) -> None:
        """
        Inicializa una nueva carga distribuida.
        
        Args:
            q_i: Carga de corte inicial, perpendicular al eje de la viga.
            q_j: Carga de corte final, perpendicular al eje de la viga.
            p_i: Carga axial inicial, a lo largo del eje x de la viga.
            p_j: Carga axial final, a lo largo del eje x de la viga.
            m_i: Momento inicial alrededor del eje z.
            m_j: Momento final alrededor del eje z.
            id: Identificador único opcional.
            
        Raises:
            TypeError: Si alguno de los valores no es un número real.
        """
        super().__init__(id)
        
        # Validación de tipos y conversión
        load_params = {'q_i': q_i, 'q_j': q_j, 'p_i': p_i, 
                    'p_j': p_j, 'm_i': m_i, 'm_j': m_j}
        
        for name, value in load_params.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"El parámetro {name} debe ser un número real")
            setattr(self, name, float(value))

    @property
    def components(self) -> np.ndarray:
        """
        Obtiene los componentes de la carga como un array NumPy.
        
        Returns:
            np.ndarray: Array con [q_i, q_j, p_i, p_j, m_i, m_j].
        """
        return np.array([self.q_i, self.q_j, self.p_i, self.p_j, self.m_i, self.m_j], 
                       dtype=np.float64)

    def to_dict(self) -> Dict[str, Union[int, float, None]]:
        """
        Convierte la carga distribuida a un diccionario serializable.
        
        Returns:
            Dict[str, Union[int, float, None]]: Diccionario con los componentes de la carga.
        """
        return {
            "q_i": self.q_i, "q_j": self.q_j,
            "p_i": self.p_i, "p_j": self.p_j,
            "m_i": self.m_i, "m_j": self.m_j,
            "id": self.id
        }

    def __add__(self, other: "DistributedLoad") -> "DistributedLoad":
        """
        Suma dos cargas distribuidas.
        
        Args:
            other: Otra carga distribuida a sumar.
            
        Returns:
            DistributedLoad: Nueva carga distribuida resultante de la suma.
            
        Raises:
            TypeError: Si other no es una instancia de DistributedLoad.
        """
        if not isinstance(other, DistributedLoad):
            return NotImplemented
        return DistributedLoad(
            self.q_i + other.q_i,
            self.q_j + other.q_j,
            self.p_i + other.p_i,
            self.p_j + other.p_j,
            self.m_i + other.m_i,
            self.m_j + other.m_j,
            self.id or other.id
        )

    def __sub__(self, other: "DistributedLoad") -> "DistributedLoad":
        """
        Resta dos cargas distribuidas.
        
        Args:
            other: Carga distribuida a restar.
            
        Returns:
            DistributedLoad: Nueva carga distribuida resultante de la resta.
            
        Raises:
            TypeError: Si other no es una instancia de DistributedLoad.
        """
        if not isinstance(other, DistributedLoad):
            return NotImplemented
        return DistributedLoad(
            self.q_i - other.q_i,
            self.q_j - other.q_j,
            self.p_i - other.p_i,
            self.p_j - other.p_j,
            self.m_i - other.m_i,
            self.m_j - other.m_j,
            self.id or other.id
        )

    def __mul__(self, scalar: Union[float, int]) -> "DistributedLoad":
        """
        Multiplica la carga por un escalar.
        
        Args:
            scalar: Factor de escala.
            
        Returns:
            DistributedLoad: Nueva carga distribuida escalada.
            
        Raises:
            TypeError: Si scalar no es un número.
        """
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return DistributedLoad(
            self.q_i * scalar,
            self.q_j * scalar,
            self.p_i * scalar,
            self.p_j * scalar,
            self.m_i * scalar,
            self.m_j * scalar,
            self.id
        )

    __rmul__ = __mul__

    def __truediv__(self, scalar: Union[float, int]) -> "DistributedLoad":
        """
        Divide la carga por un escalar.
        
        Args:
            scalar: Divisor.
            
        Returns:
            DistributedLoad: Nueva carga distribuida dividida.
            
        Raises:
            TypeError: Si scalar no es un número.
            ZeroDivisionError: Si scalar es cero.
        """
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("No se puede dividir por cero.")
        return DistributedLoad(
            self.q_i / scalar,
            self.q_j / scalar,
            self.p_i / scalar,
            self.p_j / scalar,
            self.m_i / scalar,
            self.m_j / scalar,
            self.id
        )

