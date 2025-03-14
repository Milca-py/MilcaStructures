from typing import TYPE_CHECKING, Literal, Sequence, Union, Tuple
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from milcapy.utils.vertex import Vertex
    from milcapy.elements.analysis import DynamicsAnalysis, StaticAnalysis


# Tipos básicos
NumberLike = Union[float, int, np.number]
VertexLike = Union[Sequence[Union[float, int]], np.ndarray, 'Vertex']
SequenceLike = Union[Sequence[Union[float, int]], np.ndarray]
TypeAnalysis = Union["DynamicsAnalysis", "StaticAnalysis"]

# Definición de ejes y restricciones
AxisNumber = Literal[1, 2, 3]
Restraints = Tuple[bool, bool, bool]


class ElementType(Enum):
    """Tipos de elementos en la estructura."""
    FRAME = 'FRAME'
    TRUSS = 'TRUSS'


class LoadPatternType(Enum):
    """Tipos de patrones de carga."""
    DEAD = 'DEAD'
    LIVE = 'LIVE'
    THERMAL = 'THERMAL'
    OTHER = 'OTHER'


class CoordinateSystemType(Enum):
    """Tipos de sistemas de coordenadas."""
    GLOBAL = 'GLOBAL'
    LOCAL = 'LOCAL'


class DirectionType(Enum):
    """Direcciones en el sistema global y local."""
    X = 'X'
    Y = 'Y'
    X_PROJ = 'X_PROJ'
    Y_PROJ = 'Y_PROJ'
    GRAVITY = 'GRAVITY'
    GRAVITY_PROJ = 'GRAVITY_PROJ'
    MOMENT = 'MOMENT'
    LOCAL_1 = 'LOCAL_1' # AXIAL
    LOCAL_2 = 'LOCAL_2' # SHEAR
    LOCAL_3 = 'LOCAL_3' # MOMENT


class CodeType(Enum):
    """Códigos para materiales y normas de diseño."""
    USER = 'USER'  # Coeficientes personalizados
    E010 = 'E010'  # Madera
    E020 = 'E020'  # Cargas
    E030 = 'E030'  # Diseño Sismorresistente
    E031 = 'E031'  # Aislamiento Sísmico
    E040 = 'E040'  # Vidrio
    E050 = 'E050'  # Suelos y Cimentaciones
    E060 = 'E060'  # Concreto Armado
    E070 = 'E070'  # Albañilería
    E080 = 'E080'  # Diseño y construcción con tierra reforzada
    E090 = 'E090'  # Estructuras Metálicas
    E100 = 'E100'  # Bambú


class State(Enum):
    """Estado de los elementos o componentes."""
    ACTIVE = 'ACTIVE'
    INACTIVE = 'INACTIVE'
    PENDING = 'PENDING'
    ERROR = 'ERROR'


class LoadType(Enum):
    """Tipos de cargas (fuerzas o momentos)."""
    FORCE = 'FORCE'
    MOMENT = 'MOMENT'


class LoadCaseType(Enum):
    """Tipos de casos de carga."""
    STATIC_LINEAR = 'STATIC_LINEAR'


class CaseLoadType(Enum):
    """Tipos de carga para un caso específico."""
    LOAD = 'LOAD'
    ACCELERATION = 'ACCELERATION'


class ComboType(Enum):
    """Tipos de combinaciones de carga."""
    LINEAR = 'LINEAR'
    ENVELOPE = 'ENVELOPE'
    ABSOLUTE_ADDITIVE = 'ABSOLUTE_ADDITIVE'
    SRSS = 'SRSS'
    RANGE_ADDITIVE = 'RANGE_ADDITIVE'



def to_enum(key: str, enum: Enum) -> Enum:
    """Convierte un string a un miembro de un Enum."""
    assert isinstance(key, str), 'La clave debe ser un string.'
    assert issubclass(enum, Enum), 'El segundo argumento debe ser una clase Enum.'
    try:
        return enum(key)
    except ValueError:
        raise ValueError(f'La clave "{key}" no se encuentra en el Enum "{enum.__name__}".')
