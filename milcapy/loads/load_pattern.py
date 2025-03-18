from typing import Dict, Optional, TYPE_CHECKING
import warnings
import numpy as np

from milcapy.loads import PointLoad, DistributedLoad
from milcapy.utils import LoadPatternType, CoordinateSystemType, State, DirectionType, LoadType

if TYPE_CHECKING:
    from milcapy.elements.system import SystemMilcaModel
    from milcapy.elements.analysis import AnalysisOptions


def loads_to_global_system(load: PointLoad, angle: float) -> PointLoad:
    """
    Transforma una carga puntual del sistema local al sistema global.

    Args:
        load: Carga puntual a transformar.
        angle: Ángulo de rotación en radianes.

    Returns:
        PointLoad: Carga transformada al sistema global.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return PointLoad(
        fx=load.fx * cos_a - load.fy * sin_a,
        fy=load.fx * sin_a + load.fy * cos_a,
        mz=load.mz
    )


def distributed_load_to_local_system(
    system: "SystemMilcaModel",
    load_start: float,
    load_end: float,
    csys: CoordinateSystemType,
    direction: DirectionType,
    load_type: LoadType,
    element_id: int
) -> DistributedLoad:
    """
    Transforma una carga distribuida del sistema global al sistema local del elemento.

    Args:
        system: Modelo estructural que contiene el elemento.
        load_start: Valor inicial de la carga distribuida.
        load_end: Valor final de la carga distribuida.
        csys: Sistema de coordenadas de la carga.
        direction: Dirección de aplicación de la carga.
        load_type: Tipo de carga (fuerza o momento).
        element_id: Identificador del elemento.

    Returns:
        DistributedLoad: Carga transformada al sistema local del elemento.

    Raises:
        ValueError: Si la dirección especificada no es válida.
    """
    if csys == CoordinateSystemType.LOCAL:
        if load_type == LoadType.FORCE:
            if direction == DirectionType.LOCAL_1:
                return DistributedLoad(
                    q_i=0, q_j=0,
                    p_i=load_start, p_j=load_end,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.LOCAL_2:
                return DistributedLoad(
                    q_i=load_start, q_j=load_end,
                    p_i=0, p_j=0,
                    m_i=0, m_j=0
                )
            else:
                raise ValueError(f"Dirección de carga no válida: {direction}")
        elif load_type == LoadType.MOMENT:
            if direction == DirectionType.LOCAL_3:
                return DistributedLoad(
                    q_i=0, q_j=0,
                    p_i=0, p_j=0,
                    m_i=load_start, m_j=load_end
                )
            else:
                raise ValueError(f"Dirección de carga no válida para momento: {direction}")
    
    elif csys == CoordinateSystemType.GLOBAL:
        angle = system.element_map[element_id].angle_x
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        li, lj = load_start, load_end
        
        if load_type == LoadType.FORCE:
            if direction == DirectionType.X:
                return DistributedLoad(
                    q_i=-li * sin_a, q_j=-lj * sin_a,
                    p_i=li * cos_a, p_j=lj * cos_a,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.Y:
                return DistributedLoad(
                    q_i=li * cos_a, q_j=lj * cos_a,
                    p_i=li * sin_a, p_j=lj * sin_a,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.GRAVITY:
                return DistributedLoad(
                    q_i=-li * cos_a, q_j=-lj * cos_a,
                    p_i=-li * sin_a, p_j=-lj * sin_a,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.X_PROJ:
                return DistributedLoad(
                    q_i=-li * sin_a * sin_a, q_j=-lj * sin_a * sin_a,
                    p_i=li * cos_a * sin_a, p_j=lj * cos_a * sin_a,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.Y_PROJ:
                return DistributedLoad(
                    q_i=li * cos_a * cos_a, q_j=lj * cos_a * cos_a,
                    p_i=li * sin_a * cos_a, p_j=lj * sin_a * cos_a,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.GRAVITY_PROJ:
                return DistributedLoad(
                    q_i=-li * cos_a * cos_a, q_j=-lj * cos_a * cos_a,
                    p_i=-li * sin_a * cos_a, p_j=-lj * sin_a * cos_a,
                    m_i=0, m_j=0
                )
            else:
                raise ValueError(f"Dirección de carga no válida: {direction}")
        
        elif load_type == LoadType.MOMENT and direction == DirectionType.MOMENT:
            return DistributedLoad(
                q_i=0, q_j=0,
                p_i=0, p_j=0,
                m_i=li, m_j=lj
            )
        else:
            raise ValueError(f"Combinación no válida de tipo de carga y dirección: {load_type}, {direction}")


class LoadPattern:
    """
    Representa un patrón de carga en un modelo estructural.
    
    Esta clase gestiona la aplicación y transformación de cargas puntuales y distribuidas
    en un sistema estructural, permitiendo su manipulación en diferentes sistemas de coordenadas.
    """

    def __init__(
        self,
        name: str,
        pattern_type: LoadPatternType = LoadPatternType.DEAD,
        self_weight_multiplier: float = 0.0,
        auto_load_pattern: bool = False,
        create_load_case: bool = False,
        state: State = State.ACTIVE,
        system: Optional["SystemMilcaModel"] = None,
    ) -> None:
        """
        Inicializa un nuevo patrón de carga con los parámetros especificados.

        Args:
            name: Nombre identificativo del patrón de carga.
            pattern_type: Tipo de patrón de carga a aplicar.
            self_weight_multiplier: Factor multiplicador para el peso propio.
            auto_load_pattern: Si True, el patrón se genera automáticamente.
            create_load_case: Si True, se crea un caso de carga asociado.
            state: Estado inicial del patrón de carga.
            system: Sistema estructural al que pertenece el patrón de carga.

        Raises:
            ValueError: Si el nombre está vacío o el multiplicador es negativo.
        """
        if not name.strip():
            raise ValueError("El nombre del patrón de carga no puede estar vacío")
        if self_weight_multiplier < 0:
            raise ValueError("El multiplicador de peso propio no puede ser negativo")

        self._system = system

        self.name = name
        self.pattern_type = pattern_type
        self.self_weight_multiplier = float(self_weight_multiplier)
        self.auto_load_pattern = auto_load_pattern
        self.create_load_case = create_load_case
        self.state = state
        self.analyzed = False

        self.point_loads_map: Dict[int, PointLoad] = {}
        self.distributed_loads_map: Dict[int, DistributedLoad] = {}

    def add_point_load(
        self,
        node_id: int,
        forces: PointLoad,
        csys: CoordinateSystemType = CoordinateSystemType.GLOBAL,
        angle_rot: Optional[float] = None,
        replace: bool = False,
    ) -> None:
        """
        Agrega o actualiza una carga puntual en un nodo específico.

        Args:
            node_id: Identificador del nodo objetivo.
            forces: Carga puntual a aplicar.
            csys: Sistema de coordenadas de la carga.
            angle_rot: Ángulo de rotación en radianes (solo para sistema LOCAL).
            replace: Si True, reemplaza cualquier carga existente en el nodo.

        Raises:
            TypeError: Si forces no es del tipo PointLoad.
            ValueError: Si el sistema de coordenadas es inválido o el ángulo es incorrecto.
        """
        if not isinstance(forces, PointLoad):
            raise TypeError("forces debe ser una instancia de PointLoad")

        if csys not in (CoordinateSystemType.GLOBAL, CoordinateSystemType.LOCAL):
            raise ValueError("Sistema de coordenadas debe ser GLOBAL o LOCAL")

        transformed_forces = forces
        if csys == CoordinateSystemType.GLOBAL:
            if angle_rot is not None:
                warnings.warn("El ángulo de rotación se ignora en sistema GLOBAL")
        elif csys == CoordinateSystemType.LOCAL:
            if angle_rot is None:
                raise ValueError("Se debe indicar el ángulo de rotación en sistema LOCAL")
            transformed_forces = loads_to_global_system(forces, angle_rot)

        if replace:
            self.point_loads_map[node_id] = transformed_forces
        else:
            existing_load = self.point_loads_map.get(node_id, PointLoad())
            self.point_loads_map[node_id] = existing_load + transformed_forces

    def add_distributed_load(
        self,
        element_id: int,
        load_start: float,
        load_end: float,
        csys: CoordinateSystemType = CoordinateSystemType.LOCAL,
        direction: DirectionType = DirectionType.LOCAL_2,
        load_type: LoadType = LoadType.FORCE,
        replace: bool = False,
    ) -> None:
        """
        Agrega o actualiza una carga distribuida en un elemento específico.

        Args:
            element_id: Identificador del elemento objetivo.
            load_start: Valor inicial de la carga distribuida.
            load_end: Valor final de la carga distribuida.
            csys: Sistema de coordenadas de la carga.
            direction: Dirección de aplicación de la carga.
            load_type: Tipo de carga (fuerza o momento).
            replace: Si True, reemplaza cualquier carga existente en el elemento.

        Raises:
            ValueError: Si el sistema de coordenadas es inválido.
        """
        if self._system is None:
            raise ValueError("Se requiere un sistema válido para transformar cargas distribuidas")
        
        transformed_load = distributed_load_to_local_system(
            system=self._system,
            load_start=load_start,
            load_end=load_end,
            csys=csys,
            direction=direction,
            load_type=load_type,
            element_id=element_id
        )

        if replace:
            self.distributed_loads_map[element_id] = transformed_load
        else:
            existing_load = self.distributed_loads_map.get(element_id, DistributedLoad())
            self.distributed_loads_map[element_id] = existing_load + transformed_load

    def assign_loads_to_nodes(self, system: "SystemMilcaModel") -> None:
        """
        Asigna las cargas puntuales a los nodos correspondientes del sistema.

        Args:
            system: Modelo estructural que contiene los nodos.
        """
        for node_id, load in self.point_loads_map.items():
            node = system.node_map.get(node_id)
            if node:
                node.add_forces(load)
            else:
                warnings.warn(f"Nodo {node_id} no encontrado en el sistema")

    def assign_loads_to_elements(self, system: "SystemMilcaModel") -> None:
        """
        Asigna las cargas distribuidas a los elementos correspondientes del sistema.

        Args:
            system: Modelo estructural que contiene los elementos.
        """
        for element_id, load in self.distributed_loads_map.items():
            element = system.element_map.get(element_id)
            if element:
                element.add_distributed_load(load)
            else:
                warnings.warn(f"Elemento {element_id} no encontrado en el sistema")