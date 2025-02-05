from typing import Dict, TYPE_CHECKING
import warnings
import numpy as np

from loads import PointLoad, DistributedLoad
from utils import(
    LoadPatternType,
    CoordinateSystemType,
    State,
)

if TYPE_CHECKING:
    from core.system import SystemMilcaModel


class LoadPattern:
    """Clase que representa un patrón de carga en el modelo estructural."""

    def __init__(
        self,
        name: str,
        load_type: LoadPatternType = LoadPatternType.DEAD,
        self_weight_multiplier: float = 0.0,
        auto_load_pattern: bool = False,
        create_load_case: bool = False,
        state: State = State.ACTIVE,
    ) -> None:
        """
        Inicializa un patrón de carga.

        Args:
            name (str): Nombre del patrón de carga.
            load_type (LoadPatternType, opcional): Tipo de carga (default: DEAD).
            self_weight_multiplier (float, opcional): Multiplicador del peso propio (default: 0.0).
            auto_load_pattern (bool, opcional): Indica si el patrón de carga se genera automáticamente.
            create_load_case (bool, opcional): Indica si se crea un caso de carga asociado.
            state (State, opcional): Estado del patrón de carga (default: ACTIVE).
        """
        self.name: str = name
        self.load_type: LoadPatternType = load_type
        self.self_weight_multiplier: float = float(self_weight_multiplier)
        self.auto_load_pattern: bool = auto_load_pattern
        self.create_load_case: bool = create_load_case
        self.state: State = state

        self.point_loads_map: Dict[int, PointLoad] = {}
        self.distributed_loads_map: Dict[int, DistributedLoad] = {}

    def add_point_load(
        self,
        node_id: int,
        forces: PointLoad,
        CSys: CoordinateSystemType = CoordinateSystemType.GLOBAL,
        angle: float = 0.0,
        replace: bool = False,
    ) -> None:
        """
        Agrega o reemplaza una carga puntual en el nodo especificado.

        Args:
            node_id (int): ID del nodo al que se asigna la carga.
            forces (PointLoad): Carga puntual aplicada.
            CSys (CoordinateSystemType, opcional): Sistema de coordenadas de la carga (default: GLOBAL).
            angle (float, opcional): Ángulo de rotación en radianes (default: 0.0).
            replace (bool, opcional): Si es True, reemplaza la carga existente en el nodo (default: False).

        Raises:
            TypeError: Si `forces` no es una instancia de `PointLoad`.
            ValueError: Si `CSys` no es GLOBAL o LOCAL.
        """
        if not isinstance(forces, PointLoad):
            raise TypeError("La carga debe ser un objeto de tipo PointLoad.")

        if CSys == CoordinateSystemType.LOCAL:
            forces = loads_to_global_system(forces, angle)
        elif CSys != CoordinateSystemType.GLOBAL:
            raise ValueError("El sistema de coordenadas debe ser LOCAL o GLOBAL.")

        if replace:
            self.point_loads_map[node_id] = forces
        else:
            self.point_loads_map[node_id] = self.point_loads_map.get(node_id, PointLoad()) + forces

    def add_distributed_load(
        self,
        element_id: int,
        load: DistributedLoad,
        CSys: CoordinateSystemType = CoordinateSystemType.LOCAL,
        angle: float = 0.0,
        replace: bool = False,
    ) -> None:
        """
        Agrega o reemplaza una carga distribuida en el elemento especificado.

        Args:
            element_id (int): ID del elemento al que se asigna la carga.
            load (DistributedLoad): Carga distribuida aplicada.
            CSys (CoordinateSystemType, opcional): Sistema de coordenadas de la carga (default: GLOBAL).
            angle (float, opcional): Ángulo de rotación en radianes (default: 0.0).
            replace (bool, opcional): Si es True, reemplaza la carga existente en el elemento (default: False).

        Raises:
            TypeError: Si `load` no es una instancia de `DistributedLoad`.
            ValueError: Si `CSys` no es GLOBAL o LOCAL.
        """
        if not isinstance(load, DistributedLoad):
            raise TypeError("La carga debe ser un objeto de tipo DistributedLoad.")

        if CSys == CoordinateSystemType.LOCAL:
            pass
            # load = loads_to_global_system(load, angle)
        elif CSys != CoordinateSystemType.GLOBAL:
            raise ValueError("El sistema de coordenadas debe ser LOCAL o GLOBAL.")

        if replace:
            self.distributed_loads_map[element_id] = load
        else:
            self.distributed_loads_map[element_id] = load #self.distributed_loads_map.get(element_id, DistributedLoad()) + load        

    def assign_loads_to_nodes(self, system: "SystemMilcaModel") -> None:
        """
        Asigna las cargas puntuales a los nodos del modelo.

        Args:
            system (SystemMilcaModel): Modelo estructural.

        Raises:
            Warning: Si algún nodo no existe en el modelo.
        """
        for node_id, load in self.point_loads_map.items():
            node = system.node_map.get(node_id)
            if node:
                node.add_forces(load)
            else:
                warnings.warn(f"El nodo {node_id} no existe en el modelo estructural.", UserWarning)

    def assign_loads_to_elements(self, system: "SystemMilcaModel") -> None:
        """
        Asigna las cargas distribuidas a los elementos del modelo.

        Args:
            system (SystemMilcaModel): Modelo estructural.

        Raises:
            Warning: Si algún elemento no existe en el modelo.
        """
        for element_id, load in self.distributed_loads_map.items():
            element = system.element_map.get(element_id)
            if element:
                element.add_distributed_load(load)
            else:
                warnings.warn(f"El elemento {element_id} no existe en el modelo estructural.", UserWarning)

def loads_to_global_system(load: PointLoad, angle: float) -> PointLoad:
    """
    Convierte una carga puntual de coordenadas locales a globales.

    Args:
        load (PointLoad): Carga puntual.
        angle (float): Ángulo de rotación en radianes.

    Returns:
        PointLoad: Carga puntual en coordenadas globales.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    return PointLoad(
        fx=load.fx * cos_a - load.fy * sin_a,
        fy=load.fx * sin_a + load.fy * cos_a,
        mz=load.mz  # Si el momento necesita transformación adicional, modificar aquí.
    )
