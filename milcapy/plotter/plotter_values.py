from typing import Dict, Tuple, List, Optional, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from milcapy.core.results import Results


class InternalForceType(Enum):
    """Enumeración de tipos de Fuerzas Internas."""
    AXIAL_FORCE = "axial_force"
    SHEAR_FORCE = "shear_force"
    BENDING_MOMENT = "bending_moment"
    SLOPE = "slope"
    DEFLECTION = "deflection"
    DEFORMED = "deformed"
    RIGID_DEFORMED = "rigid_deformed"


@dataclass
class StaticStructureData:
    """
    Datos estáticos de la estructura que no cambian entre load patterns.
    Se inicializan una vez y se comparten entre todas las instancias de PlotterValues.
    """
    # {id_node: (x, y)}
    nodes: Dict[int, Tuple[float, float]]
    # {id_element: array[array[xi, yi], array[xj, yj]]}
    elements: Dict[int, np.ndarray[np.float64]]
    # {id_node: (dx, dy, rz)}   
    restraints: Dict[int, Tuple[bool, bool, bool]]


class PlotterValues:
    """
    Clase que prepara los valores calculados para ser representados gráficamente.
    Mantiene una referencia a datos estáticos compartidos entre todas las instancias.
    """

    _static_data: Optional[StaticStructureData] = None

    @classmethod
    def initialize_static_data(cls, system: 'SystemMilcaModel') -> None:
        """
        Inicializa los datos estáticos de la estructura una sola vez.

        Args:
            system: Sistema estructural analizado
        """
        if cls._static_data is None:
            # Coordenadas de nodos
            nodes = {node.id: (node.vertex.x, node.vertex.y)
                    for node in system.node_map.values()}

            # Coordenadas de elementos
            elements = {}
            for element in system.element_map.values():
                node_i = element.node_i
                node_j = element.node_j
                elements[element.id] = (
                    (node_i.vertex.x, node_i.vertex.y),
                    (node_j.vertex.x, node_j.vertex.y)
                )

            # Restricciones de nodos
            restraints = {}
            for node in system.node_map.values():
                if node.restraints != (False, False, False):
                    restraints[node.id] = node.restraints

            cls._static_data = StaticStructureData(
                nodes=nodes,
                elements=elements,
                restraints=restraints
            )

    def __init__(
        self,
        system: 'SystemMilcaModel',
        load_pattern_name: str,
        results: 'Results'
    ) -> None:
        """
        Inicializa los valores para un load pattern específico.

        Args:
            system: Sistema estructural analizado
            load_pattern_name: Nombre del load pattern
            results: Resultados del análisis para este load pattern
        """
        # Inicializar datos estáticos si aún no se ha hecho
        self.initialize_static_data(system)

        self.system = system
        self.load_pattern_name = load_pattern_name
        self.results = results
        self.load_pattern = system.load_pattern_map.get(load_pattern_name, None)

        if not self.load_pattern:
            raise ValueError(
                f"Load pattern con nombre '{load_pattern_name}' no encontrado")

        # Procesamiento de datos dinámicos específicos del load pattern
        self._process_load_data()
        self._process_analysis_results()
        self._process_post_processing_results()

    def _process_load_data(self) -> None:
        """Procesa los datos de cargas para el load pattern actual."""
        # Cargas distribuidas
        self.distributed_loads = {}
        for element_id, loads in self.load_pattern.distributed_loads_map.items():
            self.distributed_loads[element_id] = loads.to_dict()

        # Cargas puntuales
        self.point_loads = {}
        for node_id, loads in self.load_pattern.point_loads_map.items():
            self.point_loads[node_id] = loads.to_dict()

    def _process_analysis_results(self) -> None:
        """Procesa los resultados del análisis matricial."""
        # Desplazamientos nodales
        self.nodal_displacements = {}
        for node_id, node in self.system.node_map.items():
            # Extraer desplazamientos del vector global
            dofs = node.dof - 1  # Convertir a índice base 0
            ux = self.results.displacements[dofs[0]] if dofs[0] >= 0 else 0.0
            uy = self.results.displacements[dofs[1]] if dofs[1] >= 0 else 0.0
            rz = self.results.displacements[dofs[2]] if dofs[2] >= 0 else 0.0
            self.nodal_displacements[node_id] = (ux, uy, rz)

        # Reacciones nodales
        self.reactions = {}
        for node_id, node in self.system.node_map.items():
            # Extraer reacciones del vector global
            dofs = node.dof - 1  # Convertir a índice base 0
            rx = self.results.reactions[dofs[0]] if dofs[0] >= 0 else 0.0
            ry = self.results.reactions[dofs[1]] if dofs[1] >= 0 else 0.0
            rz = self.results.reactions[dofs[2]] if dofs[2] >= 0 else 0.0
            self.reactions[node_id] = (rx, ry, rz)

        # Fuerzas internas
        self.internal_forces = {}
        for element_id, element in self.system.element_map.items():
            forces = self.results.local_internal_forces_elements[element_id]

            # Organizando las fuerzas internas por tipo
            # Asumiendo el orden: [Ni, Vi, Mi, Nj, Vj, Mj]
            self.internal_forces[element_id] = {
                'axial': (forces[0], forces[3]),  # Ni, Nj
                'shear': (forces[1], forces[4]),  # Vi, Vj
                'moment': (forces[2], forces[5])  # Mi, Mj
            }

    def _process_post_processing_results(self) -> None:
        """Procesa los resultados del post-procesamiento."""
        # Fuerzas axiales
        self.axial_forces = self.results.values_axial_force_elements

        # Fuerzas cortantes
        self.shear_forces = self.results.values_shear_force_elements

        # Momentos flectores
        self.bending_moments = self.results.values_bending_moment_elements

        # Giros
        self.slopes = self.results.values_slope_elements

        # Deflexiones
        self.deflections = self.results.values_deflection_elements

        # Deformada
        self.deformed_shapes = self.results.values_deformed_elements

        # Deformada rígida
        self.rigid_deformed_shapes = self.results.values_rigid_deformed_elements

    # Propiedades para acceder a datos estáticos
    @property
    def nodes(self) -> Dict[int, Tuple[float, float]]:
        """Devuelve las coordenadas de los nodos."""
        return self._static_data.nodes

    @property
    def elements(self) -> Dict[int, List[Tuple[float, float]]]:
        """Devuelve las coordenadas de los elementos."""
        return self._static_data.elements

    @property
    def restraints(self) -> Dict[int, Tuple[bool, bool, bool]]:
        """Devuelve las restricciones de los nodos."""
        return self._static_data.restraints

    def get_element_global_coordinates(self, element_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve las coordenadas del elemento en el sistema global.

        Args:
            element_id: ID del elemento

        Returns:
            Tuple[np.ndarray, np.ndarray]: Coordenadas (x, y) del elemento
        """
        if element_id not in self._static_data.elements:
            raise ValueError(f"Element with ID {element_id} not found")

        element_coords = self._static_data.elements[element_id]
        x_coords = np.array([coord[0] for coord in element_coords])
        y_coords = np.array([coord[1] for coord in element_coords])
        return x_coords, y_coords

    def get_deformed_shape_global(self, element_id: int, factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve la forma deformada del elemento en coordenadas globales.

        Args:
            element_id: ID del elemento
            factor: Factor de escala para la visualización

        Returns:
            Tuple[np.ndarray, np.ndarray]: Coordenadas (x, y) de la forma deformada
        """
        if element_id not in self.deformed_shapes:
            raise ValueError(
                f"Deformed shape for element with ID {element_id} not found")

        # Obtener la deformada en coordenadas globales
        deformed_x, deformed_y = self.deformed_shapes[element_id]

        # Aplicar factor de escala
        if factor != 1.0:
            original_x, original_y = self.get_element_global_coordinates(
                element_id)
            deformed_x = original_x + (deformed_x - original_x) * factor
            deformed_y = original_y + (deformed_y - original_y) * factor

        return deformed_x, deformed_y


class PlotterValuesFactory:
    """
    Factory para crear instancias de PlotterValues para cada load pattern.
    Implementa un patrón Singleton para asegurar una única instancia en la aplicación.
    """
    _instance: Optional['PlotterValuesFactory'] = None

    def __new__(cls, *args, **kwargs):
        """Implementación del patrón Singleton."""
        if cls._instance is None:
            cls._instance = super(PlotterValuesFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, system: 'SystemMilcaModel') -> None:
        """
        Inicializa la factory con el sistema estructural.

        Args:
            system: Sistema estructural analizado
        """
        # Inicializar solo una vez (patrón Singleton)
        if self._initialized:
            return

        self.system = system
        self._plotter_values_cache: Dict[str, 'PlotterValues'] = {}
        self._initialized = True

    def get_plotter_values(self, load_pattern_name: str) -> 'PlotterValues':
        """
        Devuelve una instancia de PlotterValues para el load pattern especificado.
        Si ya existe en caché, devuelve la instancia existente.

        Args:
            load_pattern_name: Nombre del load pattern

        Returns:
            PlotterValues: Instancia con los valores para el load pattern
        """
        # Comprobar si ya existe en caché
        if load_pattern_name in self._plotter_values_cache:
            return self._plotter_values_cache[load_pattern_name]

        # Si no existe, crear nueva instancia
        from milcapy.plotter.plotter_values import PlotterValues

        # Verificar que el load pattern existe
        if load_pattern_name not in self.system.load_pattern_map:
            raise ValueError(
                f"Load pattern with name '{load_pattern_name}' not found")

        # Verificar que existen resultados para este load pattern
        if load_pattern_name not in self.system.loadpattern_results:
            raise ValueError(
                f"Results for load pattern with name '{load_pattern_name}' not found")

        # Obtener los resultados para este load pattern
        results = self.system.loadpattern_results[load_pattern_name]

        # Crear nueva instancia de PlotterValues
        plotter_values = PlotterValues(self.system, load_pattern_name, results)

        # Guardar en caché
        self._plotter_values_cache[load_pattern_name] = plotter_values

        return plotter_values

    def clear_cache(self) -> None:
        """Limpia la caché de instancias de PlotterValues."""
        self._plotter_values_cache.clear()

    def invalidate_cache(self, load_pattern_name: str) -> None:
        """
        Invalida la caché para un load pattern específico.

        Args:
            load_pattern_name: Nombre del load pattern
        """
        if load_pattern_name in self._plotter_values_cache:
            del self._plotter_values_cache[load_pattern_name]

























# # =======================================================================================
# class PlotterValues:
#     """Clase que proporciona los valores necesarios para graficar la estructura."""

#     def __init__(self, system: 'SystemMilcaModel'):
#         """
#         Inicializa un objeto PlotterValues.

#         Args:
#             system: Sistema de estructura a graficar.
#         """
#         self.system = system

#     def structure(self) -> Tuple[Dict[int, Tuple[float, float]],
#                                  Dict[int, List[Tuple[float, float]]],
#                                  Dict[int, Dict],
#                                  Dict[int, Dict],
#                                  Dict[int, Tuple[bool, bool, bool]]]:
#         """
#         Devuelve los valores necesarios para graficar la estructura.

#         Returns:
#             Tuple que contiene:
#             - node_values: Diccionario de nodos {id: (x, y)}
#             - element_values: Diccionario de elementos {id: [(x1, y1), (x2, y2)]}
#             - load_elements: Diccionario de cargas distribuidas {id_element: {q_i, q_j, p_i, p_j, m_i, m_j}}
#             - load_nodes: Diccionario de cargas puntuales {id_node: {fx, fy, mz}}
#             - restrained_nodes: Diccionario de nodos restringidos {id: (restricciones)}
#         """
#         # Obtener los valores para graficar los nodos {id: (x, y)}
#         node_values = {node.id: (node.vertex.x, node.vertex.y)
#                        for node in self.system.node_map.values()}

#         # Obtener los valores para graficar los elementos {id: [(x1, y1), (x2, y2)]}
#         element_values = {}
#         for element in self.system.element_map.values():
#             node_i, node_j = element.node_i, element.node_j
#             element_values[element.id] = [
#                 (node_i.vertex.x, node_i.vertex.y),
#                 (node_j.vertex.x, node_j.vertex.y)
#             ]

#         # Obtener los elementos cargados {id: {q_i, q_j, p_i, p_j, m_i, m_j}}
#         load_elements = {}
#         for load_pattern in self.system.load_pattern_map.values():
#             for id_element, load in load_pattern.distributed_loads_map.items():
#                 load_elements[id_element] = load.to_dict()

#         # Obtener nodos cargados {id: {fx, fy, mz}}
#         load_nodes = {}
#         for load_pattern in self.system.load_pattern_map.values():
#             for id_node, load in load_pattern.point_loads_map.items():
#                 load_nodes[id_node] = load.to_dict()

#         # Obtener los nodos restringidos {id: (restricciones)}
#         restrained_nodes = {}
#         for node in self.system.node_map.values():
#             if node.restraints != (False, False, False):
#                 restrained_nodes[node.id] = node.restraints

#         return node_values, element_values, load_elements, load_nodes, restrained_nodes

# # =======================================================================================