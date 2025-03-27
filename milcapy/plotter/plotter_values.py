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

class PlotterValues:
    """
    Clase que prepara los valores calculados para ser representados gráficamente.
    Mantiene una referencia a datos estáticos compartidos entre todas las instancias.
    """

    _static_data: dict = None

    @classmethod
    def initialize_static_data(cls, system: 'SystemMilcaModel') -> None:
        """
        Inicializa los datos estáticos de la estructura una sola vez.

        los datos son de la forma:
        {
            'nodes': {node_id: (x, y)},
            'members': {member_id: ((x1, y1), (x2, y2))},
            'restraints': {node_id: (bool, bool, bool)}
        }
        """
        if cls._static_data is None:
            # Coordenadas de nodos
            nodes = {node.id: (node.vertex.x, node.vertex.y)
                    for node in system.nodes.values()}

            # Coordenadas de elementos
            members = {}
            for member in system.members.values():
                node_i = member.node_i
                node_j = member.node_j
                members[member.id] = (
                    (node_i.vertex.x, node_i.vertex.y),
                    (node_j.vertex.x, node_j.vertex.y)
                )

            # Restricciones de nodos
            restraints = {}
            for node in system.nodes.values():
                if node.restraints != (False, False, False):
                    restraints[node.id] = node.restraints

            cls._static_data = {
                'nodes': nodes,
                'members': members,
                'restraints': restraints
            }

    def __init__(
        self,
        model: 'SystemMilcaModel',
        current_load_pattern: str,
    ) -> None:
        """
        Inicializa los valores para un load pattern específico.

        Args:
            system: Sistema estructural analizado
            current_load_pattern: Nombre del load pattern
            results: Resultados del análisis para este load pattern
        """
        # Inicializar datos estáticos si aún no se ha hecho
        self.initialize_static_data(model)

        self.model = model
        self.current_load_pattern = current_load_pattern
        self.results = model.results[current_load_pattern]
        self.load_pattern = model.load_patterns.get(current_load_pattern, None)

        if not self.load_pattern:
            raise ValueError(
                f"Load pattern con nombre '{current_load_pattern}' no encontrado")

        # Procesamiento de datos dinámicos específicos del load pattern
        self._process_load_data()

    def _process_load_data(self) -> None:
        """Procesa los datos de cargas para el load pattern actual."""
        # Cargas distribuidas
        self.distributed_loads = {}
        for member_id, loads in self.load_pattern.distributed_loads.items():
            self.distributed_loads[member_id] = loads.to_dict()

        # Cargas puntuales
        self.point_loads = {}
        for node_id, loads in self.load_pattern.point_loads.items():
            self.point_loads[node_id] = loads.to_dict()


    # Propiedades para acceder a datos estáticos
    @property
    def nodes(self) -> Dict[int, Tuple[float, float]]:
        """Devuelve las coordenadas de los nodos."""
        return self._static_data['nodes']

    @property
    def members(self) -> Dict[int, List[Tuple[float, float]]]:
        """Devuelve las coordenadas de los miembros."""
        return self._static_data['members']

    @property
    def restraints(self) -> Dict[int, Tuple[bool, bool, bool]]:
        """Devuelve las restricciones de los nodos."""
        return self._static_data['restraints']

    def rigid_deformed(self, member_id: int, escale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        member = self.model.members[member_id]
        MT = member.transformation_matrix()
        arraydisp = self.results.members[member_id]['displacements']
        arraydisp = np.dot(MT.T, arraydisp)
        x_val = np.array([
            member.node_i.vertex.x + arraydisp[0] * escale,
            member.node_j.vertex.x + arraydisp[3] * escale
        ])
        y_val = np.array([
            member.node_i.vertex.y + arraydisp[1] * escale,
            member.node_j.vertex.y + arraydisp[4] * escale
        ])
        return x_val, y_val


    # def get_deformed_shape_global(self, element_id: int, factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    #     if element_id not in self.deformed_shapes:
    #         raise ValueError(
    #             f"Deformed shape for element with ID {element_id} not found")

    #     # Obtener la deformada en coordenadas globales
    #     deformed_x, deformed_y = self.deformed_shapes[element_id]

    #     # Aplicar factor de escala
    #     if factor != 1.0:
    #         original_x, original_y = self.get_member_global_coordinates(
    #             element_id)
    #         deformed_x = original_x + (deformed_x - original_x) * factor
    #         deformed_y = original_y + (deformed_y - original_y) * factor

    #     return deformed_x, deformed_y


"""
VALORES EN PlotterValues

*** ESTRUCTURA ***
1. Nodos                    : {id_node: (x, y)}
2. Miembros                 : {id_member: [(x1, y1), (x2, y2)]}
3. Restricciones            : {id_node: (restricciones)}
4. Cargas_distribuidas      : {id_member: {q_i, q_j, p_i, p_j, m_i, m_j}}
5. Cargas_puntuales         : {id_node: {fx, fy, mz}}

*** RESULTADOS DE ANALISIS MATRICIAL***
1. Desplazamientos nodales  : {id_node: (ux, vy, wz)}
2. Reacciones               : {id_node: (rx, ry, rz)}
3. Fuerzas internas         : {id_member: {axial, shear, moment}}

*** RESULTADOS DE POST-PROCESSING***
1. Fuerzas Axiales          : {id_member: np.ndarray}
2. Fuerzas Cortantes        : {id_member: np.ndarray}
3. Momentos Flectores       : {id_member: np.ndarray}
4. Giros                    : {id_member: np.ndarray}
5. Deflexiones              : {id_member: np.ndarray}
6. Deformada                : {id_member: np.ndarray}
7. Deformada Rígida         : {id_member: np.ndarray}
"""

# ? IMPLEMTAR FLECHAS CON FI
# ? CALCULAR LA DEFORMADA EN SISTEMA LOCAL
# ? TRANSFORMAR A GLOBALES



