from typing import Dict, TYPE_CHECKING, Optional
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from milcapy.elements.element import Element
    from milcapy.core.node import Node



class Results:
    def __init__(
        self, 
        system: 'SystemMilcaModel', 
        ) -> None:

        # parametdos de entrada (objetos)
        self.system = system
        
        self.reactions = self.system.reactions
        self.displacements = self.system.displacements

        # en coordenadas locales
        self.integration_coefficients_elements = {} # {element_id: np.ndarray}
        self.values_axial_force_elements = {}       # {element_id: np.ndarray}
        self.values_shear_force_elements = {}       # {element_id: np.ndarray}
        self.values_bending_moment_elements = {}    # {element_id: np.ndarray}
        self.values_slope_elements = {}             # {element_id: np.ndarray}
        self.values_deflection_elements = {}        # {element_id: np.ndarray}
        
        # en coordenadas globales (x_val, y_val)
        self.values_deformed_elements = {}          # {element_id: Tuple[np.ndarray, np.ndarray]}
        self.values_rigid_deformed_elements = {}    # {element_id: Tuple[np.ndarray, np.ndarray]}

    @property
    def global_displacements_nodes(self) -> Dict[int, np.ndarray]:
        """Desplazamientos globales de los nodos."""
        displacements_nodes = {}
        for node in self.system.node_map.values():
            displacements_nodes[node.id] = self.displacements[node.dof-1]
        return displacements_nodes
    
    @property
    def global_displacements_elements(self) -> Dict[int, np.ndarray]:
        """Desplazamientos globales de los elementos."""
        displacements_elements = {}
        for element in self.system.element_map.values():
            displacements_elements[element.id] = np.concatenate([
                self.displacements[element.node_i.dof-1], 
                self.displacements[element.node_j.dof-1]
            ])
        return displacements_elements
    
    @property
    def local_displacements_elements(self) -> Dict[int, np.ndarray]:
        """Desplazamientos locales de los elementos."""
        global_disps = self.global_displacements_elements
        displacements_elements = {}
        for element in self.system.element_map.values():
            displacements_elements[element.id] = np.dot(
                element.transformation_matrix, 
                global_disps[element.id]
            )
        return displacements_elements
    
    @property
    def local_internal_forces_elements(self) -> Dict[int, np.ndarray]:
        """Fuerzas internas en los elementos en coordenadas locales."""
        local_disps = self.local_displacements_elements
        forces_elements = {}
        for element in self.system.element_map.values():
            forces_elements[element.id] = np.dot(
                element.local_stiffness_matrix, 
                local_disps[element.id]
            ) - element.local_load_vector
        return forces_elements















class Results:
    """Clase que almacena los resultados de un análisis de un Load Pattern."""
    def __init__(self):
        self.nodes: Dict[int, NodeResults] = {}
        self.elements: Dict[int, ElementResults] = {}


class NodeResults:
    """Clase que almacena los resultados de un nodo (coordenadas globales)."""
    def __init__(self, node: "Node") -> None:
        self.displacement: Optional[np.ndarray] = None
        self.reaction: Optional[np.ndarray] = None


class ElementResults:
    """Clase que almacena los resultados de un elemento (coordenadas locales)."""
    def __init__(self, element: "Element") -> None:
        self.displacement: Optional[np.ndarray] = None
        self.internal_forces: Optional[np.ndarray] = None
        
        # Resultados del postprocesamiento (en coordenada local)
        self.integration_coefficients: Optional[np.ndarray] = None
        self.axial_force: Optional[np.ndarray] = None
        self.shear_force: Optional[np.ndarray] = None
        self.bending_moment: Optional[np.ndarray] = None
        self.deflection: Optional[np.ndarray] = None
        self.slope: Optional[np.ndarray] = None
        self.deformed_shape: Optional[np.ndarray] = None


class MoldelResults:
    """Clase que almacena los resultados de un análisis de un modelo."""
    def __init__(self):
        self.stiffness_matrix: Optional[np.ndarray] = None
        self.load_vector: Optional[np.ndarray] = None
        self.displacements: Optional[np.ndarray] = None
        self.reactions: Optional[np.ndarray] = None


# ELIMINAR LOS ATRIBUTOS TRANSITORIOS DE LOS ELEMENTOS Y NODOS
# DISEÑAR CLASE RESULTS
# ANALYSYS POR CASO DE CARGA
# IMPLENETAR FUNCIONES HELPERS
# IMPLEMENTAR ATRIBUTOS DELEGADOS
# USAR NOTIFICACIONES