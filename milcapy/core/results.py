from typing import Dict, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from milcapy.core.system import SystemMilcaModel


class Results:
    """Clase que extrae, organiza y almacena los resultados de una simulación."""
    
    def __init__(self, system: 'SystemMilcaModel'):
        """
        Inicializa los resultados para un sistema estructural.
        
        Args:
            system: Sistema estructural analizado
        """
        self.system = system
        self.reactions = self.system.reactions
        self.displacements = self.system.displacements
        
        # Siempre que se inicializa se calculan los resultados
        self.all_to_nodes()
        self.all_to_elements()

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
    
    def all_to_nodes(self) -> None:
        """Agrega los resultados a los objetos de nodos."""
        for node in self.system.node_map.values():
            node.displacement = self.global_displacements_nodes[node.id]
            node.reaction = self.reactions[node.dof-1]
    
    def all_to_elements(self) -> None:
        """Agrega los resultados a los objetos de elementos."""
        for element in self.system.element_map.values():
            element.displacement = self.local_displacements_elements[element.id]
            element.internal_forces = self.local_internal_forces_elements[element.id]