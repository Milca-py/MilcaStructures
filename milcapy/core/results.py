from typing import List, Dict, Union, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from core.system import SystemMilcaModel


class Results:
    "Clase que extrae, organiza y almacena los resultados de una simulacion."
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system
        self.reactions = self.system.reactions
        self.desplacements = self.system.displacements
        
        
        
        # siepre que se inicializa se calculan los resultados
        self.all_to_nodes()
        self.all_to_elements()

    @property
    def global_desplacements_nodes(self) -> Dict[int, np.ndarray]:
        "Desplazamientos de los nodos."
        desplacements_nodes = {}
        for node in self.system.node_map.values():
            desplacements_nodes[node.id] = self.desplacements[node.dof-1]
        return desplacements_nodes
    
    @property
    def global_desplacements_elements(self) -> Dict[int, np.ndarray]:
        "Desplazamientos de los elementos."
        desplacements_elements = {}
        for element in self.system.element_map.values():
            desplacements_elements[element.id] = np.concatenate([self.desplacements[element.node_i.dof-1], self.desplacements[element.node_j.dof-1]])
        return desplacements_elements
    
    @property
    def local_desplacements_elements(self) -> Dict[int, np.ndarray]:
        "Desplazamientos locales de los elementos."
        desplacements_elements = {}
        for element in self.system.element_map.values():
            desplacements_elements[element.id] = np.dot(element.transformation_matrix, self.global_desplacements_elements[element.id])
        return desplacements_elements
    
    @property
    def local_internal_forces_elements(self) -> Dict[int, np.ndarray]:
        "Fuerzas en los elementos."
        forces_elements = {}
        for element in self.system.element_map.values():
            forces_elements[element.id] = np.dot(element.local_stiffness_matrix, self.local_desplacements_elements[element.id]) - element.local_load_vector
        return forces_elements
    
    def all_to_nodes(self) -> None:
        "agrega a los resutados de los nodos"
        for node in self.system.node_map.values():
            node.desplacement = self.global_desplacements_nodes[node.id]
            node.reaction = self.reactions[node.dof-1]
    
    def all_to_elements(self) -> None:
        "agrega a los resultados de los elementos"
        for element in self.system.element_map.values():
            element.desplacement = self.local_desplacements_elements[element.id]
            element.internal_forces = self.local_internal_forces_elements[element.id]
        