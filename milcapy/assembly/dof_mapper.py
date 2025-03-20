import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel



class DOFMapper:
    def __init__(self, model):
        self.model: "SystemMilcaModel" = model
        self.node_ids = None
        self.restraints = None              # restricciones en orden de los nodos
        self.dofs = None
        self.free_dofs = None
        self.restrained_dofs = None
        # self.constrained_dofs = None
    
        self.map_dofs()

    def map_dofs(self):
        """Mapea los grados de libertad del modelo.
            [0, 1, 2, ] pertenece al nodo 1 y así sucesivamente.
        """
        nn = len(self.model.node_map)
        self.node_ids = np.zeros(nn, dtype=int)
        self.dofs = np.zeros(nn * 3, dtype=int)
        self.restraints = np.zeros(nn * 3, dtype=bool)
        
        for node in self.model.node_map.values():
            self.restraints[node.dof[0] - 1] = node.restraints[0]
            self.restraints[node.dof[1] - 1] = node.restraints[1]
            self.restraints[node.dof[2] - 1] = node.restraints[2]
            
            self.dofs[node.dof[0] - 1] = node.dof[0]
            self.dofs[node.dof[1] - 1] = node.dof[1]
            self.dofs[node.dof[2] - 1] = node.dof[2]
            
            self.node_ids[node.id - 1] = node.id
        
        self.free_dofs = np.where(~self.restraints)[0]
        self.restrained_dofs = np.where(self.restraints)[0]