import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milcapy.core.node import Node

class RigidLink:
    def __init__(self, node_master: "Node", node_slave: "Node", r_AB: np.ndarray):
        self.node_master = node_master  # Nodo maestro (A)
        self.node_slave = node_slave    # Nodo esclavo (B)
        self.r_AB = np.array(r_AB)      # Vector [x, y] de A a B

    def get_constraint_matrix(self):
        """Retorna la matriz de transformación T (3x3)."""
        x, y = self.r_AB
        T = np.array([
            [1, 0, -y],
            [0, 1,  x],
            [0, 0,  1]
        ])
        return T

    def apply_constraints(self, global_stiffness: np.ndarray, global_load: np.ndarray):
        """Aplica restricciones a la matriz de rigidez y vector de cargas."""
        # Ejemplo simplificado usando eliminación de GDL esclavos
        T = self.get_constraint_matrix()
        idx_master = self.node_master.dofs  # Índices de GDL maestro (ej: [0, 1, 2])
        idx_slave = self.node_slave.dofs    # Índices de GDL esclavo (ej: [3, 4, 5])

        # Modificar matriz de rigidez y vector de carga
        K_slave_master = global_stiffness[idx_slave, :][:, idx_master]
        global_stiffness[idx_master, :] += T.T @ K_slave_master
        global_stiffness[:, idx_master] += K_slave_master.T @ T
        global_stiffness = np.delete(global_stiffness, idx_slave, axis=0)
        global_stiffness = np.delete(global_stiffness, idx_slave, axis=1)

        # Similar para el vector de cargas
        F_slave_master = global_load[idx_slave]
        global_load[idx_master] += T.T @ F_slave_master
        global_load = np.delete(global_load, idx_slave)

        return global_stiffness, global_load