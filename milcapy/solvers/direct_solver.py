import numpy as np
import time

from typing import TYPE_CHECKING
from milcapy.assembly.dof_mapper import DOFMapper

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from milcapy.analysis.static import LinearStaticOptions


class DirectStiffnessSolver:
    """
    Solucionador basado en el Método de Rigidez Directa.
    Resuelve sistemas de ecuaciones estructurales K*u = F usando métodos de solución directa.
    Usa numpy.linalg.solve(K, F)
    """

    def __init__(
        self, 
        model: "SystemMilcaModel",
        analysis_options: "LinearStaticOptions",
        ) -> None:
        """
        Inicializa el solucionador con el modelo y el método de solución.

        Args:
            model: Modelo estructural que contiene K_global y F_global.
            method (str): Método de solución. Opciones: "LU", "Cholesky", "QR".
        """
        self.model = model
        self.K_global = model.global_stiffness_matrix  # Referencia a la matriz de rigidez del modelo
        self.F_global = model.global_load_vector  # Referencia al vector de cargas
        self.analysis_options = analysis_options

        # Información de grados de libertad
        self._dof_mapper = DOFMapper(self.model)
        self.dofs = self._dof_mapper.dofs
        self.free_dofs = self._dof_mapper.free_dofs
        self.restrained_dofs = self._dof_mapper.restrained_dofs
        # self.constrained_dofs = self.dof_mapper.constrained_dofs 

        # Rendimiento y diagnóstico
        self.solution_time = 0.0
        self.assembly_time = 0.0

    def assemble_global_load_vector(self) -> np.ndarray:
        """Calcula el vector de carga global del sistema.

        Returns:
            np.ndarray: Vector de carga global.
        """
        star_time = time.time()
        
        nn = len(self.model.node_map)
        self.model.global_load_vector = np.zeros(nn * 3)
        F = self.model.global_load_vector

        # Asignar fuerzas nodales almacenadas en los nodos
        for nodo in self.model.node_map.values():
            F[nodo.dof[0] - 1] = nodo.forces.fx
            F[nodo.dof[1] - 1] = nodo.forces.fy
            F[nodo.dof[2] - 1] = nodo.forces.mz

        # Agregar el vector de fuerzas globales almacenadas en los elementos
        for elemento in self.model.element_map.values():
            f = elemento.global_load_vector
            dof = elemento.dof_map
            F[dof - 1] += f

        end_time = time.time()
        self.assembly_time += (end_time - star_time)
        
        return F

    def assemble_global_stiffness_matrix(self) -> np.ndarray:
        """Ensamblaje de la matriz de rigidez global.

        Returns:
            np.ndarray: Matriz de rigidez global.
        """
        star_time = time.time()
        nn = len(self.model.node_map)
        self.model.global_stiffness_matrix = np.zeros((nn * 3, nn * 3))
        K = self.model.global_stiffness_matrix

        # Ensamblar la matriz de rigidez global
        for elemento in self.model.element_map.values():
            k = elemento.global_stiffness_matrix
            dof = elemento.dof_map
            
            rows = np.repeat(dof - 1, 6)
            cols = np.tile(dof - 1, 6)
            values = k.flatten()
            
            for i, j, val in zip(rows, cols, values):
                K[i, j] += val

        end_time = time.time()
        self.assembly_time += (end_time - star_time)

        return K

    def apply_boundary_conditions(self):
        """Aplica condiciones de frontera.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - K_d: Matriz de rigidez para grados de libertad libres.
                - K_dc: Matriz de rigidez que relaciona GDL libres con restringidos.
                - K_cd: Matriz de rigidez que relaciona GDL restringidos con libres.
                - K_c: Matriz de rigidez para GDL restringidos.
                - F_d: Vector de fuerzas para GDL libres.
                - F_c: Vector de fuerzas para GDL restringidos.
        """
        # Identificar grados de libertad libres y restringidos
        free_dofs = self.free_dofs
        restrained_dofs = self.restrained_dofs

        # Reducir la matriz de rigidez (d: desconocidos, c: conocidos)
        #     | Kd   Kdc |
        # K = |          |
        #     | Kcd   Kc |
        K_d = self.model.global_stiffness_matrix[np.ix_(free_dofs, free_dofs)]
        K_dc = self.model.global_stiffness_matrix[np.ix_(free_dofs, restrained_dofs)]
        K_cd = self.model.global_stiffness_matrix[np.ix_(restrained_dofs, free_dofs)]
        K_c = self.model.global_stiffness_matrix[np.ix_(restrained_dofs, restrained_dofs)]
        
        # Reducir el vector de fuerzas
        #     | Fd |
        # F = |    |
        #     | Fc |
        F_d = self.model.global_load_vector[free_dofs]
        F_c = self.model.global_load_vector[restrained_dofs]

        return K_d, K_dc, K_cd, K_c, F_d, F_c

    def solve(self):
        """Resuelve el sistema de ecuaciones F = KU.

        Args:
            modelo (SystemMilcaModel): Modelo estructural.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Desplazamientos nodales
                - Reacciones en los apoyos.
        """

        star_time = time.time()
        
        K_d, K_dc, K_cd, K_c, F_d, F_c = self.apply_boundary_conditions()

        free_dofs = self.free_dofs
        restrained_dofs = self.restrained_dofs
        # Resolver el sistema de ecuaciones
        U_red = np.linalg.solve(K_d, F_d)

        # Colocar los desplazamientos en los grados de libertad libres
        nn = len(self.model.node_map)
        
        self.model.displacements = np.zeros(nn * 3)
        self.model.displacements[free_dofs] = U_red

        # Calcular las reacciones en los apoyos
        # R = Kcd * Ud - Fc
        reacciones = K_cd @ U_red - F_c
        
        # Completar el vector de reacciones
        self.model.reactions = np.zeros(nn * 3)
        self.model.reactions[restrained_dofs] = reacciones

        # Otra forma de alcular las reacciones en los apoyos, para no estar completando el vector de reacciones
        #  R  =  K_global * U_global - F_global
        # reacciones = modelo.global_stiffness_matrix @ desplazamientos - modelo.global_load_vector

        end_time = time.time()
        self.solution_time = (end_time - star_time)

        return self.model.displacements, self.model.reactions