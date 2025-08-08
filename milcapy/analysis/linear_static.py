from typing import TYPE_CHECKING, Tuple
import numpy as np
import time

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel



class LinearStaticAnalysis:
    """
    Solucionador basado en el Método de Rigidez Directa.
    Resuelve sistemas de ecuaciones estructurales K*u = F usando métodos de solución directa.
    Usa numpy.linalg.solve(K, F)
    """

    def __init__(
        self,
        model: "SystemMilcaModel",
        ) -> None:
        """
        Inicializa el solucionador con el modelo y el método de solución.

        Args:
            model: Modelo estructural que contiene K_global y F_global.
        """
        self.model = model

        # Rendimiento y diagnóstico
        self.solution_time = 0.0
        self.assembly_time = 0.0


    def _dof_map(self):
        """Mapea los grados de libertad del modelo para trabajar con numpy.
            [0, 1, 2, ] pertenece al nodo 1 y así sucesivamente."""
        nn = len(self.model.nodes)
        restraints = np.zeros(nn * 3, dtype=bool)

        for node in self.model.nodes.values():
            restraints[node.dofs[0] - 1] = node.restraints[0]
            restraints[node.dofs[1] - 1] = node.restraints[1]
            restraints[node.dofs[2] - 1] = node.restraints[2]

        free_dofs = np.where(~restraints)[0]
        restrained_dofs = np.where(restraints)[0]

        return free_dofs, restrained_dofs

    def assemble_global_load_vector(self) -> np.ndarray:
        """Calcula el vector de carga global del sistema.

        Returns:
            np.ndarray: Vector de carga global.
        """
        start_time = time.time()

        nn = len(self.model.nodes)
        F = np.zeros(nn * 3)

        # Asignar fuerzas nodales almacenadas en los nodos
        for node in self.model.nodes.values():
            f = node.load_vector()
            dofs = node.dofs
            F[dofs - 1] += f

        # Agregar el vector de fuerzas globales almacenadas en los miembros
        for member in self.model.members.values():
            f = member.global_load_vector()
            dofs = member.dofs
            F[dofs - 1] += f

        end_time = time.time()
        self.assembly_time += (end_time - start_time)

        return F

    def assemble_global_stiffness_matrix(self) -> np.ndarray:
        """Ensamblaje de la matriz de rigidez global.

        Returns:
            np.ndarray: Matriz de rigidez global.
        """
        start_time = time.time()
        nn = len(self.model.nodes)
        K = np.zeros((nn * 3, nn * 3))

        # Ensamblar la matriz de rigidez global
        for member in self.model.members.values():
            k = member.global_stiffness_matrix()
            dofs = member.dofs

            rows = np.repeat(dofs - 1, 6)
            cols = np.tile(dofs - 1, 6)
            values = k.flatten()

            for i, j, val in zip(rows, cols, values):
                K[i, j] += val

        end_time = time.time()
        self.assembly_time += (end_time - start_time)

        return K

    def apply_boundary_conditions(
        self,
        K: np.ndarray,
        F: np.ndarray,
        free_dofs: np.ndarray,
        restrained_dofs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aplica las condiciones de frontera.

        Args:
            K (np.ndarray): Matriz de rigidez global.
            F (np.ndarray): Vector de fuerzas global.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - K_d: Matriz de rigidez para grados de libertad libres.
                - K_dc: Matriz de rigidez que relaciona GDL libres con restringidos.
                - K_cd: Matriz de rigidez que relaciona GDL restringidos con libres.
                - K_c: Matriz de rigidez para GDL restringidos.
                - F_d: Vector de fuerzas para GDL libres.
                - F_c: Vector de fuerzas para GDL restringidos.
        """
        # Reducir la matriz de rigidez (d: desconocidos, c: conocidos)
        #     | Kd   Kdc |
        # K = |          |
        #     | Kcd   Kc |
        K_d  = K[np.ix_(free_dofs, free_dofs)]
        K_dc = K[np.ix_(free_dofs, restrained_dofs)]
        K_cd = K[np.ix_(restrained_dofs, free_dofs)]
        K_c  = K[np.ix_(restrained_dofs, restrained_dofs)]

        # Reducir el vector de fuerzas
        #     | Fd |
        # F = |    |
        #     | Fc |
        F_d = F[free_dofs]
        F_c = F[restrained_dofs]

        return K_d, K_dc, K_cd, K_c, F_d, F_c

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Resuelve el sistema de ecuaciones F = KU.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Desplazamientos nodales
                - Reacciones en los apoyos.
        """

        start_time = time.time()
        # Obtener la matriz de rigidez global y el vector de fuerzas global
        K = self.assemble_global_stiffness_matrix()
        F = self.assemble_global_load_vector()

        # Aplicar las condiciones de frontera
        free_dofs, restrained_dofs = self._dof_map()
        K_d, K_dc, K_cd, K_c, F_d, F_c = self.apply_boundary_conditions(K, F, free_dofs, restrained_dofs)

        # Resolver el sistema de ecuaciones
        #     |Ud|
        # U = |  | = displacements
        #     |Uc|

        U_d = np.linalg.solve(K_d, F_d) # respuesta solo para los maestros

        # Colocar los desplazamientos en los grados de libertad libres
        nn = len(self.model.nodes)
        # completar el vector de desplazamientos
        displacements = np.zeros(nn * 3)
        displacements[free_dofs] = U_d

        # Calcular las reacciones en los apoyos
        # R = Kcd * Ud - Fc
        R = K_cd @ U_d - F_c
        # Completar el vector de reacciones
        reactions = np.zeros(nn * 3)
        reactions[restrained_dofs] = R

        # Otra forma de alcular las reacciones en los apoyos, para no estar completando el vector de reacciones
        #  R  =  K_global * U_global - F_global
        # reactions = K @ displacements - F

        end_time = time.time()
        self.solution_time = (end_time - start_time)

        return displacements, reactions