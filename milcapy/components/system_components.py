"""
Métodos de análisis estructural mediante el método de rigidez.
"""

from typing import TYPE_CHECKING, Tuple
import numpy as np

if TYPE_CHECKING:
    from milcapy.core.system import SystemMilcaModel


def assemble_global_load_vector(modelo: "SystemMilcaModel") -> np.ndarray:
    """Calcula el vector de carga global del sistema.

    Args:
        modelo (SystemMilcaModel): Modelo estructural.

    Returns:
        np.ndarray: Vector de carga global.
    """
    nn = len(modelo.node_map)
    F = np.zeros(nn * 3)

    # Construcción del vector de fuerzas
    # Asignar fuerzas nodales almacenadas en los nodos
    for nodo in modelo.node_map.values():
        F[nodo.dof[0] - 1] = nodo.forces.fx
        F[nodo.dof[1] - 1] = nodo.forces.fy
        F[nodo.dof[2] - 1] = nodo.forces.mz

    # Agregar las cargas MEP que estaban almacenadas en los elementos
    for elemento in modelo.element_map.values():
        f = elemento.global_load_vector
        dof = elemento.dof_map
        F[dof - 1] += f

    return F


def assemble_global_stiffness_matrix(modelo: "SystemMilcaModel") -> np.ndarray:
    """Ensamblaje de la matriz de rigidez global.

    Args:
        modelo (SystemMilcaModel): Modelo estructural.

    Returns:
        np.ndarray: Matriz de rigidez global.
    """
    nn = len(modelo.node_map)
    K = np.zeros((nn * 3, nn * 3))

    # Ensamblar la matriz de rigidez global
    for elemento in modelo.element_map.values():
        k = elemento.global_stiffness_matrix
        dof = elemento.dof_map
        
        rows = np.repeat(dof - 1, 6)
        cols = np.tile(dof - 1, 6)
        values = k.flatten()
        
        for i, j, val in zip(rows, cols, values):
            K[i, j] += val
        
        # for i in range(6):
        #     for j in range(6):
        #         K[dof[i]-1, dof[j]-1] += k[i, j]

    return K


def process_conditions(modelo: "SystemMilcaModel") -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                         np.ndarray, np.ndarray, np.ndarray, 
                                                         np.ndarray, np.ndarray]:
    """Aplica condiciones de frontera.

    Args:
        modelo (SystemMilcaModel): Modelo estructural.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - K_d: Matriz de rigidez para grados de libertad libres.
            - K_dc: Matriz de rigidez que relaciona GDL libres con restringidos.
            - K_cd: Matriz de rigidez que relaciona GDL restringidos con libres.
            - K_c: Matriz de rigidez para GDL restringidos.
            - F_d: Vector de fuerzas para GDL libres.
            - F_c: Vector de fuerzas para GDL restringidos.
            - dofs_libres: Índices de los GDL libres.
            - dofs_restringidos: Índices de los GDL restringidos.
    """
    nn = len(modelo.node_map)
    restricciones = np.zeros(nn * 3, dtype=bool)

    # Asignar restricciones a cada nodo
    for nodo in modelo.node_map.values():
        restricciones[nodo.dof[0] - 1] = nodo.restraints[0]
        restricciones[nodo.dof[1] - 1] = nodo.restraints[1]
        restricciones[nodo.dof[2] - 1] = nodo.restraints[2]

    # Identificar grados de libertad libres y restringidos
    dofs_libres = np.where(~restricciones)[0]
    dofs_restringidos = np.where(restricciones)[0]

    # Reducir la matriz de rigidez 
    #     | Kd   Kdc |
    # K = |          |
    #     | Kcd   Kc |
    K_d = modelo.global_stiffness_matrix[np.ix_(dofs_libres, dofs_libres)]
    K_dc = modelo.global_stiffness_matrix[np.ix_(dofs_libres, dofs_restringidos)]
    K_cd = modelo.global_stiffness_matrix[np.ix_(dofs_restringidos, dofs_libres)]
    K_c = modelo.global_stiffness_matrix[np.ix_(dofs_restringidos, dofs_restringidos)]
    
    # Reducir el vector de fuerzas
    #     | Fd |
    # F = |    |
    #     | Fc |
    F_d = modelo.global_load_vector[dofs_libres]
    F_c = modelo.global_load_vector[dofs_restringidos]

    return K_d, K_dc, K_cd, K_c, F_d, F_c, dofs_libres, dofs_restringidos


def solve(modelo: "SystemMilcaModel") -> Tuple[np.ndarray, np.ndarray]:
    """Resuelve el sistema de ecuaciones F = KU.

    Args:
        modelo (SystemMilcaModel): Modelo estructural.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Desplazamientos nodales.
            - Reacciones en los apoyos.
    """
    K_d, K_dc, K_cd, K_c, F_d, F_c, dofs_libres, dofs_restringidos = process_conditions(modelo)

    # Resolver el sistema de ecuaciones
    U_red = np.linalg.solve(K_d, F_d)

    # Colocar los desplazamientos en los grados de libertad libres
    nn = len(modelo.node_map)
    desplazamientos = np.zeros(nn * 3)
    desplazamientos[dofs_libres] = U_red

    # Calcular las reacciones en los apoyos
    # R = Kcd * Ud - Fc
    reacciones = K_cd @ U_red - F_c
    
    # Completar el vector de reacciones
    reacciones_completas = np.zeros(nn * 3)
    reacciones_completas[dofs_restringidos] = reacciones

    # Otra forma de alcular las reacciones en los apoyos, para no estar completando el vector de reacciones
    #  R  =  K_global * U_global - F_global
    # reacciones = modelo.global_stiffness_matrix @ desplazamientos - modelo.global_load_vector

    return desplazamientos, reacciones_completas