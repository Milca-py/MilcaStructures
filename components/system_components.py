"""
Métodos de análisis estructural mediante el método de rigidez.
"""

from typing import TYPE_CHECKING, Tuple
import numpy as np

if TYPE_CHECKING:
    from core.system import SystemMilcaModel

def calculate_load_vector(modelo: "SystemMilcaModel") -> np.ndarray:
    """Calcula el vector de carga global del sistema.

    Args:
        modelo (SystemMilcaModel): Modelo estructural.

    Returns:
        np.ndarray: Vector de carga global.
    """
    nn = len(modelo.node_map)
    F = np.zeros(nn * 3)

    # CONSTRUCCION DEL VECTOR DE FUERZAS
    # Asignar fuerzas nodales alamacenadas en los nodos
    for nodo in modelo.node_map.values():
        F[nodo.dof[0]-1] = nodo.forces.fx
        F[nodo.dof[1]-1] = nodo.forces.fy
        F[nodo.dof[2]-1] = nodo.forces.mz

    # Agregar las cargas MEP que estaban almacendos en los elementos
    for elemento in modelo.element_map.values():
        elemento.compile_stiffness_matrix()
        elemento.compile_transformation_matrix()
        elemento.compile_load_vector()
        elemento.compile_stiffness_matrix_global()
        elemento.compile_load_vector_global()

        f = elemento.load_vector_global
        dof = elemento._dof_map
        F[dof-1] += f

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
        k = elemento.stiffness_matrix_global
        dof = elemento._dof_map
        for i in range(6):
            for j in range(6):
                K[dof[i]-1, dof[j]-1] += k[i, j]

    return K

def process_conditions(modelo: "SystemMilcaModel") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica condiciones de frontera.

    Args:
        modelo (SystemMilcaModel): Modelo estructural.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        - Matriz de rigidez reducida.
        - Vector de fuerzas reducido.
        - Grados de libertad libres.
    """
    nn = len(modelo.node_map)
    restricciones = np.zeros(nn * 3, dtype=bool)

    # Asignar restricciones a cada nodo
    for nodo in modelo.node_map.values():
        restricciones[nodo.dof[0]-1] = nodo.restraints[0]
        restricciones[nodo.dof[1]-1] = nodo.restraints[1]
        restricciones[nodo.dof[2]-1] = nodo.restraints[2]

    # Identificar grados de libertad libres
    dofs_libres = np.where(~restricciones)[0]

    # Reducir la matriz de rigidez y el vector de fuerzas
    K_red = modelo.global_stiffness_matrix[np.ix_(dofs_libres, dofs_libres)]
    F_red = modelo.global_force_vector[dofs_libres]

    return K_red, F_red, dofs_libres

def solve(modelo: "SystemMilcaModel") -> Tuple[np.ndarray, np.ndarray]:
    """Resuelve el sistema de ecuaciones F = KU.

    Args:
        modelo (SystemMilcaModel): Modelo estructural.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
        - Desplazamientos nodales.
        - Reacciones en los apoyos.
    """
    K_red, F_red, dofs_libres = process_conditions(modelo)

    # Resolver el sistema de ecuaciones
    U_red = np.linalg.solve(K_red, F_red)

    # Colocar los desplazamientos en los grados de libertad libres
    nn = len(modelo.node_map)
    desplazamientos = np.zeros(nn * 3)
    desplazamientos[dofs_libres] = U_red

    # Calcular las reacciones en los apoyos
    reacciones = modelo.global_stiffness_matrix @ desplazamientos - modelo.global_force_vector

    return desplazamientos, reacciones
