from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import time
from scipy.sparse import linalg as sparse_linalg
import numpy as np


class Utils:
    """
    Clase con funciones utilitarias para análisis estructural.

    Proporciona un conjunto de métodos estáticos para operaciones comunes
    en el análisis estructural y el procesamiento de datos relacionados.
    """

    @staticmethod
    def rotation_matrix_2d(angle_rad):
        """
        Crea una matriz de rotación 2D para el ángulo dado.

        Args:
            angle_rad (float): Ángulo de rotación en radianes.

        Returns:
            numpy.ndarray: Matriz de rotación 2D (2x2).
        """
        import numpy as np

        c = np.cos(angle_rad)
        s = np.sin(angle_rad)

        return np.array([
            [c, -s],
            [s, c]
        ])

    @staticmethod
    def rotation_matrix_3d(angles):
        """
        Crea una matriz de rotación 3D para los ángulos dados.

        Args:
            angles (tuple): Ángulos de rotación (en radianes) alrededor de los ejes X, Y, Z.

        Returns:
            numpy.ndarray: Matriz de rotación 3D (3x3).
        """
        import numpy as np

        # Desempaquetar ángulos
        theta_x, theta_y, theta_z = angles

        # Rotación alrededor del eje X
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        # Rotación alrededor del eje Y
        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        # Rotación alrededor del eje Z
        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        # Combinar rotaciones: R = Rz * Ry * Rx (orden de aplicación)
        R = np.matmul(Rz, np.matmul(Ry, Rx))

        return R

    @staticmethod
    def direction_cosines(start_point, end_point):
        """
        Calcula los cosenos directores de un segmento de línea en 3D.

        Args:
            start_point (tuple): Coordenadas (x, y, z) del punto inicial.
            end_point (tuple): Coordenadas (x, y, z) del punto final.

        Returns:
            tuple: Cosenos directores (cos_x, cos_y, cos_z).
        """
        import numpy as np

        # Vector desde el punto inicial al final
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        dz = end_point[2] - start_point[2]

        # Longitud del vector
        length = np.sqrt(dx**2 + dy**2 + dz**2)

        if length < 1e-10:  # Prevenir división por cero
            return (0, 0, 0)

        # Cosenos directores
        cos_x = dx / length
        cos_y = dy / length
        cos_z = dz / length

        return (cos_x, cos_y, cos_z)

    @staticmethod
    def local_coordinate_system(direction_vector, up_vector=None):
        """
        Genera un sistema de coordenadas locales ortogonales dado un vector de dirección.

        Args:
            direction_vector (tuple): Vector que define la dirección del eje x local.
            up_vector (tuple, opcional): Vector que ayuda a definir el plano xy local.
                                       Si es None, se genera automáticamente.

        Returns:
            dict: Diccionario con los vectores unitarios de los ejes locales x, y, z.
        """
        import numpy as np

        # Normalizar el vector de dirección para el eje x local
        v = np.array(direction_vector)
        length = np.linalg.norm(v)

        if length < 1e-10:
            return None

        local_x = v / length

        # Si no se proporciona un vector de referencia, crear uno automáticamente
        if up_vector is None:
            # Elegir un vector que no sea paralelo a local_x
            if abs(local_x[0]) < 0.9:  # Si x no es dominante
                up_vector = (1, 0, 0)
            else:  # Si x es dominante, usar y
                up_vector = (0, 1, 0)

        up = np.array(up_vector)

        # Generar el eje z local (producto cruz de x y up_vector)
        local_z = np.cross(local_x, up)
        z_length = np.linalg.norm(local_z)

        # Si local_x y up_vector son casi paralelos, elegir otro up_vector
        if z_length < 1e-10:
            if abs(local_x[1]) < 0.9:  # Si y no es dominante
                up_vector = (0, 1, 0)
            else:  # Si y es dominante, usar z
                up_vector = (0, 0, 1)
            up = np.array(up_vector)
            local_z = np.cross(local_x, up)
            z_length = np.linalg.norm(local_z)

        local_z = local_z / z_length  # Normalizar z

        # Generar el eje y local (producto cruz de z y x)
        local_y = np.cross(local_z, local_x)

        return {
            'x': local_x.tolist(),
            'y': local_y.tolist(),
            'z': local_z.tolist()
        }

    @staticmethod
    def create_transformation_matrix(local_system, dimension=3):
        """
        Crea una matriz de transformación de coordenadas locales a globales.

        Args:
            local_system (dict): Sistema de coordenadas locales (vectores unitarios).
            dimension (int): Dimensión del problema (2 o 3).

        Returns:
            numpy.ndarray: Matriz de transformación.
        """
        import numpy as np

        if dimension == 2:
            # Matriz de transformación 2D (rotación en el plano)
            local_x = local_system['x'][:2]  # Solo componentes x, y
            local_y = local_system['y'][:2]  # Solo componentes x, y

            T = np.array([
                [local_x[0], local_x[1]],
                [local_y[0], local_y[1]]
            ]).T  # Transpuesta para que los vectores queden como filas

            return T

        elif dimension == 3:
            # Matriz de transformación 3D completa
            local_x = local_system['x']
            local_y = local_system['y']
            local_z = local_system['z']

            T = np.array([
                [local_x[0], local_x[1], local_x[2]],
                [local_y[0], local_y[1], local_y[2]],
                [local_z[0], local_z[1], local_z[2]]
            ]).T  # Transpuesta para que los vectores queden como filas

            return T

        else:
            raise ValueError(f"Dimensión no soportada: {dimension}")

    @staticmethod
    def expand_transformation_matrix(basic_transform, dofs_per_node):
        """
        Expande una matriz de transformación básica para incluir múltiples DOFs por nodo.

        Args:
            basic_transform (numpy.ndarray): Matriz de transformación básica (2x2 o 3x3).
            dofs_per_node (int): Número de DOFs por nodo.

        Returns:
            numpy.ndarray: Matriz de transformación expandida.
        """
        import numpy as np

        n = basic_transform.shape[0]  # Dimensión de la transformación básica

        if dofs_per_node % n != 0:
            raise ValueError(
                f"El número de DOFs por nodo debe ser múltiplo de {n}")

        blocks = dofs_per_node // n  # Número de bloques de transformación por nodo

        # Crear matriz expandida
        expanded = np.zeros((dofs_per_node, dofs_per_node))

        # Llenar con bloques de la transformación básica
        for i in range(blocks):
            row_start = i * n
            row_end = (i + 1) * n

            col_start = i * n
            col_end = (i + 1) * n

            expanded[row_start:row_end, col_start:col_end] = basic_transform

        return expanded

    @staticmethod
    def transform_stiffness_matrix(local_stiffness, transform_matrix):
        """
        Transforma una matriz de rigidez local a global usando una matriz de transformación.

        Args:
            local_stiffness (numpy.ndarray): Matriz de rigidez en coordenadas locales.
            transform_matrix (numpy.ndarray): Matriz de transformación.

        Returns:
            numpy.ndarray: Matriz de rigidez en coordenadas globales.
        """
        import numpy as np

        # K_global = T^T * K_local * T
        return np.matmul(np.matmul(transform_matrix.T, local_stiffness), transform_matrix)

    @staticmethod
    def transform_load_vector(local_load, transform_matrix):
        """
        Transforma un vector de carga local a global usando una matriz de transformación.

        Args:
            local_load (numpy.ndarray): Vector de carga en coordenadas locales.
            transform_matrix (numpy.ndarray): Matriz de transformación.

        Returns:
            numpy.ndarray: Vector de carga en coordenadas globales.
        """
        import numpy as np

        # F_global = T^T * F_local
        return np.matmul(transform_matrix.T, local_load)

    @staticmethod
    def transform_displacement_vector(global_displacement, transform_matrix):
        """
        Transforma un vector de desplazamiento global a local usando una matriz de transformación.

        Args:
            global_displacement (numpy.ndarray): Vector de desplazamiento en coordenadas globales.
            transform_matrix (numpy.ndarray): Matriz de transformación.

        Returns:
            numpy.ndarray: Vector de desplazamiento en coordenadas locales.
        """
        import numpy as np

        # u_local = T * u_global
        return np.matmul(transform_matrix, global_displacement)


class Vertex:
    """
    Clase que representa un vértice geométrico en el espacio.

    Un vértice es simplemente un punto geométrico en el espacio 2D o 3D
    sin ningún comportamiento físico asociado.
    """

    def __init__(self, id, x, y, z=0.0):
        """
        Inicializa un vértice con sus coordenadas.

        Args:
            id (int): Identificador único del vértice.
            x (float): Coordenada en dirección X.
            y (float): Coordenada en dirección Y.
            z (float): Coordenada en dirección Z (por defecto 0.0 para casos 2D).
        """
        self.id = id
        self.x = x
        self.y = y
        self.z = z

    def set_coordinates(self, x, y, z=None):
        """
        Actualiza las coordenadas del vértice.

        Args:
            x (float): Nueva coordenada en dirección X.
            y (float): Nueva coordenada en dirección Y.
            z (float, opcional): Nueva coordenada en dirección Z.
        """
        self.x = x
        self.y = y
        if z is not None:
            self.z = z

    def get_coordinates(self):
        """
        Obtiene las coordenadas del vértice como una tupla.

        Returns:
            tuple: Tupla con las coordenadas (x, y, z).
        """
        return (self.x, self.y, self.z)

    def get_dimension(self):
        """
        Determina la dimensión en la que se encuentra el vértice.

        Returns:
            int: 2 para vértices en el plano XY (z=0), 3 para vértices en el espacio.
        """
        return 3 if abs(self.z) > 1e-10 else 2

    def distance_to(self, other_vertex):
        """
        Calcula la distancia euclidiana entre este vértice y otro.

        Args:
            other_vertex (Vertex): Otro vértice para calcular la distancia.

        Returns:
            float: Distancia euclidiana entre los dos vértices.
        """
        return ((self.x - other_vertex.x)**2 +
                (self.y - other_vertex.y)**2 +
                (self.z - other_vertex.z)**2)**0.5

    def midpoint(self, other_vertex):
        """
        Calcula el punto medio entre este vértice y otro.

        Args:
            other_vertex (Vertex): Otro vértice.

        Returns:
            Vertex: Un nuevo vértice que representa el punto medio.
        """
        mid_x = (self.x + other_vertex.x) / 2.0
        mid_y = (self.y + other_vertex.y) / 2.0
        mid_z = (self.z + other_vertex.z) / 2.0
        return Vertex(None, mid_x, mid_y, mid_z)

    def translate(self, dx, dy, dz=0.0):
        """
        Traslada el vértice según los desplazamientos dados.

        Args:
            dx (float): Desplazamiento en dirección X.
            dy (float): Desplazamiento en dirección Y.
            dz (float): Desplazamiento en dirección Z.
        """
        self.x += dx
        self.y += dy
        self.z += dz

    def __str__(self):
        """
        Representación en string del vértice.

        Returns:
            str: Representación legible del vértice y sus coordenadas.
        """
        return f"Vertex {self.id}: ({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


class Constraint:
    """
    Clase que representa una restricción general entre grados de libertad en un análisis estructural.

    Puede usarse para representar relaciones entre DOFs como restricciones de cuerpo rígido,
    ecuaciones de restricción multipoint (MPC), condiciones de simetría, etc.
    """

    def __init__(self, id, constraint_type):
        """
        Inicializa una restricción general.

        Args:
            id (int): Identificador único de la restricción.
            constraint_type (str): Tipo de restricción ('mpc', 'rigid', 'symmetry', etc.).
        """
        self.id = id
        self.constraint_type = constraint_type
        self.terms = []  # Lista de términos en la ecuación de restricción
        self.constant_term = 0.0  # Término constante de la ecuación

    def add_term(self, node, dof_name, coefficient):
        """
        Añade un término a la ecuación de restricción.

        Args:
            node (Node): Nodo asociado al término.
            dof_name (str): Nombre del grado de libertad.
            coefficient (float): Coeficiente del término en la ecuación.

        Returns:
            bool: True si se añadió correctamente, False en caso contrario.
        """
        # Verificar que el DOF exista en el nodo
        if dof_name not in node.dofs.indices:
            return False

        self.terms.append({
            'node': node,
            'dof': dof_name,
            'coefficient': coefficient
        })
        return True

    def set_constant_term(self, value):
        """
        Establece el valor del término constante en la ecuación de restricción.

        Args:
            value (float): Valor del término constante.
        """
        self.constant_term = value

    def get_constraint_equation(self):
        """
        Obtiene la ecuación de restricción en forma de string.

        Returns:
            str: Representación de la ecuación de restricción.
        """
        equation = ""
        for i, term in enumerate(self.terms):
            coefficient = term['coefficient']
            node_id = term['node'].id
            dof = term['dof']

            if i > 0:
                if coefficient >= 0:
                    equation += " + "
                else:
                    equation += " - "
                    coefficient = abs(coefficient)
            elif coefficient < 0:
                equation += "-"
                coefficient = abs(coefficient)

            if abs(coefficient - 1.0) < 1e-10:
                equation += f"{dof}_{node_id}"
            else:
                equation += f"{coefficient:.4f}*{dof}_{node_id}"

        if abs(self.constant_term) > 1e-10:
            if self.constant_term >= 0:
                equation += f" = {self.constant_term:.4f}"
            else:
                equation += f" = {self.constant_term:.4f}"
        else:
            equation += " = 0"

        return equation

    def get_master_slave_form(self):
        """
        Convierte la restricción a forma maestro-esclavo, donde un DOF se expresa
        en función de otros DOFs.

        Returns:
            tuple: (slave_term, master_terms, constant) donde slave_term es el término esclavo,
                  master_terms es una lista de términos maestros y constant es el término constante.
        """
        if not self.terms:
            return None, [], self.constant_term

        # Seleccionar el primer término como esclavo (puede mejorarse con una selección más inteligente)
        slave_term = self.terms[0]
        master_terms = self.terms[1:]

        # Normalizar respecto al coeficiente del esclavo
        slave_coef = slave_term['coefficient']

        normalized_master_terms = []
        for term in master_terms:
            normalized_term = term.copy()
            normalized_term['coefficient'] = -term['coefficient'] / slave_coef
            normalized_master_terms.append(normalized_term)

        normalized_constant = self.constant_term / slave_coef

        return slave_term, normalized_master_terms, normalized_constant

    def apply_to_system(self, global_matrix, global_vector):
        """
        Aplica la restricción al sistema global de ecuaciones usando el método de eliminación.

        Args:
            global_matrix (numpy.ndarray): Matriz global del sistema.
            global_vector (numpy.ndarray): Vector de cargas global.

        Returns:
            tuple: (global_matrix, global_vector) modificados con la restricción aplicada.
        """
        import numpy as np

        # Convertir a forma maestro-esclavo
        slave_term, master_terms, constant = self.get_master_slave_form()

        if slave_term is None:
            return global_matrix, global_vector

        # Obtener índice global del DOF esclavo
        slave_node = slave_term['node']
        slave_dof = slave_term['dof']
        slave_index = slave_node.dofs.indices[slave_dof]

        if slave_index < 0:
            return global_matrix, global_vector

        # Aplicar la restricción mediante eliminación de ecuaciones
        n = global_matrix.shape[0]

        # Ajustar el vector de carga para los términos maestros
        for term in master_terms:
            master_node = term['node']
            master_dof = term['dof']
            master_index = master_node.dofs.indices[master_dof]
            coef = term['coefficient']

            if master_index >= 0:
                # Actualizar el vector de carga
                global_vector -= coef * global_matrix[:, slave_index]

                # Actualizar la matriz
                global_matrix[:, master_index] += coef * \
                    global_matrix[:, slave_index]
                global_matrix[master_index, :] += coef * \
                    global_matrix[slave_index, :]

        # Ajustar el vector de carga para el término constante
        if abs(constant) > 1e-10:
            global_vector -= constant * global_matrix[:, slave_index]

        # Eliminar la fila y columna del DOF esclavo (técnica de eliminación de filas/columnas)
        # Reemplazar la fila/columna con 0's excepto el diagonal que es 1
        global_matrix[slave_index, :] = 0
        global_matrix[:, slave_index] = 0
        global_matrix[slave_index, slave_index] = 1

        # Actualizar el vector de carga para el DOF esclavo
        global_vector[slave_index] = 0

        return global_matrix, global_vector

    def __str__(self):
        """
        Representación en string de la restricción.

        Returns:
            str: Representación legible de la restricción.
        """
        return f"Constraint {self.id} ({self.constraint_type}): {self.get_constraint_equation()}"


class Restraint:
    """
    Clase que representa un conjunto de restricciones de apoyo en una estructura.

    Permite definir y aplicar condiciones de contorno como apoyos simples,
    empotramientos, apoyos elásticos, etc.
    """

    def __init__(self, id, name=None):
        """
        Inicializa un objeto Restraint.

        Args:
            id (int): Identificador único de la restricción.
            name (str, opcional): Nombre descriptivo de la restricción.
        """
        self.id = id
        self.name = name or f"Restraint_{id}"
        self.conditions = {}  # Diccionario de condiciones de restricción por nodo

    def add_fixed_support(self, node):
        """
        Añade una condición de apoyo empotrado (todos los DOFs restringidos).

        Args:
            node (Node): Nodo al que se aplica el empotramiento.

        Returns:
            bool: True si se aplicó correctamente, False en caso contrario.
        """
        if node.id not in self.conditions:
            self.conditions[node.id] = {}

        # Restringir todos los DOFs del nodo
        result = True
        for dof_name in node.dofs.indices.keys():
            node_result = node.apply_constraint(dof_name, 0.0)
            self.conditions[node.id][dof_name] = {
                'type': 'fixed', 'value': 0.0}
            result = result and node_result

        return result

    def add_pinned_support(self, node):
        """
        Añade una condición de apoyo articulado (traslaciones restringidas, rotaciones libres).

        Args:
            node (Node): Nodo al que se aplica la articulación.

        Returns:
            bool: True si se aplicó correctamente, False en caso contrario.
        """
        if node.id not in self.conditions:
            self.conditions[node.id] = {}

        result = True

        # Restringir desplazamientos
        for dof_name in node.dofs.indices.keys():
            if dof_name.startswith('U'):  # DOFs de traslación
                node_result = node.apply_constraint(dof_name, 0.0)
                self.conditions[node.id][dof_name] = {
                    'type': 'fixed', 'value': 0.0}
                result = result and node_result

        return result

    def add_roller_support(self, node, free_direction):
        """
        Añade una condición de apoyo de rodillo (un desplazamiento libre, otros restringidos).

        Args:
            node (Node): Nodo al que se aplica el apoyo.
            free_direction (str): Dirección libre ('X', 'Y', 'Z').

        Returns:
            bool: True si se aplicó correctamente, False en caso contrario.
        """
        if node.id not in self.conditions:
            self.conditions[node.id] = {}

        result = True
        free_dof = f"U{free_direction}"

        # Restringir desplazamientos excepto en la dirección libre
        for dof_name in node.dofs.indices.keys():
            # DOFs de traslación excepto libre
            if dof_name.startswith('U') and dof_name != free_dof:
                node_result = node.apply_constraint(dof_name, 0.0)
                self.conditions[node.id][dof_name] = {
                    'type': 'fixed', 'value': 0.0}
                result = result and node_result

        return result

    def add_elastic_support(self, node, dof_name, stiffness):
        """
        Añade una condición de apoyo elástico con una rigidez específica.

        Args:
            node (Node): Nodo al que se aplica el apoyo elástico.
            dof_name (str): Nombre del grado de libertad ('UX', 'UY', etc.).
            stiffness (float): Coeficiente de rigidez del apoyo elástico.

        Returns:
            bool: True si se aplicó correctamente, False en caso contrario.
        """
        if dof_name not in node.dofs.indices:
            return False

        if node.id not in self.conditions:
            self.conditions[node.id] = {}

        # No aplicar restricción directa, sino almacenar para su posterior aplicación
        # en la matriz global
        self.conditions[node.id][dof_name] = {
            'type': 'elastic', 'stiffness': stiffness}

        return True

    def add_prescribed_displacement(self, node, dof_name, value):
        """
        Añade un desplazamiento prescrito (impuesto) a un grado de libertad.

        Args:
            node (Node): Nodo al que se aplica el desplazamiento prescrito.
            dof_name (str): Nombre del grado de libertad ('UX', 'UY', etc.).
            value (float): Valor del desplazamiento prescrito.

        Returns:
            bool: True si se aplicó correctamente, False en caso contrario.
        """
        if dof_name not in node.dofs.indices:
            return False

        if node.id not in self.conditions:
            self.conditions[node.id] = {}

        # Aplicar restricción con el valor prescrito
        node_result = node.apply_constraint(dof_name, value)
        self.conditions[node.id][dof_name] = {
            'type': 'prescribed', 'value': value}

        return node_result

    def add_settlement(self, node, dof_name, value):
        """
        Añade un asentamiento de apoyo.

        Args:
            node (Node): Nodo al que se aplica el asentamiento.
            dof_name (str): Nombre del grado de libertad ('UX', 'UY', etc.).
            value (float): Valor del asentamiento.

        Returns:
            bool: True si se aplicó correctamente, False en caso contrario.
        """
        # En términos de modelado, es similar a un desplazamiento prescrito
        return self.add_prescribed_displacement(node, dof_name, value)

    def add_custom_restraint(self, node, dof_dict):
        """
        Añade restricciones personalizadas a un nodo.

        Args:
            node (Node): Nodo al que se aplican las restricciones.
            dof_dict (dict): Diccionario de DOFs y sus condiciones de restricción.
                            Formato: {dof_name: {'type': tipo, 'value'/'stiffness': valor}}

        Returns:
            bool: True si todas las restricciones se aplicaron correctamente.
        """
        if node.id not in self.conditions:
            self.conditions[node.id] = {}

        result = True

        for dof_name, condition in dof_dict.items():
            if dof_name not in node.dofs.indices:
                result = False
                continue

            restraint_type = condition.get('type', 'fixed')

            if restraint_type in ['fixed', 'prescribed']:
                value = condition.get('value', 0.0)
                node_result = node.apply_constraint(dof_name, value)
                self.conditions[node.id][dof_name] = {
                    'type': restraint_type, 'value': value}
                result = result and node_result
            elif restraint_type == 'elastic':
                stiffness = condition.get('stiffness', 0.0)
                self.conditions[node.id][dof_name] = {
                    'type': restraint_type, 'stiffness': stiffness}

        return result

    def apply_to_system(self, global_matrix, global_vector):
        """
        Aplica todas las condiciones de restraint al sistema global de ecuaciones.

        Args:
            global_matrix (numpy.ndarray): Matriz global del sistema.
            global_vector (numpy.ndarray): Vector de cargas global.

        Returns:
            tuple: (global_matrix, global_vector) modificados con las restricciones aplicadas.
        """
        import numpy as np

        for node_id, node_conditions in self.conditions.items():
            for dof_name, condition in node_conditions.items():
                # Buscar el nodo en el modelo (debe ser pasado como parámetro o buscado de otra manera)
                # Por simplicidad, asumimos que tenemos una función o manera de obtener el nodo
                # node = get_node_by_id(node_id)
                #
                # Como no tenemos acceso al modelo completo, esto sería parte de una clase Model o Solver
                # Aquí nos limitamos a la lógica básica asumiendo que los índices son conocidos

                # En una implementación real, estos índices vendrían del nodo
                # global_index = node.dofs.indices.get(dof_name, -1)
                global_index = -1  # Placeholder

                if global_index < 0:
                    continue

                restraint_type = condition.get('type', 'fixed')

                if restraint_type in ['fixed', 'prescribed']:
                    value = condition.get('value', 0.0)

                    # Técnica de penalización para restricciones fijas
                    penalty = 1e15  # Factor de penalización (muy alto)

                    # Guardar la diagonal original
                    original_k = global_matrix[global_index, global_index]

                    # Aplicar penalización
                    global_matrix[global_index,
                                  global_index] = original_k + penalty
                    global_vector[global_index] = penalty * value

                elif restraint_type == 'elastic':
                    stiffness = condition.get('stiffness', 0.0)

                    # Añadir la rigidez elástica a la diagonal
                    global_matrix[global_index, global_index] += stiffness

        return global_matrix, global_vector

    def clear(self):
        """
        Elimina todas las condiciones de restricción.
        """
        self.conditions = {}

    def __str__(self):
        """
        Representación en string del objeto Restraint.

        Returns:
            str: Representación legible de las restricciones.
        """
        result = f"Restraint {self.id} ({self.name}):\n"

        for node_id, node_conditions in self.conditions.items():
            result += f"  Node {node_id}:\n"
            for dof_name, condition in node_conditions.items():
                restraint_type = condition.get('type', 'fixed')

                if restraint_type in ['fixed', 'prescribed']:
                    value = condition.get('value', 0.0)
                    result += f"    {dof_name}: {restraint_type}, Value={value:.6f}\n"
                elif restraint_type == 'elastic':
                    stiffness = condition.get('stiffness', 0.0)
                    result += f"    {dof_name}: {restraint_type}, Stiffness={stiffness:.6f}\n"

        return result


class DegreesOfFreedom:
    """
    Clase que representa los grados de libertad en un nodo de la estructura.

    En análisis estructural, los grados de libertad (DOF) son las variables independientes
    que definen el estado de desplazamiento/rotación de un nodo y su posición en el
    sistema global de ecuaciones.
    """

    def __init__(self, node_id, dimension=3):
        """
        Inicializa un objeto DegreesOfFreedom para un nodo específico.

        Args:
            node_id (int): Identificador único del nodo asociado.
            dimension (int): Dimensión del problema (2 para 2D, 3 para 3D).
        """
        self.node_id = node_id
        self.dimension = dimension

        # Inicialización de los índices globales para cada grado de libertad
        # Un valor de -1 indica que el DOF no está asignado en el sistema global
        self.indices = {}

        # Inicialización de restricciones (True = restringido, False = libre)
        self.constraints = {}

        # Inicialización de valores conocidos (desplazamientos/rotaciones impuestos)
        self.prescribed_values = {}

        # Inicialización de los DOFs básicos según la dimensión
        if dimension == 2:
            # Traslación en X, Y y rotación en Z para problemas 2D
            self._init_2d_dofs()
        elif dimension == 3:
            # Traslación en X, Y, Z y rotación en X, Y, Z para problemas 3D
            self._init_3d_dofs()
        else:
            raise ValueError(f"Dimensión no soportada: {dimension}")

    def _init_2d_dofs(self):
        """Inicializa los grados de libertad para un problema 2D."""
        # Desplazamientos
        self.indices['UX'] = -1
        self.indices['UY'] = -1

        # Rotación
        self.indices['RZ'] = -1

        # Inicializa restricciones (por defecto, todos los DOFs son libres)
        self.constraints['UX'] = False
        self.constraints['UY'] = False
        self.constraints['RZ'] = False

        # Inicializa valores prescritos (por defecto, cero)
        self.prescribed_values['UX'] = 0.0
        self.prescribed_values['UY'] = 0.0
        self.prescribed_values['RZ'] = 0.0

    def _init_3d_dofs(self):
        """Inicializa los grados de libertad para un problema 3D."""
        # Desplazamientos
        self.indices['UX'] = -1
        self.indices['UY'] = -1
        self.indices['UZ'] = -1

        # Rotaciones
        self.indices['RX'] = -1
        self.indices['RY'] = -1
        self.indices['RZ'] = -1

        # Inicializa restricciones (por defecto, todos los DOFs son libres)
        self.constraints['UX'] = False
        self.constraints['UY'] = False
        self.constraints['UZ'] = False
        self.constraints['RX'] = False
        self.constraints['RY'] = False
        self.constraints['RZ'] = False

        # Inicializa valores prescritos (por defecto, cero)
        self.prescribed_values['UX'] = 0.0
        self.prescribed_values['UY'] = 0.0
        self.prescribed_values['UZ'] = 0.0
        self.prescribed_values['RX'] = 0.0
        self.prescribed_values['RY'] = 0.0
        self.prescribed_values['RZ'] = 0.0

    def set_global_index(self, dof_name, global_index):
        """
        Asigna un índice global a un grado de libertad específico.

        Args:
            dof_name (str): Nombre del grado de libertad ('UX', 'UY', etc.).
            global_index (int): Índice en el sistema global de ecuaciones.

        Returns:
            bool: True si la asignación fue exitosa, False en caso contrario.
        """
        if dof_name in self.indices:
            self.indices[dof_name] = global_index
            return True
        return False

    def add_custom_dof(self, dof_name):
        """
        Añade un grado de libertad personalizado (útil para elementos especiales).

        Args:
            dof_name (str): Nombre del nuevo grado de libertad.

        Returns:
            bool: True si se añadió correctamente, False si ya existía.
        """
        if dof_name not in self.indices:
            self.indices[dof_name] = -1
            self.constraints[dof_name] = False
            self.prescribed_values[dof_name] = 0.0
            return True
        return False

    def get_active_dofs(self):
        """
        Obtiene la lista de grados de libertad que están activos (no restringidos).

        Returns:
            list: Lista de nombres de DOFs activos.
        """
        return [dof for dof, constrained in self.constraints.items()
                if not constrained]

    def get_constrained_dofs(self):
        """
        Obtiene la lista de grados de libertad que están restringidos.

        Returns:
            list: Lista de nombres de DOFs restringidos.
        """
        return [dof for dof, constrained in self.constraints.items()
                if constrained]

    def apply_constraint(self, dof_name, value=0.0):
        """
        Aplica una restricción a un grado de libertad específico.

        Args:
            dof_name (str): Nombre del grado de libertad a restringir.
            value (float): Valor prescrito para el grado de libertad.

        Returns:
            bool: True si se aplicó correctamente, False en caso contrario.
        """
        if dof_name in self.constraints:
            self.constraints[dof_name] = True
            self.prescribed_values[dof_name] = value
            return True
        return False

    def release_constraint(self, dof_name):
        """
        Libera una restricción de un grado de libertad específico.

        Args:
            dof_name (str): Nombre del grado de libertad a liberar.

        Returns:
            bool: True si se liberó correctamente, False en caso contrario.
        """
        if dof_name in self.constraints:
            self.constraints[dof_name] = False
            return True
        return False

    def is_constrained(self, dof_name):
        """
        Verifica si un grado de libertad está restringido.

        Args:
            dof_name (str): Nombre del grado de libertad.

        Returns:
            bool: True si está restringido, False si está libre.
        """
        return self.constraints.get(dof_name, False)

    def get_global_indices(self):
        """
        Obtiene todos los índices globales de los grados de libertad activos.

        Returns:
            dict: Diccionario con los nombres de DOFs como claves y sus índices globales como valores.
        """
        return {dof: idx for dof, idx in self.indices.items()
                if not self.constraints.get(dof, False) and idx >= 0}

    def get_prescribed_values(self):
        """
        Obtiene todos los valores prescritos para los grados de libertad restringidos.

        Returns:
            dict: Diccionario con los nombres de DOFs como claves y sus valores prescritos como valores.
        """
        return {dof: val for dof, val in self.prescribed_values.items()
                if self.constraints.get(dof, False)}

    def __str__(self):
        """
        Representación en string del objeto DegreesOfFreedom.

        Returns:
            str: Representación legible de los grados de libertad y sus estados.
        """
        result = f"DOFs for Node {self.node_id} ({self.dimension}D):\n"
        for dof in sorted(self.indices.keys()):
            status = "Restringido" if self.constraints.get(
                dof, False) else "Libre"
            index = self.indices.get(dof, -1)
            value = self.prescribed_values.get(dof, 0.0)

            if self.constraints.get(dof, False):
                result += f"  {dof}: {status}, Valor={value:.6f}\n"
            else:
                result += f"  {dof}: {status}, Índice Global={index}\n"

        return result


class Load:
    """
    Clase base para representar cargas en una estructura.
    Diseñada para ser modularizada y extendida para diferentes tipos de cargas.
    """

    def __init__(self, load_id=None, description=None):
        """
        Constructor de la clase base Load

        Parámetros:
        -----------
        load_id : str o int
            Identificador único de la carga
        description : str
            Descripción textual de la carga
        """
        self.load_id = load_id
        self.description = description

    def get_load_vector(self):
        """
        Método abstracto para obtener el vector de carga.
        Debe ser implementado por las subclases.

        Retorna:
        --------
        numpy.ndarray
            Vector de cargas en el sistema global de coordenadas
        """
        raise NotImplementedError(
            "Método debe ser implementado por las subclases")

    def transform_to_global(self):
        """
        Método abstracto para transformar la carga al sistema global de coordenadas.
        Debe ser implementado por las subclases.
        """
        raise NotImplementedError(
            "Método debe ser implementado por las subclases")

    def combine_with(self, other_load, factor=1.0):
        """
        Método para combinar esta carga con otra.

        Parámetros:
        -----------
        other_load : Load
            Otra instancia de carga para combinar
        factor : float
            Factor de escala para la combinación

        Retorna:
        --------
        Load
            Una nueva instancia de carga que representa la combinación
        """
        raise NotImplementedError(
            "Método debe ser implementado por las subclases")


class PointLoad(Load):
    """
    Clase para representar cargas puntuales aplicadas a nodos.
    """

    def __init__(self, node, force_vector, moment_vector=None, load_id=None, description=None):
        """
        Constructor para una carga puntual

        Parámetros:
        -----------
        node : Node
            Nodo al que se aplica la carga
        force_vector : list o numpy.ndarray
            Vector de fuerza [Fx, Fy, Fz]
        moment_vector : list o numpy.ndarray, opcional
            Vector de momento [Mx, My, Mz]
        load_id : str o int
            Identificador único de la carga
        description : str
            Descripción textual de la carga
        """
        super().__init__(load_id, description)
        self.node = node
        self.force_vector = force_vector
        self.moment_vector = moment_vector if moment_vector is not None else [
            0.0, 0.0, 0.0]

    def get_load_vector(self):
        """
        Obtiene el vector de carga para el sistema global.

        Retorna:
        --------
        numpy.ndarray
            Vector de cargas en el sistema global de coordenadas
        """
        import numpy as np
        return np.array(self.force_vector + self.moment_vector)

    def transform_to_global(self):
        """
        Para cargas puntuales en nodos, generalmente ya están en coordenadas globales.
        Este método está implementado para mantener la consistencia con la interfaz.

        Retorna:
        --------
        PointLoad
            Esta misma instancia, ya que ya está en coordenadas globales
        """
        return self

    def scale(self, factor):
        """
        Escala la carga por un factor.

        Parámetros:
        -----------
        factor : float
            Factor de escala

        Retorna:
        --------
        PointLoad
            Una nueva instancia de carga con las fuerzas escaladas
        """
        import numpy as np
        new_force = np.array(self.force_vector) * factor
        new_moment = np.array(self.moment_vector) * factor
        return PointLoad(self.node, new_force.tolist(), new_moment.tolist(),
                         self.load_id, self.description)

    def combine_with(self, other_load, factor=1.0):
        """
        Combina esta carga con otra carga puntual aplicada al mismo nodo.

        Parámetros:
        -----------
        other_load : PointLoad
            Otra instancia de carga puntual
        factor : float
            Factor para escalar la otra carga

        Retorna:
        --------
        PointLoad
            Una nueva instancia de carga que representa la combinación
        """
        import numpy as np
        if not isinstance(other_load, PointLoad) or self.node != other_load.node:
            raise ValueError(
                "Solo se pueden combinar cargas puntuales en el mismo nodo")

        other_scaled = other_load.scale(factor)
        combined_force = np.array(self.force_vector) + \
            np.array(other_scaled.force_vector)
        combined_moment = np.array(
            self.moment_vector) + np.array(other_scaled.moment_vector)

        return PointLoad(self.node, combined_force.tolist(), combined_moment.tolist(),
                         f"{self.load_id}+{other_load.load_id}", "Carga combinada")


class DistributedLoad(Load):
    """
    Clase para representar cargas distribuidas aplicadas a elementos.
    """

    def __init__(self, element, load_vector, direction, start_magnitude, end_magnitude=None,
                 start_position=0.0, end_position=1.0, load_id=None, description=None):
        """
        Constructor para una carga distribuida

        Parámetros:
        -----------
        element : Element
            Elemento al que se aplica la carga
        load_vector : list o numpy.ndarray
            Vector que define la dirección de la carga [x, y, z]
        direction : str
            Dirección de la carga ('global' o 'local')
        start_magnitude : float
            Magnitud de la carga en el punto inicial
        end_magnitude : float, opcional
            Magnitud de la carga en el punto final (si es None, se asume igual a start_magnitude)
        start_position : float, opcional
            Posición relativa de inicio (0.0 a 1.0)
        end_position : float, opcional
            Posición relativa de fin (0.0 a 1.0)
        load_id : str o int
            Identificador único de la carga
        description : str
            Descripción textual de la carga
        """
        super().__init__(load_id, description)
        self.element = element
        self.load_vector = load_vector
        self.direction = direction.lower()
        self.start_magnitude = start_magnitude
        self.end_magnitude = end_magnitude if end_magnitude is not None else start_magnitude
        self.start_position = start_position
        self.end_position = end_position

    def get_equivalent_nodal_loads(self):
        """
        Calcula las cargas nodales equivalentes para esta carga distribuida.

        Retorna:
        --------
        list of PointLoad
            Lista de cargas puntuales equivalentes en los nodos del elemento
        """
        # Implementación para diferentes tipos de elementos (viga, shell, etc.)
        if hasattr(self.element, 'get_equivalent_nodal_loads'):
            return self.element.get_equivalent_nodal_loads(self)
        else:
            # Implementación básica para una viga simple
            import numpy as np
            L = self.element.length

            # Para carga uniforme en una viga
            if self.start_magnitude == self.end_magnitude and self.start_position == 0.0 and self.end_position == 1.0:
                w = self.start_magnitude

                # Cargas nodales equivalentes para carga uniforme:
                # Para una viga en coordenadas locales, dirección y:
                # [0, wL/2, -wL^2/12, 0, wL/2, wL^2/12]
                if np.argmax(np.abs(self.load_vector)) == 1:  # Carga en dirección y local
                    force1 = [0, w * L / 2, 0]
                    moment1 = [0, 0, -w * L * L / 12]
                    force2 = [0, w * L / 2, 0]
                    moment2 = [0, 0, w * L * L / 12]

                    node1_load = PointLoad(
                        self.element.nodes[0], force1, moment1)
                    node2_load = PointLoad(
                        self.element.nodes[1], force2, moment2)

                    return [node1_load, node2_load]

            # Para implementación más compleja, se requiere integración numérica
            # o fórmulas específicas según el tipo de elemento y distribución de carga
            raise NotImplementedError(
                "Cálculo de cargas nodales equivalentes no implementado para esta distribución")

    def get_load_vector(self):
        """
        Obtiene el vector de carga distribuida.
        Para uso en el análisis FEM, generalmente se convierte a cargas nodales equivalentes.

        Retorna:
        --------
        tuple
            (start_vector, end_vector) con los vectores de carga en el inicio y fin
        """
        import numpy as np
        start_vector = np.array(self.load_vector) * self.start_magnitude
        end_vector = np.array(self.load_vector) * self.end_magnitude
        return (start_vector, end_vector)

    def transform_to_global(self):
        """
        Transforma la carga distribuida de coordenadas locales a globales.

        Retorna:
        --------
        DistributedLoad
            Una nueva instancia con la carga en coordenadas globales
        """
        if self.direction == 'global':
            return self

        # Obtener matriz de transformación del elemento
        if hasattr(self.element, 'get_transformation_matrix'):
            import numpy as np
            T = self.element.get_transformation_matrix()

            # Transformar vector de carga de local a global
            global_load_vector = T.dot(np.array(self.load_vector))

            return DistributedLoad(
                self.element, global_load_vector.tolist(), 'global',
                self.start_magnitude, self.end_magnitude,
                self.start_position, self.end_position,
                self.load_id, self.description
            )
        else:
            raise NotImplementedError(
                "El elemento no proporciona matriz de transformación")

    def combine_with(self, other_load, factor=1.0):
        """
        Combina esta carga con otra carga distribuida en el mismo elemento.
        Nota: Esta es una implementación simplificada que asume cargas en la misma dirección.

        Parámetros:
        -----------
        other_load : DistributedLoad
            Otra instancia de carga distribuida
        factor : float
            Factor para escalar la otra carga

        Retorna:
        --------
        DistributedLoad o list of DistributedLoad
            Carga(s) que representa(n) la combinación
        """
        if not isinstance(other_load, DistributedLoad) or self.element != other_load.element:
            raise ValueError(
                "Solo se pueden combinar cargas distribuidas en el mismo elemento")

        # Caso simple: ambas cargas tienen el mismo rango y dirección
        if (self.start_position == other_load.start_position and
            self.end_position == other_load.end_position and
                self.direction == other_load.direction):

            import numpy as np
            # Si los vectores de carga son colineales
            v1 = np.array(self.load_vector)
            v2 = np.array(other_load.load_vector)
            if np.allclose(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)):
                new_start_mag = self.start_magnitude + other_load.start_magnitude * factor
                new_end_mag = self.end_magnitude + other_load.end_magnitude * factor

                return DistributedLoad(
                    self.element, self.load_vector, self.direction,
                    new_start_mag, new_end_mag,
                    self.start_position, self.end_position,
                    f"{self.load_id}+{other_load.load_id}", "Carga combinada"
                )

        # Para casos más complejos, se podría devolver una lista de cargas
        # o utilizar una representación más avanzada
        raise NotImplementedError(
            "Combinación de cargas distribuidas complejas no implementada")


class TemperatureLoad(Load):
    """
    Clase para representar cargas debidas a cambios de temperatura.
    """

    def __init__(self, element, delta_t, alpha=None, gradient=None, load_id=None, description=None):
        """
        Constructor para una carga térmica

        Parámetros:
        -----------
        element : Element
            Elemento al que se aplica la carga térmica
        delta_t : float
            Cambio de temperatura uniforme
        alpha : float, opcional
            Coeficiente de expansión térmica (si no se proporciona, se toma del material del elemento)
        gradient : tuple, opcional
            Gradiente de temperatura (delta_t_top, delta_t_bottom) para flexión
        load_id : str o int
            Identificador único de la carga
        description : str
            Descripción textual de la carga
        """
        super().__init__(load_id, description)
        self.element = element
        self.delta_t = delta_t
        self.alpha = alpha if alpha is not None else element.material.alpha
        self.gradient = gradient

    def get_equivalent_strain(self):
        """
        Calcula la deformación equivalente debido a cambios de temperatura.

        Retorna:
        --------
        numpy.ndarray
            Vector de deformaciones iniciales
        """
        # Deformación axial por cambio uniforme de temperatura
        axial_strain = self.alpha * self.delta_t

        # Si hay gradiente térmico, calcular curvatura
        curvature = [0.0, 0.0, 0.0]
        if self.gradient:
            h = self.element.section.height  # Altura de la sección
            delta_t_diff = self.gradient[0] - self.gradient[1]
            # Curvatura = alpha * ΔT / h
            curvature[2] = self.alpha * delta_t_diff / \
                h  # Curvatura en el plano xy

        return (axial_strain, curvature)

    def get_load_vector(self):
        """
        Obtiene el vector de carga equivalente para cargas térmicas.
        Implementa la fórmula F_t = K * ε_t donde ε_t es la deformación térmica.

        Retorna:
        --------
        numpy.ndarray
            Vector de cargas nodales equivalentes
        """
        if hasattr(self.element, 'get_thermal_load_vector'):
            return self.element.get_thermal_load_vector(self)
        else:
            # Implementación básica para elemento de barra
            import numpy as np
            E = self.element.material.elastic_modulus
            A = self.element.section.area
            L = self.element.length

            # Fuerza axial por expansión térmica: F = E*A*α*ΔT
            force = E * A * self.alpha * self.delta_t

            # Vector de fuerzas nodales para un elemento de barra
            # [F, 0, 0, -F, 0, 0] (en coordenadas locales)
            load_vector = np.zeros(
                2 * self.element.nodes_per_element * self.element.dof_per_node)
            load_vector[0] = -force  # Fuerza en nodo 1
            load_vector[self.element.dof_per_node] = force  # Fuerza en nodo 2

            return load_vector

    def transform_to_global(self):
        """
        Transforma la carga térmica de coordenadas locales a globales.

        Retorna:
        --------
        TemperatureLoad
            Esta misma instancia (la transformación se aplica al obtener el vector de carga)
        """
        # Para cargas térmicas, la transformación generalmente se realiza
        # cuando se calcula el vector de carga equivalente
        return self


class SelfWeightLoad(Load):
    """
    Clase para representar cargas debido al peso propio de los elementos.
    """

    def __init__(self, elements, direction=[-1.0, 0.0, 0.0], gravity=9.81, load_id=None, description=None):
        """
        Constructor para carga de peso propio

        Parámetros:
        -----------
        elements : list of Element
            Lista de elementos a considerar para el peso propio
        direction : list o numpy.ndarray
            Vector unitario que indica la dirección de la gravedad
        gravity : float
            Aceleración gravitacional (m/s²)
        load_id : str o int
            Identificador único de la carga
        description : str
            Descripción textual de la carga
        """
        super().__init__(load_id, description)
        self.elements = elements if isinstance(elements, list) else [elements]
        self.direction = direction
        self.gravity = gravity

    def get_load_vector(self):
        """
        Calcula el vector de carga para todos los elementos afectados.

        Retorna:
        --------
        dict
            Diccionario con elementos como claves y vectores de carga como valores
        """
        result = {}
        for element in self.elements:
            if hasattr(element, 'get_self_weight_load'):
                result[element] = element.get_self_weight_load(
                    self.direction, self.gravity)
        return result

    def get_distributed_loads(self):
        """
        Convierte el peso propio en cargas distribuidas equivalentes para cada elemento.

        Retorna:
        --------
        list of DistributedLoad
            Lista de cargas distribuidas que representan el peso propio
        """
        distributed_loads = []

        for element in self.elements:
            if hasattr(element, 'material') and hasattr(element, 'section'):
                # Calcular peso por unidad de longitud
                density = element.material.density
                area = element.section.area

                # Magnitud de la carga distribuida: ρ * A * g
                magnitude = density * area * self.gravity

                # Crear carga distribuida
                load = DistributedLoad(
                    element, self.direction, 'global', magnitude, magnitude,
                    0.0, 1.0, f"{self.load_id}_elem{element.id}", f"Peso propio de elemento {element.id}"
                )

                distributed_loads.append(load)

        return distributed_loads


class LoadCase:
    """
    Clase para agrupar cargas relacionadas en un caso de carga.
    """

    def __init__(self, name, loads=None, description=None, factor=1.0):
        """
        Constructor para un caso de carga

        Parámetros:
        -----------
        name : str
            Nombre del caso de carga
        loads : list of Load, opcional
            Lista de cargas que componen este caso
        description : str, opcional
            Descripción del caso de carga
        factor : float
            Factor de escala para todas las cargas en este caso
        """
        self.name = name
        self.loads = loads if loads is not None else []
        self.description = description
        self.factor = factor

    def add_load(self, load):
        """
        Añade una carga al caso de carga.

        Parámetros:
        -----------
        load : Load
            Carga a añadir
        """
        self.loads.append(load)

    def apply_to_model(self, model):
        """
        Aplica todas las cargas de este caso al modelo.

        Parámetros:
        -----------
        model : Model
            Modelo de análisis estructural
        """
        for load in self.loads:
            model.apply_load(load, self.factor)

    def get_global_force_vector(self, model):
        """
        Obtiene el vector de fuerzas global para este caso de carga.

        Parámetros:
        -----------
        model : Model
            Modelo de análisis estructural que contiene la información del sistema

        Retorna:
        --------
        numpy.ndarray
            Vector de fuerzas global para el modelo completo
        """
        import numpy as np
        ndof = model.get_total_dof()
        global_force = np.zeros(ndof)

        for load in self.loads:
            # Procesar según el tipo de carga
            if isinstance(load, PointLoad):
                node_id = model.get_node_id(load.node)
                dof_indices = model.get_node_dof_indices(node_id)

                # Para cada grado de libertad del nodo
                load_vector = load.get_load_vector() * self.factor
                for i, dof_idx in enumerate(dof_indices):
                    if dof_idx >= 0:  # Verifica que no sea un dof restringido
                        global_force[dof_idx] += load_vector[i]

            elif isinstance(load, DistributedLoad):
                # Convertir carga distribuida a cargas nodales equivalentes
                nodal_loads = load.get_equivalent_nodal_loads()
                for nodal_load in nodal_loads:
                    node_id = model.get_node_id(nodal_load.node)
                    dof_indices = model.get_node_dof_indices(node_id)

                    load_vector = nodal_load.get_load_vector() * self.factor
                    for i, dof_idx in enumerate(dof_indices):
                        if dof_idx >= 0:
                            global_force[dof_idx] += load_vector[i]

            elif isinstance(load, TemperatureLoad):
                # Obtener vector de carga térmica
                element_id = model.get_element_id(load.element)
                element_dof_indices = model.get_element_dof_indices(element_id)

                thermal_load = load.get_load_vector() * self.factor
                for i, dof_idx in enumerate(element_dof_indices):
                    if dof_idx >= 0:
                        global_force[dof_idx] += thermal_load[i]

            elif isinstance(load, SelfWeightLoad):
                # Convertir a cargas distribuidas y procesarlas
                distributed_loads = load.get_distributed_loads()
                for dist_load in distributed_loads:
                    nodal_loads = dist_load.get_equivalent_nodal_loads()
                    for nodal_load in nodal_loads:
                        node_id = model.get_node_id(nodal_load.node)
                        dof_indices = model.get_node_dof_indices(node_id)

                        load_vector = nodal_load.get_load_vector() * self.factor
                        for i, dof_idx in enumerate(dof_indices):
                            if dof_idx >= 0:
                                global_force[dof_idx] += load_vector[i]

        return global_force


class LoadCombination:
    """
    Clase para combinar varios casos de carga.
    """

    def __init__(self, name, combinations=None, description=None):
        """
        Constructor para una combinación de cargas

        Parámetros:
        -----------
        name : str
            Nombre de la combinación
        combinations : list of tuple, opcional
            Lista de tuplas (caso_carga, factor) que definen la combinación
        description : str, opcional
            Descripción de la combinación
        """
        self.name = name
        self.combinations = combinations if combinations is not None else []
        self.description = description

    def add_load_case(self, load_case, factor=1.0):
        """
        Añade un caso de carga a la combinación.

        Parámetros:
        -----------
        load_case : LoadCase
            Caso de carga a añadir
        factor : float
            Factor de escala para este caso de carga en la combinación
        """
        self.combinations.append((load_case, factor))

    def get_global_force_vector(self, model):
        """
        Obtiene el vector de fuerzas global para esta combinación de cargas.

        Parámetros:
        -----------
        model : Model
            Modelo de análisis estructural

        Retorna:
        --------
        numpy.ndarray
            Vector de fuerzas global para el modelo completo
        """
        import numpy as np
        ndof = model.get_total_dof()
        global_force = np.zeros(ndof)

        for load_case, factor in self.combinations:
            case_force = load_case.get_global_force_vector(model)
            global_force += case_force * factor

        return global_force


class LoadPattern:
    """
    Clase para representar patrones de carga que pueden ser aplicados a la estructura.
    Un patrón de carga define la distribución espacial de cargas en la estructura,
    pero no su magnitud, que se definirá en los casos de carga.
    """

    def __init__(self, pattern_id, name=None, description=None):
        """
        Constructor para un patrón de carga.

        Parámetros:
        -----------
        pattern_id : str o int
            Identificador único del patrón
        name : str, opcional
            Nombre del patrón
        description : str, opcional
            Descripción detallada del patrón
        """
        self.pattern_id = pattern_id
        self.name = name if name is not None else f"Pattern-{pattern_id}"
        self.description = description
        self.loads = []

    def add_load(self, load):
        """
        Añade una carga al patrón.

        Parámetros:
        -----------
        load : Load
            Carga a añadir al patrón
        """
        self.loads.append(load)

    def remove_load(self, load):
        """
        Elimina una carga del patrón.

        Parámetros:
        -----------
        load : Load
            Carga a eliminar

        Retorna:
        --------
        bool
            True si la carga fue eliminada, False si no se encontró
        """
        if load in self.loads:
            self.loads.remove(load)
            return True
        return False

    def get_loads(self):
        """
        Obtiene todas las cargas definidas en este patrón.

        Retorna:
        --------
        list
            Lista de cargas en el patrón
        """
        return self.loads

    def scale_loads(self, factor):
        """
        Crea un nuevo patrón con todas las cargas escaladas por un factor.

        Parámetros:
        -----------
        factor : float
            Factor de escala a aplicar

        Retorna:
        --------
        LoadPattern
            Nuevo patrón con cargas escaladas
        """
        new_pattern = LoadPattern(f"{self.pattern_id}_scaled",
                                  f"{self.name} (x{factor})",
                                  f"{self.description} - Escalado por {factor}")

        for load in self.loads:
            if hasattr(load, "scale"):
                new_pattern.add_load(load.scale(factor))

        return new_pattern

    def get_global_force_vector(self, model, factor=1.0):
        """
        Obtiene el vector de fuerzas global para este patrón de carga.

        Parámetros:
        -----------
        model : Model
            Modelo de análisis estructural
        factor : float
            Factor de escala a aplicar

        Retorna:
        --------
        numpy.ndarray
            Vector de fuerzas global para el modelo completo
        """
        import numpy as np
        ndof = model.get_total_dof()
        global_force = np.zeros(ndof)

        for load in self.loads:
            # Procesar según el tipo de carga
            if isinstance(load, PointLoad):
                node_id = model.get_node_id(load.node)
                dof_indices = model.get_node_dof_indices(node_id)

                # Para cada grado de libertad del nodo
                load_vector = load.get_load_vector() * factor
                for i, dof_idx in enumerate(dof_indices):
                    if dof_idx >= 0:  # Verifica que no sea un dof restringido
                        global_force[dof_idx] += load_vector[i]

            elif isinstance(load, DistributedLoad):
                # Convertir carga distribuida a cargas nodales equivalentes
                nodal_loads = load.get_equivalent_nodal_loads()
                for nodal_load in nodal_loads:
                    node_id = model.get_node_id(nodal_load.node)
                    dof_indices = model.get_node_dof_indices(node_id)

                    load_vector = nodal_load.get_load_vector() * factor
                    for i, dof_idx in enumerate(dof_indices):
                        if dof_idx >= 0:
                            global_force[dof_idx] += load_vector[i]

            elif isinstance(load, TemperatureLoad):
                # Obtener vector de carga térmica
                element_id = model.get_element_id(load.element)
                element_dof_indices = model.get_element_dof_indices(element_id)

                thermal_load = load.get_load_vector() * factor
                for i, dof_idx in enumerate(element_dof_indices):
                    if dof_idx >= 0:
                        global_force[dof_idx] += thermal_load[i]

            elif isinstance(load, SelfWeightLoad):
                # Convertir a cargas distribuidas y procesarlas
                distributed_loads = load.get_distributed_loads()
                for dist_load in distributed_loads:
                    nodal_loads = dist_load.get_equivalent_nodal_loads()
                    for nodal_load in nodal_loads:
                        node_id = model.get_node_id(nodal_load.node)
                        dof_indices = model.get_node_dof_indices(node_id)

                        load_vector = nodal_load.get_load_vector() * factor
                        for i, dof_idx in enumerate(dof_indices):
                            if dof_idx >= 0:
                                global_force[dof_idx] += load_vector[i]

        return global_force


class LoadCase:
    """
    Clase para representar casos de carga. Un caso de carga es la aplicación de uno
    o más patrones de carga con factores específicos para un análisis particular.
    """

    def __init__(self, case_id, name=None, description=None, analysis_type="STATIC"):
        """
        Constructor para un caso de carga.

        Parámetros:
        -----------
        case_id : str o int
            Identificador único del caso de carga
        name : str, opcional
            Nombre del caso de carga
        description : str, opcional
            Descripción detallada del caso de carga
        analysis_type : str, opcional
            Tipo de análisis asociado a este caso de carga
            (STATIC, MODAL, BUCKLING, etc.)
        """
        self.case_id = case_id
        self.name = name if name is not None else f"Case-{case_id}"
        self.description = description
        self.analysis_type = analysis_type
        self.pattern_factors = {}  # Diccionario de {patrón: factor}

    def add_pattern(self, pattern, factor=1.0):
        """
        Añade un patrón de carga al caso de carga con un factor específico.

        Parámetros:
        -----------
        pattern : LoadPattern
            Patrón de carga a añadir
        factor : float
            Factor de escala para este patrón en este caso de carga
        """
        self.pattern_factors[pattern] = factor

    def remove_pattern(self, pattern):
        """
        Elimina un patrón de carga del caso de carga.

        Parámetros:
        -----------
        pattern : LoadPattern
            Patrón de carga a eliminar

        Retorna:
        --------
        bool
            True si el patrón fue eliminado, False si no se encontró
        """
        if pattern in self.pattern_factors:
            del self.pattern_factors[pattern]
            return True
        return False

    def update_factor(self, pattern, new_factor):
        """
        Actualiza el factor para un patrón de carga existente.

        Parámetros:
        -----------
        pattern : LoadPattern
            Patrón de carga a actualizar
        new_factor : float
            Nuevo factor de escala

        Retorna:
        --------
        bool
            True si el patrón fue actualizado, False si no se encontró
        """
        if pattern in self.pattern_factors:
            self.pattern_factors[pattern] = new_factor
            return True
        return False

    def get_patterns(self):
        """
        Obtiene todos los patrones y sus factores.

        Retorna:
        --------
        dict
            Diccionario de {patrón: factor}
        """
        return self.pattern_factors

    def apply_to_model(self, model):
        """
        Aplica el caso de carga al modelo para análisis.

        Parámetros:
        -----------
        model : Model
            Modelo de análisis estructural
        """
        for pattern, factor in self.pattern_factors.items():
            for load in pattern.get_loads():
                model.apply_load(load, factor)

    def get_global_force_vector(self, model):
        """
        Obtiene el vector de fuerzas global para este caso de carga.

        Parámetros:
        -----------
        model : Model
            Modelo de análisis estructural

        Retorna:
        --------
        numpy.ndarray
            Vector de fuerzas global para el modelo completo
        """
        import numpy as np
        ndof = model.get_total_dof()
        global_force = np.zeros(ndof)

        for pattern, factor in self.pattern_factors.items():
            pattern_force = pattern.get_global_force_vector(model, factor)
            global_force += pattern_force

        return global_force

    def clone(self, new_case_id=None, new_name=None):
        """
        Crea una copia del caso de carga actual.

        Parámetros:
        -----------
        new_case_id : str o int, opcional
            Nuevo ID para el caso clonado
        new_name : str, opcional
            Nuevo nombre para el caso clonado

        Retorna:
        --------
        LoadCase
            Un nuevo caso de carga con los mismos patrones y factores
        """
        case_id = new_case_id if new_case_id is not None else f"{self.case_id}_copy"
        name = new_name if new_name is not None else f"{self.name} (Copy)"

        new_case = LoadCase(
            case_id, name, self.description, self.analysis_type)

        for pattern, factor in self.pattern_factors.items():
            new_case.add_pattern(pattern, factor)

        return new_case


class LoadCombination:
    """
    Clase para definir combinaciones de casos de carga.
    """

    def __init__(self, combo_id, name=None, description=None, design_type=None):
        """
        Constructor para una combinación de cargas.

        Parámetros:
        -----------
        combo_id : str o int
            Identificador único de la combinación
        name : str, opcional
            Nombre de la combinación
        description : str, opcional
            Descripción detallada de la combinación
        design_type : str, opcional
            Tipo de diseño (ULS, SLS, etc.)
        """
        self.combo_id = combo_id
        self.name = name if name is not None else f"Combo-{combo_id}"
        self.description = description
        self.design_type = design_type
        self.case_factors = {}  # Diccionario de {caso: factor}

    def add_case(self, case, factor=1.0):
        """
        Añade un caso de carga a la combinación.

        Parámetros:
        -----------
        case : LoadCase
            Caso de carga a añadir
        factor : float
            Factor de escala para este caso en esta combinación
        """
        self.case_factors[case] = factor

    def remove_case(self, case):
        """
        Elimina un caso de carga de la combinación.

        Parámetros:
        -----------
        case : LoadCase
            Caso de carga a eliminar

        Retorna:
        --------
        bool
            True si el caso fue eliminado, False si no se encontró
        """
        if case in self.case_factors:
            del self.case_factors[case]
            return True
        return False

    def update_factor(self, case, new_factor):
        """
        Actualiza el factor para un caso de carga existente.

        Parámetros:
        -----------
        case : LoadCase
            Caso de carga a actualizar
        new_factor : float
            Nuevo factor de escala

        Retorna:
        --------
        bool
            True si el caso fue actualizado, False si no se encontró
        """
        if case in self.case_factors:
            self.case_factors[case] = new_factor
            return True
        return False

    def get_cases(self):
        """
        Obtiene todos los casos y sus factores.

        Retorna:
        --------
        dict
            Diccionario de {caso: factor}
        """
        return self.case_factors

    def get_global_force_vector(self, model):
        """
        Obtiene el vector de fuerzas global para esta combinación.

        Parámetros:
        -----------
        model : Model
            Modelo de análisis estructural

        Retorna:
        --------
        numpy.ndarray
            Vector de fuerzas global para el modelo completo
        """
        import numpy as np
        ndof = model.get_total_dof()
        global_force = np.zeros(ndof)

        for case, factor in self.case_factors.items():
            case_force = case.get_global_force_vector(model)
            global_force += case_force * factor

        return global_force

    def apply_to_model(self, model):
        """
        Aplica la combinación de cargas al modelo para análisis.

        Parámetros:
        -----------
        model : Model
            Modelo de análisis estructural
        """
        for case, factor in self.case_factors.items():
            for pattern, pattern_factor in case.get_patterns().items():
                for load in pattern.get_loads():
                    model.apply_load(load, factor * pattern_factor)

    @classmethod
    def create_envelope(cls, combo_id, combinations, name=None, description=None):
        """
        Crea una combinación envolvente a partir de varias combinaciones.

        Parámetros:
        -----------
        combo_id : str o int
            Identificador único de la combinación envolvente
        combinations : list
            Lista de combinaciones a incluir en la envolvente
        name : str, opcional
            Nombre de la combinación envolvente
        description : str, opcional
            Descripción detallada de la combinación envolvente

        Retorna:
        --------
        LoadCombination
            Objeto que representa la combinación envolvente
        """
        name = name if name is not None else f"Envelope-{combo_id}"
        description = description if description is not None else "Envolvente de combinaciones"

        envelope = cls(combo_id, name, description, "ENVELOPE")
        envelope.combinations = combinations

        # Agregamos un método especial para obtener la envolvente de resultados
        def get_envelope_results(self, model, result_type="displacement"):
            """
            Obtiene los resultados de la envolvente para un tipo de resultado específico.

            Parámetros:
            -----------
            model : Model
                Modelo de análisis estructural
            result_type : str
                Tipo de resultado ("displacement", "force", "stress", etc.)

            Retorna:
            --------
            tuple
                (max_values, min_values, max_combos, min_combos)
            """
            import numpy as np

            if result_type == "displacement":
                # Inicializar con valores extremos opuestos
                ndof = model.get_total_dof()
                max_values = np.full(ndof, -np.inf)
                min_values = np.full(ndof, np.inf)
                max_combos = np.empty(ndof, dtype=object)
                min_combos = np.empty(ndof, dtype=object)

                # Para cada combinación, calcular desplazamientos y actualizar envolvente
                for combo in self.combinations:
                    force_vector = combo.get_global_force_vector(model)
                    displacements = model.solve_static(force_vector)

                    # Actualizar valores máximos y mínimos
                    for i in range(ndof):
                        if displacements[i] > max_values[i]:
                            max_values[i] = displacements[i]
                            max_combos[i] = combo.name
                        if displacements[i] < min_values[i]:
                            min_values[i] = displacements[i]
                            min_combos[i] = combo.name

                return (max_values, min_values, max_combos, min_combos)

            elif result_type == "force" or result_type == "stress":
                # Similar al anterior, pero para fuerzas internas o tensiones
                # Requiere cálculos adicionales específicos para cada elemento
                raise NotImplementedError(
                    "Envolvente para fuerzas/tensiones no implementada")

            else:
                raise ValueError(
                    f"Tipo de resultado no soportado: {result_type}")

        # Agregar el método a la instancia
        import types
        envelope.get_envelope_results = types.MethodType(
            get_envelope_results, envelope)

        return envelope


class TimeHistory:
    """
    Clase para representar historias temporales de carga para análisis dinámico.
    """

    def __init__(self, history_id, name=None, description=None, time_step=0.01):
        """
        Constructor para una historia temporal de carga.

        Parámetros:
        -----------
        history_id : str o int
            Identificador único de la historia temporal
        name : str, opcional
            Nombre de la historia temporal
        description : str, opcional
            Descripción detallada de la historia temporal
        time_step : float
            Paso de tiempo para la historia temporal
        """
        self.history_id = history_id
        self.name = name if name is not None else f"History-{history_id}"
        self.description = description
        self.time_step = time_step
        self.time_values = []
        self.factor_values = []

    def add_point(self, time, factor):
        """
        Añade un punto a la historia temporal.

        Parámetros:
        -----------
        time : float
            Tiempo en segundos
        factor : float
            Factor de carga en ese instante
        """
        # Insertar en orden cronológico
        index = 0
        while index < len(self.time_values) and self.time_values[index] < time:
            index += 1

        self.time_values.insert(index, time)
        self.factor_values.insert(index, factor)

    def remove_point(self, time):
        """
        Elimina un punto de la historia temporal.

        Parámetros:
        -----------
        time : float
            Tiempo del punto a eliminar

        Retorna:
        --------
        bool
            True si el punto fue eliminado, False si no se encontró
        """
        try:
            index = self.time_values.index(time)
            self.time_values.pop(index)
            self.factor_values.pop(index)
            return True
        except ValueError:
            return False

    def get_factor_at_time(self, time):
        """
        Obtiene el factor de carga en un instante dado, interpolando si es necesario.

        Parámetros:
        -----------
        time : float
            Tiempo en segundos

        Retorna:
        --------
        float
            Factor de carga interpolado
        """
        if not self.time_values:
            return 0.0

        if time <= self.time_values[0]:
            return self.factor_values[0]

        if time >= self.time_values[-1]:
            return self.factor_values[-1]

        # Buscar los puntos de interpolación
        index = 0
        while index < len(self.time_values) and self.time_values[index] < time:
            index += 1

        # Interpolar linealmente
        t1, t2 = self.time_values[index-1], self.time_values[index]
        f1, f2 = self.factor_values[index-1], self.factor_values[index]

        return f1 + (f2 - f1) * (time - t1) / (t2 - t1)

    def get_time_series(self):
        """
        Obtiene la serie de tiempo completa.

        Retorna:
        --------
        tuple
            (time_values, factor_values)
        """
        return (self.time_values, self.factor_values)

    @classmethod
    def from_function(cls, history_id, func, start_time, end_time, time_step, name=None, description=None):
        """
        Crea una historia temporal a partir de una función.

        Parámetros:
        -----------
        history_id : str o int
            Identificador único de la historia temporal
        func : callable
            Función que toma un tiempo y devuelve un factor
        start_time : float
            Tiempo de inicio
        end_time : float
            Tiempo de fin
        time_step : float
            Paso de tiempo
        name : str, opcional
            Nombre de la historia temporal
        description : str, opcional
            Descripción detallada de la historia temporal

        Retorna:
        --------
        TimeHistory
            Objeto que representa la historia temporal
        """
        history = cls(history_id, name, description, time_step)

        import numpy as np
        times = np.arange(start_time, end_time + time_step/2, time_step)

        for t in times:
            history.add_point(t, func(t))

        return history

    @classmethod
    def from_accelerogram(cls, history_id, file_path, scale_factor=1.0, name=None, description=None):
        """
        Crea una historia temporal a partir de un archivo de acelerograma.

        Parámetros:
        -----------
        history_id : str o int
            Identificador único de la historia temporal
        file_path : str
            Ruta al archivo del acelerograma
        scale_factor : float
            Factor de escala para el acelerograma
        name : str, opcional
            Nombre de la historia temporal
        description : str, opcional
            Descripción detallada de la historia temporal

        Retorna:
        --------
        TimeHistory
            Objeto que representa la historia temporal
        """
        import numpy as np

        # Leer archivo (asume formato simple: tiempo, aceleración)
        data = np.loadtxt(file_path, delimiter=',')

        # Determinar paso de tiempo
        time_step = data[1, 0] - data[0, 0] if len(data) > 1 else 0.01

        history = cls(history_id, name, description, time_step)

        for row in data:
            history.add_point(row[0], row[1] * scale_factor)

        return history


class DynamicLoadCase(LoadCase):
    """
    Clase para representar casos de carga dinámica.
    Hereda de LoadCase y añade funcionalidad específica para dinámica.
    """

    def __init__(self, case_id, name=None, description=None, analysis_type="TIME_HISTORY"):
        """
        Constructor para un caso de carga dinámica.

        Parámetros:
        -----------
        case_id : str o int
            Identificador único del caso de carga
        name : str, opcional
            Nombre del caso de carga
        description : str, opcional
            Descripción detallada del caso de carga
        analysis_type : str, opcional
            Tipo de análisis dinámico (TIME_HISTORY, MODAL, RESPONSE_SPECTRUM)
        """
        super().__init__(case_id, name, description, analysis_type)
        self.time_histories = {}  # Diccionario de {patrón: historia_temporal}
        self.damping_ratio = 0.05  # Amortiguamiento por defecto (5%)
        self.integration_params = {
            "method": "Newmark",
            "gamma": 0.5,
            "beta": 0.25,
            "dt": 0.01
        }

    def add_time_history(self, pattern, time_history):
        """
        Asocia una historia temporal a un patrón de carga.

        Parámetros:
        -----------
        pattern : LoadPattern
            Patrón de carga
        time_history : TimeHistory
            Historia temporal a asociar
        """
        self.time_histories[pattern] = time_history

        # Asegurarse de que el patrón está en el caso de carga
        if pattern not in self.pattern_factors:
            self.add_pattern(pattern, 1.0)

    def set_damping(self, damping_ratio):
        """
        Establece el amortiguamiento para el análisis dinámico.

        Parámetros:
        -----------
        damping_ratio : float
            Ratio de amortiguamiento (0.0 a 1.0)
        """
        if 0.0 <= damping_ratio <= 1.0:
            self.damping_ratio = damping_ratio
        else:
            raise ValueError(
                "El ratio de amortiguamiento debe estar entre 0.0 y 1.0")

    def set_integration_method(self, method, **params):
        """
        Establece el método de integración para análisis de historia temporal.

        Parámetros:
        -----------
        method : str
            Método de integración ("Newmark", "Wilson", "Central-Difference")
        params : dict
            Parámetros específicos del método
        """
        self.integration_params["method"] = method

        # Actualizar parámetros proporcionados
        for key, value in params.items():
            self.integration_params[key] = value

        # Establecer valores predeterminados según el método
        if method == "Newmark" and "gamma" not in params and "beta" not in params:
            self.integration_params["gamma"] = 0.5  # Sin disipación numérica
            # Método de aceleración promedio
            self.integration_params["beta"] = 0.25

        elif method == "Wilson" and "theta" not in params:
            # Estable para theta >= 1.37
            self.integration_params["theta"] = 1.4

        elif method == "Central-Difference":
            pass  # No requiere parámetros adicionales

    def get_load_at_time(self, time, model):
        """
        Obtiene el vector de carga en un instante específico.

        Parámetros:
        -----------
        time : float
            Tiempo en segundos
        model : Model
            Modelo de análisis estructural

        Retorna:
        --------
        numpy.ndarray
            Vector de fuerzas global para el instante dado
        """
        import numpy as np
        ndof = model.get_total_dof()
        global_force = np.zeros(ndof)

        for pattern, time_history in self.time_histories.items():
            factor = time_history.get_factor_at_time(time)
            pattern_factor = self.pattern_factors.get(pattern, 1.0)

            pattern_force = pattern.get_global_force_vector(
                model, factor * pattern_factor)
            global_force += pattern_force

        return global_force

    def get_time_series_for_analysis(self, model, start_time, end_time):
        """
        Genera la serie de tiempo para análisis dinámico.

        Parámetros:
        -----------
        model : Model
            Modelo de análisis estructural
        start_time : float
            Tiempo de inicio
        end_time : float
            Tiempo de fin

        Retorna:
        --------
        tuple
            (time_points, force_vectors)
        """
        import numpy as np

        # Determinar el paso de tiempo más pequeño entre todas las historias temporales
        dt = min([th.time_step for th in self.time_histories.values()])

        # Generar puntos de tiempo
        time_points = np.arange(start_time, end_time + dt/2, dt)

        # Inicializar matriz de fuerzas
        ndof = model.get_total_dof()
        force_vectors = np.zeros((len(time_points), ndof))

        # Calcular fuerzas para cada instante
        for i, t in enumerate(time_points):
            force_vectors[i] = self.get_load_at_time(t, model)

        return (time_points, force_vectors)


class LoadCombination:
    """
    Clase para definir combinaciones de casos de carga.
    Las combinaciones de carga permiten aplicar múltiples casos de carga
    con diferentes factores para realizar análisis de diseño según
    diferentes normativas (ACI, AISC, Eurocódigo, etc.)
    """

    def __init__(self, combo_id, name=None, description=None, design_type=None):
        """
        Constructor para una combinación de cargas.

        Parámetros:
        -----------
        combo_id : str o int
            Identificador único de la combinación
        name : str, opcional
            Nombre de la combinación
        description : str, opcional
            Descripción detallada de la combinación
        design_type : str, opcional
            Tipo de diseño (ULS, SLS, ASD, LRFD, etc.)
        """
        self.combo_id = combo_id
        self.name = name if name is not None else f"Combo-{combo_id}"
        self.description = description
        self.design_type = design_type
        self.case_factors = {}  # Diccionario de {caso: factor}

    def add_case(self, case, factor=1.0):
        """
        Añade un caso de carga a la combinación con un factor específico.

        Parámetros:
        -----------
        case : LoadCase
            Caso de carga a añadir a la combinación
        factor : float
            Factor a aplicar al caso de carga

        Retorna:
        --------
        self : LoadCombination
            Retorna la instancia para permitir encadenamiento de métodos
        """
        if case.case_id in [c.case_id for c in self.case_factors.keys()]:
            raise ValueError(
                f"El caso de carga con ID {case.case_id} ya existe en esta combinación")

        self.case_factors[case] = factor
        return self

    def remove_case(self, case):
        """
        Elimina un caso de carga de la combinación.

        Parámetros:
        -----------
        case : LoadCase
            Caso de carga a eliminar

        Retorna:
        --------
        bool
            True si el caso fue eliminado, False si no se encontró
        """
        if case in self.case_factors:
            del self.case_factors[case]
            return True

        # Intentar buscar por ID si no se encontró la instancia
        for c in list(self.case_factors.keys()):
            if c.case_id == case.case_id:
                del self.case_factors[c]
                return True

        return False

    def update_factor(self, case, new_factor):
        """
        Actualiza el factor de un caso de carga en la combinación.

        Parámetros:
        -----------
        case : LoadCase
            Caso de carga a actualizar
        new_factor : float
            Nuevo factor a aplicar

        Retorna:
        --------
        bool
            True si el factor fue actualizado, False si el caso no se encontró
        """
        if case in self.case_factors:
            self.case_factors[case] = new_factor
            return True

        # Intentar buscar por ID si no se encontró la instancia
        for c in self.case_factors:
            if c.case_id == case.case_id:
                self.case_factors[c] = new_factor
                return True

        return False

    def get_cases(self):
        """
        Obtiene todos los casos de carga de la combinación.

        Retorna:
        --------
        list of tuple
            Lista de tuplas (caso, factor) en la combinación
        """
        return [(case, factor) for case, factor in self.case_factors.items()]

    def get_global_force_vector(self, model):
        """
        Calcula el vector de fuerzas globales para esta combinación.

        Parámetros:
        -----------
        model : Model
            Modelo estructural al que se aplicará la combinación

        Retorna:
        --------
        numpy.ndarray
            Vector de fuerzas globales que representa la combinación
        """
        import numpy as np

        # Inicializar vector de fuerzas con ceros
        dof_count = model.get_dof_count()
        global_force = np.zeros(dof_count)

        # Sumar contribuciones de cada caso con su factor
        for case, factor in self.case_factors.items():
            case_force = case.get_global_force_vector(model)
            global_force += factor * case_force

        return global_force

    def apply_to_model(self, model):
        """
        Aplica la combinación de cargas al modelo estructural.

        Parámetros:
        -----------
        model : Model
            Modelo estructural al que se aplicará la combinación

        Retorna:
        --------
        bool
            True si la aplicación fue exitosa
        """
        # Obtener el vector de fuerzas global
        global_force = self.get_global_force_vector(model)

        # Aplicar el vector de fuerzas al modelo
        model.set_load_vector(global_force, load_case_name=self.name)

        return True

    def get_results(self, model, analysis_type="static"):
        """
        Obtiene los resultados del análisis para esta combinación.

        Parámetros:
        -----------
        model : Model
            Modelo estructural analizado
        analysis_type : str
            Tipo de análisis ('static', 'modal', etc.)

        Retorna:
        --------
        dict
            Diccionario con los resultados del análisis
        """
        # Aplicar la combinación al modelo
        self.apply_to_model(model)

        # Realizar el análisis
        if analysis_type.lower() == "static":
            results = model.analyze_static()
        elif analysis_type.lower() == "modal":
            results = model.analyze_modal()
        else:
            raise ValueError(
                f"Tipo de análisis '{analysis_type}' no soportado")

        return results

    def clone(self, new_combo_id=None, new_name=None):
        """
        Crea una copia de esta combinación de cargas.

        Parámetros:
        -----------
        new_combo_id : str o int, opcional
            Nuevo ID para la combinación clonada
        new_name : str, opcional
            Nuevo nombre para la combinación clonada

        Retorna:
        --------
        LoadCombination
            Nueva instancia de combinación con los mismos casos y factores
        """
        combo_id = new_combo_id if new_combo_id is not None else f"{self.combo_id}_copy"
        name = new_name if new_name is not None else f"{self.name} (copy)"

        # Crear nueva combinación
        new_combo = LoadCombination(
            combo_id, name, self.description, self.design_type)

        # Copiar todos los casos con sus factores
        for case, factor in self.case_factors.items():
            new_combo.add_case(case, factor)

        return new_combo

    @classmethod
    def create_envelope(cls, combo_id, combinations, envelope_type="MAX", name=None, description=None):
        """
        Crea una combinación tipo envolvente a partir de otras combinaciones.

        Parámetros:
        -----------
        combo_id : str o int
            ID para la nueva combinación envolvente
        combinations : list of LoadCombination
            Lista de combinaciones a incluir en la envolvente
        envelope_type : str
            Tipo de envolvente: "MAX", "MIN", "ABS_MAX"
        name : str, opcional
            Nombre para la combinación envolvente
        description : str, opcional
            Descripción para la combinación envolvente

        Retorna:
        --------
        EnvelopeCombination
            Nueva instancia de combinación tipo envolvente
        """
        from enum import Enum

        class EnvelopeType(Enum):
            MAX = "MAX"
            MIN = "MIN"
            ABS_MAX = "ABS_MAX"

        # Validar el tipo de envolvente
        if envelope_type not in [e.value for e in EnvelopeType]:
            raise ValueError(f"Tipo de envolvente '{envelope_type}' no válido")

        # Crear instancia de combinación envolvente
        if name is None:
            name = f"Envelope-{envelope_type}-{combo_id}"

        if description is None:
            description = f"Envolvente {envelope_type} de {len(combinations)} combinaciones"

        return EnvelopeCombination(combo_id, combinations, envelope_type, name, description)

    @classmethod
    def from_standard(cls, standard, combo_id, load_cases, design_type=None, name=None, description=None):
        """
        Crea combinaciones de carga según una normativa específica.

        Parámetros:
        -----------
        standard : str
            Normativa a utilizar ('ACI318', 'AISC360', 'EC0', etc.)
        combo_id : str o int
            Prefijo para los IDs de las combinaciones
        load_cases : dict
            Diccionario de casos de carga clasificados por tipo
            (e.j.: {'D': dead_case, 'L': live_case, 'W': wind_case})
        design_type : str, opcional
            Tipo de diseño específico dentro de la normativa
        name : str, opcional
            Prefijo para los nombres de las combinaciones
        description : str, opcional
            Descripción base para las combinaciones

        Retorna:
        --------
        list of LoadCombination
            Lista de combinaciones generadas según la normativa
        """
        # Implementación básica para ACI 318-19
        if standard.upper() == "ACI318":
            return cls._create_aci318_combinations(combo_id, load_cases, design_type, name, description)
        # Implementación básica para AISC 360-16
        elif standard.upper() == "AISC360":
            return cls._create_aisc360_combinations(combo_id, load_cases, design_type, name, description)
        # Implementación básica para Eurocódigo 0
        elif standard.upper() == "EC0":
            return cls._create_ec0_combinations(combo_id, load_cases, design_type, name, description)
        else:
            raise ValueError(f"Normativa '{standard}' no implementada")

    @staticmethod
    def _create_aci318_combinations(combo_id, load_cases, design_type=None, name=None, description=None):
        """
        Crea combinaciones según ACI 318-19.

        Método privado para crear combinaciones según ACI 318-19.
        """
        combinations = []

        # Validar casos de carga necesarios
        required_cases = ['D']
        for case_type in required_cases:
            if case_type not in load_cases:
                raise ValueError(
                    f"Caso de carga '{case_type}' requerido para combinaciones ACI 318")

        # Combinación 1: 1.4D
        combo1 = LoadCombination(f"{combo_id}_1",
                                 name=f"{name}_1.4D" if name else "1.4D",
                                 description=f"{description} - Combinación 1.4D" if description else "ACI 318-19 Combinación 1.4D",
                                 design_type=design_type)
        combo1.add_case(load_cases['D'], 1.4)
        combinations.append(combo1)

        # Combinación 2: 1.2D + 1.6L + 0.5(Lr o S o R)
        if 'L' in load_cases:
            combo2 = LoadCombination(f"{combo_id}_2",
                                     name=f"{name}_1.2D+1.6L" if name else "1.2D+1.6L",
                                     description=f"{description} - Combinación 1.2D+1.6L" if description else "ACI 318-19 Combinación 1.2D+1.6L",
                                     design_type=design_type)
            combo2.add_case(load_cases['D'], 1.2)
            combo2.add_case(load_cases['L'], 1.6)

            # Añadir carga de techo si existe
            if 'Lr' in load_cases:
                combo2.add_case(load_cases['Lr'], 0.5)
            elif 'S' in load_cases:
                combo2.add_case(load_cases['S'], 0.5)
            elif 'R' in load_cases:
                combo2.add_case(load_cases['R'], 0.5)

            combinations.append(combo2)

        # Continuar con otras combinaciones según ACI 318-19...

        return combinations

    # Métodos similares para otras normativas
    @staticmethod
    def _create_aisc360_combinations(combo_id, load_cases, design_type=None, name=None, description=None):
        """Crea combinaciones según AISC 360-16."""
        # Implementación similar a ACI 318
        pass

    @staticmethod
    def _create_ec0_combinations(combo_id, load_cases, design_type=None, name=None, description=None):
        """Crea combinaciones según Eurocódigo 0."""
        # Implementación similar a ACI 318
        pass


class EnvelopeCombination(LoadCombination):
    """
    Clase para representar combinaciones tipo envolvente.
    Hereda de LoadCombination pero redefine el comportamiento para
    obtener los valores máximos, mínimos o máximos absolutos de un conjunto de combinaciones.
    """

    def __init__(self, combo_id, combinations, envelope_type="MAX", name=None, description=None):
        """
        Constructor para una combinación tipo envolvente.

        Parámetros:
        -----------
        combo_id : str o int
            Identificador único de la combinación envolvente
        combinations : list of LoadCombination
            Lista de combinaciones a incluir en la envolvente
        envelope_type : str
            Tipo de envolvente: "MAX", "MIN", "ABS_MAX"
        name : str, opcional
            Nombre de la combinación envolvente
        description : str, opcional
            Descripción detallada de la combinación envolvente
        """
        super().__init__(combo_id, name, description)
        self.combinations = combinations
        self.envelope_type = envelope_type

    def get_global_force_vector(self, model):
        """
        Obtiene el vector de fuerzas global según el tipo de envolvente.

        Parámetros:
        -----------
        model : Model
            Modelo estructural para aplicar las combinaciones

        Retorna:
        --------
        numpy.ndarray
            Vector de fuerzas global según el tipo de envolvente
        """
        import numpy as np

        # Obtener dimensión del vector de fuerzas
        dof_count = model.get_dof_count()

        # Inicializar arrays para almacenar resultados de todas las combinaciones
        all_forces = np.zeros((len(self.combinations), dof_count))

        # Calcular fuerzas para cada combinación
        for i, combo in enumerate(self.combinations):
            all_forces[i, :] = combo.get_global_force_vector(model)

        # Aplicar el tipo de envolvente
        if self.envelope_type == "MAX":
            return np.max(all_forces, axis=0)
        elif self.envelope_type == "MIN":
            return np.min(all_forces, axis=0)
        elif self.envelope_type == "ABS_MAX":
            # Para cada DOF, buscar el valor con mayor magnitud
            abs_forces = np.abs(all_forces)
            max_abs_indices = np.argmax(abs_forces, axis=0)

            # Construir el vector resultante usando los valores originales
            result = np.zeros(dof_count)
            for i in range(dof_count):
                result[i] = all_forces[max_abs_indices[i], i]

            return result
        else:
            raise ValueError(
                f"Tipo de envolvente '{self.envelope_type}' no válido")

    def get_envelope_results(self, model, result_type="displacement"):
        """
        Obtiene los resultados envolventes para un tipo específico.

        Parámetros:
        -----------
        model : Model
            Modelo estructural analizado
        result_type : str
            Tipo de resultado ('displacement', 'reaction', 'internal_force', etc.)

        Retorna:
        --------
        dict
            Resultados envolventes según el tipo especificado
        """
        import numpy as np

        # Inicializar diccionarios para almacenar resultados
        results = {}
        max_values = {}
        min_values = {}
        max_combos = {}
        min_combos = {}

        # Obtener resultados para cada combinación
        for combo in self.combinations:
            # Aplicar la combinación y analizar
            combo_results = combo.get_results(model)

            # Procesar según el tipo de resultado
            if result_type == "displacement":
                displacements = combo_results.get("displacements", {})

                for node_id, node_disp in displacements.items():
                    if node_id not in max_values:
                        max_values[node_id] = {
                            dof: float('-inf') for dof in node_disp}
                        min_values[node_id] = {
                            dof: float('inf') for dof in node_disp}
                        max_combos[node_id] = {dof: None for dof in node_disp}
                        min_combos[node_id] = {dof: None for dof in node_disp}

                    for dof, value in node_disp.items():
                        if value > max_values[node_id][dof]:
                            max_values[node_id][dof] = value
                            max_combos[node_id][dof] = combo.name
                        if value < min_values[node_id][dof]:
                            min_values[node_id][dof] = value
                            min_combos[node_id][dof] = combo.name

            elif result_type == "internal_force":
                element_forces = combo_results.get("element_forces", {})

                for elem_id, elem_force in element_forces.items():
                    if elem_id not in max_values:
                        max_values[elem_id] = {force_type: float(
                            '-inf') for force_type in elem_force}
                        min_values[elem_id] = {force_type: float(
                            'inf') for force_type in elem_force}
                        max_combos[elem_id] = {
                            force_type: None for force_type in elem_force}
                        min_combos[elem_id] = {
                            force_type: None for force_type in elem_force}

                    for force_type, value in elem_force.items():
                        if isinstance(value, (list, np.ndarray)):
                            # Para fuerzas que varían a lo largo del elemento
                            if elem_id not in max_values:
                                max_values[elem_id] = {
                                    force_type: np.full_like(value, float('-inf'))}
                                min_values[elem_id] = {
                                    force_type: np.full_like(value, float('inf'))}
                                max_combos[elem_id] = {
                                    force_type: [None] * len(value)}
                                min_combos[elem_id] = {
                                    force_type: [None] * len(value)}

                            for i, val in enumerate(value):
                                if val > max_values[elem_id][force_type][i]:
                                    max_values[elem_id][force_type][i] = val
                                    max_combos[elem_id][force_type][i] = combo.name
                                if val < min_values[elem_id][force_type][i]:
                                    min_values[elem_id][force_type][i] = val
                                    min_combos[elem_id][force_type][i] = combo.name
                        else:
                            # Para valores simples
                            if value > max_values[elem_id][force_type]:
                                max_values[elem_id][force_type] = value
                                max_combos[elem_id][force_type] = combo.name
                            if value < min_values[elem_id][force_type]:
                                min_values[elem_id][force_type] = value
                                min_combos[elem_id][force_type] = combo.name

        # Construir resultados según el tipo de envolvente
        if self.envelope_type == "MAX":
            results = {
                "values": max_values,
                "combinations": max_combos
            }
        elif self.envelope_type == "MIN":
            results = {
                "values": min_values,
                "combinations": min_combos
            }
        elif self.envelope_type == "ABS_MAX":
            # Para cada elemento y cada componente, seleccionar el valor con mayor magnitud
            abs_max_values = {}
            abs_max_combos = {}

            for entity_id in max_values:
                abs_max_values[entity_id] = {}
                abs_max_combos[entity_id] = {}

                for component in max_values[entity_id]:
                    max_val = max_values[entity_id][component]
                    min_val = min_values[entity_id][component]

                    if isinstance(max_val, (list, np.ndarray)):
                        # Para arrays
                        abs_max = np.zeros_like(max_val)
                        combo_list = [None] * len(max_val)

                        for i in range(len(max_val)):
                            if abs(max_val[i]) >= abs(min_val[i]):
                                abs_max[i] = max_val[i]
                                combo_list[i] = max_combos[entity_id][component][i]
                            else:
                                abs_max[i] = min_val[i]
                                combo_list[i] = min_combos[entity_id][component][i]

                        abs_max_values[entity_id][component] = abs_max
                        abs_max_combos[entity_id][component] = combo_list
                    else:
                        # Para valores simples
                        if abs(max_val) >= abs(min_val):
                            abs_max_values[entity_id][component] = max_val
                            abs_max_combos[entity_id][component] = max_combos[entity_id][component]
                        else:
                            abs_max_values[entity_id][component] = min_val
                            abs_max_combos[entity_id][component] = min_combos[entity_id][component]

            results = {
                "values": abs_max_values,
                "combinations": abs_max_combos
            }

        return results


class Node:
    """
    Clase que representa un nodo en un análisis estructural.

    Un nodo en análisis estructural es un punto donde se concentran 
    los grados de libertad y donde se pueden aplicar cargas externas.
    """

    def __init__(self, id, x, y, z=0.0, dimension=None):
        """
        Inicializa un nodo con sus coordenadas y crea sus grados de libertad.

        Args:
            id (int): Identificador único del nodo.
            x (float): Coordenada en dirección X.
            y (float): Coordenada en dirección Y.
            z (float): Coordenada en dirección Z (por defecto 0.0 para casos 2D).
            dimension (int, opcional): Dimensión del problema (2 o 3). Si no se especifica,
                                     se determina automáticamente basado en las coordenadas.
        """
        self.id = id
        # Crear un vértice para manejar la geometría
        self.vertex = Vertex(id, x, y, z)

        # Determinar la dimensión si no se especifica
        if dimension is None:
            self.dimension = self.vertex.get_dimension()
        else:
            self.dimension = dimension

        # Crear el objeto de grados de libertad
        self.dofs = DegreesOfFreedom(self.id, self.dimension)

        # Diccionario para almacenar cargas aplicadas al nodo
        self.loads = {}

        # Inicializar cargas por defecto según la dimensión
        self._init_loads()

        # Resultados calculados (desplazamientos, reacciones, etc.)
        self.results = {}

    def _init_loads(self):
        """Inicializa las cargas según la dimensión del problema."""
        if self.dimension == 2:
            self.loads['FX'] = 0.0  # Fuerza en dirección X
            self.loads['FY'] = 0.0  # Fuerza en dirección Y
            self.loads['MZ'] = 0.0  # Momento alrededor del eje Z
        else:  # 3D
            self.loads['FX'] = 0.0  # Fuerza en dirección X
            self.loads['FY'] = 0.0  # Fuerza en dirección Y
            self.loads['FZ'] = 0.0  # Fuerza en dirección Z
            self.loads['MX'] = 0.0  # Momento alrededor del eje X
            self.loads['MY'] = 0.0  # Momento alrededor del eje Y
            self.loads['MZ'] = 0.0  # Momento alrededor del eje Z

    def set_coordinates(self, x, y, z=None):
        """
        Actualiza las coordenadas del nodo.

        Args:
            x (float): Nueva coordenada en dirección X.
            y (float): Nueva coordenada en dirección Y.
            z (float, opcional): Nueva coordenada en dirección Z.
        """
        old_dim = self.dimension
        self.vertex.set_coordinates(x, y, z)

        # Verificar si cambió la dimensión
        new_dim = self.vertex.get_dimension()
        if new_dim != old_dim:
            self.dimension = new_dim
            # Recrear los grados de libertad si cambia la dimensión
            self.dofs = DegreesOfFreedom(self.id, self.dimension)
            self._init_loads()

    def get_coordinates(self):
        """
        Obtiene las coordenadas del nodo.

        Returns:
            tuple: Tupla con las coordenadas (x, y, z).
        """
        return self.vertex.get_coordinates()

    def apply_load(self, load_type, value):
        """
        Aplica una carga al nodo.

        Args:
            load_type (str): Tipo de carga ('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ').
            value (float): Valor de la carga.

        Returns:
            bool: True si se aplicó correctamente, False en caso contrario.
        """
        if load_type in self.loads:
            self.loads[load_type] += value  # Acumular la carga
            return True
        return False

    def set_load(self, load_type, value):
        """
        Establece una carga específica en el nodo (sobrescribe el valor actual).

        Args:
            load_type (str): Tipo de carga ('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ').
            value (float): Valor de la carga.

        Returns:
            bool: True si se estableció correctamente, False en caso contrario.
        """
        if load_type in self.loads:
            self.loads[load_type] = value  # Establecer la carga
            return True
        return False

    def clear_loads(self):
        """Elimina todas las cargas aplicadas al nodo."""
        for load_type in self.loads:
            self.loads[load_type] = 0.0

    def apply_constraint(self, dof_name, value=0.0):
        """
        Aplica una restricción a un grado de libertad específico.

        Args:
            dof_name (str): Nombre del grado de libertad a restringir.
            value (float): Valor prescrito para el grado de libertad.

        Returns:
            bool: True si se aplicó correctamente, False en caso contrario.
        """
        return self.dofs.apply_constraint(dof_name, value)

    def release_constraint(self, dof_name):
        """
        Libera una restricción de un grado de libertad específico.

        Args:
            dof_name (str): Nombre del grado de libertad a liberar.

        Returns:
            bool: True si se liberó correctamente, False en caso contrario.
        """
        return self.dofs.release_constraint(dof_name)

    def fix_all_dofs(self):
        """
        Fija todos los grados de libertad del nodo (empotramiento).

        Returns:
            bool: True si se aplicaron todas las restricciones correctamente.
        """
        result = True
        for dof_name in self.dofs.indices.keys():
            result = result and self.dofs.apply_constraint(dof_name, 0.0)
        return result

    def set_result(self, result_type, value):
        """
        Establece un resultado calculado para el nodo.

        Args:
            result_type (str): Tipo de resultado ('UX', 'UY', 'RZ', 'RX', etc.).
            value (float): Valor del resultado.
        """
        self.results[result_type] = value

    def get_result(self, result_type):
        """
        Obtiene un resultado calculado para el nodo.

        Args:
            result_type (str): Tipo de resultado ('UX', 'UY', 'RZ', 'RX', etc.).

        Returns:
            float: Valor del resultado o None si no existe.
        """
        return self.results.get(result_type, None)

    def get_displacement_vector(self):
        """
        Obtiene el vector de desplazamientos del nodo.

        Returns:
            list: Lista con los valores de desplazamiento según la dimensión.
        """
        if self.dimension == 2:
            return [
                self.results.get('UX', 0.0),
                self.results.get('UY', 0.0),
                self.results.get('RZ', 0.0)
            ]
        else:  # 3D
            return [
                self.results.get('UX', 0.0),
                self.results.get('UY', 0.0),
                self.results.get('UZ', 0.0),
                self.results.get('RX', 0.0),
                self.results.get('RY', 0.0),
                self.results.get('RZ', 0.0)
            ]

    def get_reaction_vector(self):
        """
        Obtiene el vector de reacciones del nodo.

        Returns:
            list: Lista con los valores de reacción según la dimensión.
        """
        if self.dimension == 2:
            return [
                self.results.get('FX_reaction', 0.0),
                self.results.get('FY_reaction', 0.0),
                self.results.get('MZ_reaction', 0.0)
            ]
        else:  # 3D
            return [
                self.results.get('FX_reaction', 0.0),
                self.results.get('FY_reaction', 0.0),
                self.results.get('FZ_reaction', 0.0),
                self.results.get('MX_reaction', 0.0),
                self.results.get('MY_reaction', 0.0),
                self.results.get('MZ_reaction', 0.0)
            ]

    def distance_to(self, other_node):
        """
        Calcula la distancia euclidiana entre este nodo y otro.

        Args:
            other_node (Node): Otro nodo para calcular la distancia.

        Returns:
            float: Distancia euclidiana entre los dos nodos.
        """
        return self.vertex.distance_to(other_node.vertex)

    def get_deformed_coordinates(self, scale_factor=1.0):
        """
        Obtiene las coordenadas del nodo después de la deformación.

        Args:
            scale_factor (float): Factor de escala para los desplazamientos.

        Returns:
            tuple: Tupla con las coordenadas deformadas (x, y, z).
        """
        x, y, z = self.get_coordinates()
        ux = self.results.get('UX', 0.0) * scale_factor
        uy = self.results.get('UY', 0.0) * scale_factor
        uz = 0.0

        if self.dimension == 3:
            uz = self.results.get('UZ', 0.0) * scale_factor

        return (x + ux, y + uy, z + uz)

    def __str__(self):
        """
        Representación en string del nodo.

        Returns:
            str: Representación legible del nodo, sus coordenadas y estados.
        """
        x, y, z = self.get_coordinates()
        result = f"Node {self.id} ({self.dimension}D): ({x:.4f}, {y:.4f}, {z:.4f})\n"

        # Añadir información sobre cargas
        active_loads = {k: v for k, v in self.loads.items() if abs(v) > 1e-10}
        if active_loads:
            result += "  Loads:\n"
            for load_type, value in active_loads.items():
                result += f"    {load_type}: {value:.4f}\n"

        # Añadir información sobre resultados
        if self.results:
            result += "  Results:\n"
            for res_type, value in self.results.items():
                result += f"    {res_type}: {value:.6f}\n"

        # Añadir información sobre DOFs
        result += str(self.dofs).replace('DOFs for Node',
                                         '  DOFs:').replace('\n', '\n  ')

        return result


class Element:
    """
    Clase base para representar un elemento en análisis estructural.

    Un elemento conecta nodos y tiene propiedades físicas y geométricas que definen
    su comportamiento estructural. Esta clase base implementa las funcionalidades
    comunes a todos los tipos de elementos.
    """

    def __init__(self, id, nodes, material=None, section=None, element_type=None):
        """
        Inicializa un elemento con sus nodos y propiedades.

        Args:
            id (int): Identificador único del elemento.
            nodes (list): Lista de objetos Node que forman el elemento.
            material (Material, opcional): Material del elemento.
            section (Section, opcional): Sección transversal del elemento.
            element_type (str, opcional): Tipo de elemento ('bar', 'beam', 'truss', 'frame', etc.).
        """
        self.id = id
        self.nodes = nodes
        self.material = material
        self.section = section
        self.element_type = element_type or "generic"

        # Determinar la dimensión basada en los nodos
        self.dimension = self._determine_dimension()

        # Sistemas de coordenadas
        self.local_system = None
        self.transform_matrix = None

        # Matrices de rigidez y vectores de fuerzas
        self.local_stiffness_matrix = None
        self.global_stiffness_matrix = None
        self.equivalent_nodal_forces = {}

        # Resultados del análisis
        self.internal_forces = {}
        self.stresses = {}
        self.strains = {}

        # Inicializar sistema de coordenadas local
        self._init_local_coordinate_system()

    def _determine_dimension(self):
        """
        Determina la dimensión del elemento basado en los nodos.

        Returns:
            int: Dimensión del elemento (2 o 3).
        """
        # Verificar si algún nodo está en 3D
        for node in self.nodes:
            if node.dimension == 3:
                return 3
        return 2

    def _init_local_coordinate_system(self):
        """
        Inicializa el sistema de coordenadas local del elemento.
        Este método debe ser sobrescrito por clases derivadas específicas.
        """
        # En la clase base, solo inicializamos con identidad
        self.local_system = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
        self.transform_matrix = None  # Se calculará en las clases derivadas

    def get_length(self):
        """
        Calcula la longitud del elemento.

        Returns:
            float: Longitud del elemento.
        """
        if len(self.nodes) < 2:
            return 0.0

        # Para elementos lineales, calcular distancia entre extremos
        return self.nodes[0].distance_to(self.nodes[-1])

    def get_dof_indices(self, global_only=True):
        """
        Obtiene los índices de los grados de libertad del elemento en el sistema global.

        Args:
            global_only (bool): Si es True, solo devuelve DOFs con índice global asignado.

        Returns:
            list: Lista de índices globales de los DOFs del elemento.
        """
        indices = []
        for node in self.nodes:
            if global_only:
                # Solo incluir DOFs con índice global asignado
                node_indices = [
                    idx for dof, idx in node.dofs.get_global_indices().items() if idx >= 0]
            else:
                # Incluir todos los DOFs activos
                node_indices = [idx for dof, idx in node.dofs.indices.items()
                                if not node.dofs.is_constrained(dof)]

            indices.extend(node_indices)

        return indices

    def compute_local_stiffness_matrix(self):
        """
        Calcula la matriz de rigidez local del elemento.
        Este método debe ser sobrescrito por clases derivadas específicas.

        Returns:
            numpy.ndarray: Matriz de rigidez en coordenadas locales.
        """
        raise NotImplementedError(
            "Este método debe ser implementado en las clases derivadas")

    def compute_transform_matrix(self):
        """
        Calcula la matriz de transformación de coordenadas locales a globales.
        Este método debe ser sobrescrito por clases derivadas específicas.

        Returns:
            numpy.ndarray: Matriz de transformación.
        """
        raise NotImplementedError(
            "Este método debe ser implementado en las clases derivadas")

    def compute_global_stiffness_matrix(self):
        """
        Calcula la matriz de rigidez global del elemento.

        Returns:
            numpy.ndarray: Matriz de rigidez en coordenadas globales.
        """
        import numpy as np

        # Verificar si ya se ha calculado la matriz local
        if self.local_stiffness_matrix is None:
            self.local_stiffness_matrix = self.compute_local_stiffness_matrix()

        # Verificar si ya se ha calculado la matriz de transformación
        if self.transform_matrix is None:
            self.transform_matrix = self.compute_transform_matrix()

        # Transformar la matriz local a global: K_global = T^T * K_local * T
        T = self.transform_matrix
        K_local = self.local_stiffness_matrix

        K_global = np.matmul(np.matmul(T.transpose(), K_local), T)
        self.global_stiffness_matrix = K_global

        return K_global

    def get_element_dof_count(self):
        """
        Obtiene el número total de grados de libertad del elemento.

        Returns:
            int: Número total de DOFs del elemento.
        """
        count = 0
        for node in self.nodes:
            count += len(node.dofs.indices)
        return count

    def apply_distributed_load(self, load_type, magnitude, direction=None, local=True):
        """
        Aplica una carga distribuida al elemento y calcula las fuerzas nodales equivalentes.

        Args:
            load_type (str): Tipo de carga ('uniform', 'linear', 'point', etc.).
            magnitude (float o tuple): Magnitud de la carga o valores iniciales/finales.
            direction (list, opcional): Vector que indica la dirección de la carga.
            local (bool): Si es True, la carga se aplica en coordenadas locales.

        Returns:
            dict: Diccionario con las fuerzas nodales equivalentes.
        """
        # Este método debe ser implementado específicamente para cada tipo de elemento
        raise NotImplementedError(
            "Este método debe ser implementado en las clases derivadas")

    def compute_equivalent_nodal_forces(self):
        """
        Calcula las fuerzas nodales equivalentes para todas las cargas aplicadas.

        Returns:
            dict: Diccionario con fuerzas nodales equivalentes para cada nodo.
        """
        # Este método debe ser implementado específicamente para cada tipo de elemento
        raise NotImplementedError(
            "Este método debe ser implementado en las clases derivadas")

    def compute_internal_forces(self, nodal_displacements):
        """
        Calcula las fuerzas internas del elemento a partir de los desplazamientos nodales.

        Args:
            nodal_displacements (dict): Diccionario con los desplazamientos nodales.

        Returns:
            dict: Diccionario con las fuerzas internas del elemento.
        """
        # Este método debe ser implementado específicamente para cada tipo de elemento
        raise NotImplementedError(
            "Este método debe ser implementado en las clases derivadas")

    def compute_stresses(self, internal_forces=None):
        """
        Calcula las tensiones en el elemento a partir de las fuerzas internas.

        Args:
            internal_forces (dict, opcional): Fuerzas internas. Si es None, usa las almacenadas.

        Returns:
            dict: Diccionario con las tensiones en el elemento.
        """
        # Este método debe ser implementado específicamente para cada tipo de elemento
        raise NotImplementedError(
            "Este método debe ser implementado en las clases derivadas")

    def compute_strains(self, internal_forces=None):
        """
        Calcula las deformaciones en el elemento a partir de las fuerzas internas.

        Args:
            internal_forces (dict, opcional): Fuerzas internas. Si es None, usa las almacenadas.

        Returns:
            dict: Diccionario con las deformaciones en el elemento.
        """
        # Este método debe ser implementado específicamente para cada tipo de elemento
        raise NotImplementedError(
            "Este método debe ser implementado en las clases derivadas")

    def assemble_into_global(self, global_matrix, global_vector=None):
        """
        Ensambla la matriz de rigidez del elemento en la matriz global.

        Args:
            global_matrix (numpy.ndarray): Matriz global donde se ensamblará.
            global_vector (numpy.ndarray, opcional): Vector global de fuerzas.

        Returns:
            tuple: Matriz y vector globales actualizados.
        """
        import numpy as np

        # Obtener matriz de rigidez global del elemento
        if self.global_stiffness_matrix is None:
            self.compute_global_stiffness_matrix()

        # Obtener índices globales de los DOFs del elemento
        dof_indices = self.get_dof_indices(global_only=True)

        # Ensamblar matriz de rigidez
        n_dofs = len(dof_indices)
        for i in range(n_dofs):
            for j in range(n_dofs):
                global_i = dof_indices[i]
                global_j = dof_indices[j]

                if global_i >= 0 and global_j >= 0:
                    global_matrix[global_i,
                                  global_j] += self.global_stiffness_matrix[i, j]

        # Ensamblar vector de fuerzas nodales equivalentes si se proporciona
        if global_vector is not None:
            if not self.equivalent_nodal_forces:
                self.compute_equivalent_nodal_forces()

            # Recorrer cada nodo y ensamblar sus fuerzas
            for node_idx, node in enumerate(self.nodes):
                for dof_name, force in self.equivalent_nodal_forces.get(node_idx, {}).items():
                    # Obtener índice global para este DOF
                    global_idx = node.dofs.indices.get(dof_name, -1)

                    if global_idx >= 0:
                        global_vector[global_idx] += force

        return global_matrix, global_vector

    def get_nodal_coordinates(self):
        """
        Obtiene las coordenadas de todos los nodos del elemento.

        Returns:
            list: Lista de tuplas con las coordenadas (x, y, z) de cada nodo.
        """
        return [node.get_coordinates() for node in self.nodes]

    def get_deformed_shape(self, scale_factor=1.0):
        """
        Obtiene las coordenadas de los nodos en la configuración deformada.

        Args:
            scale_factor (float): Factor de escala para los desplazamientos.

        Returns:
            list: Lista de tuplas con las coordenadas deformadas (x, y, z) de cada nodo.
        """
        return [node.get_deformed_coordinates(scale_factor) for node in self.nodes]

    def get_element_forces_at_point(self, xi, internal_forces=None):
        """
        Calcula las fuerzas internas en un punto específico del elemento.

        Args:
            xi (float): Coordenada natural (-1 a 1 o 0 a 1 según el elemento).
            internal_forces (dict, opcional): Fuerzas internas. Si es None, usa las almacenadas.

        Returns:
            dict: Diccionario con las fuerzas internas en el punto.
        """
        # Este método debe ser implementado específicamente para cada tipo de elemento
        raise NotImplementedError(
            "Este método debe ser implementado en las clases derivadas")

    def get_stress_at_point(self, xi, y=0, z=0, internal_forces=None):
        """
        Calcula las tensiones en un punto específico del elemento.

        Args:
            xi (float): Coordenada natural a lo largo del elemento (-1 a 1 o 0 a 1).
            y (float): Coordenada local y (distancia desde el eje neutro).
            z (float): Coordenada local z (para elementos 3D).
            internal_forces (dict, opcional): Fuerzas internas. Si es None, usa las almacenadas.

        Returns:
            dict: Diccionario con las tensiones en el punto.
        """
        # Este método debe ser implementado específicamente para cada tipo de elemento
        raise NotImplementedError(
            "Este método debe ser implementado en las clases derivadas")

    def is_valid(self):
        """
        Verifica si el elemento es válido para el análisis.

        Returns:
            bool: True si el elemento es válido, False en caso contrario.
        """
        # Verificar que haya al menos dos nodos
        if len(self.nodes) < 2:
            return False

        # Verificar que todos los nodos tengan la misma dimensión
        dimensions = set(node.dimension for node in self.nodes)
        if len(dimensions) > 1:
            return False

        # Verificar que el material y la sección estén asignados
        if self.material is None or self.section is None:
            return False

        return True

    def generate_element_mesh(self, num_divisions=1):
        """
        Genera una malla más fina dividiendo el elemento en subelementos.
        Útil para elementos de orden superior o visualización detallada.

        Args:
            num_divisions (int): Número de divisiones para el elemento.

        Returns:
            list: Lista de puntos (x, y, z) que representan la malla del elemento.
        """
        if num_divisions <= 1:
            return self.get_nodal_coordinates()

        result = []
        coords = self.get_nodal_coordinates()

        # Para elementos lineales de 2 nodos, interpolar linealmente
        if len(self.nodes) == 2:
            start = coords[0]
            end = coords[1]

            for i in range(num_divisions + 1):
                t = i / num_divisions
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                z = start[2] + t * (end[2] - start[2])
                result.append((x, y, z))

        # Para elementos con más nodos, se requeriría interpolación de orden superior
        else:
            # Implementación específica según el tipo de elemento
            # Por defecto, devolver las coordenadas de los nodos
            result = coords

        return result

    def __str__(self):
        """
        Representación en string del elemento.

        Returns:
            str: Representación legible del elemento.
        """
        nodes_str = ", ".join([str(node.id) for node in self.nodes])
        result = f"Element {self.id} ({self.element_type}, {self.dimension}D): Nodes [{nodes_str}]\n"

        if self.material:
            result += f"  Material: {self.material.id}\n"

        if self.section:
            result += f"  Section: {self.section.id}\n"

        result += f"  Length: {self.get_length():.6f}\n"

        if self.internal_forces:
            result += "  Internal Forces:\n"
            for force_type, value in self.internal_forces.items():
                result += f"    {force_type}: {value:.6f}\n"

        if self.stresses:
            result += "  Stresses:\n"
            for stress_type, value in self.stresses.items():
                result += f"    {stress_type}: {value:.6f}\n"

        return result


class SystemElements:
    """
    Clase que gestiona un sistema de elementos estructurales, sus nodos, cargas,
    y proporciona métodos para el análisis estructural.
    """

    def __init__(self):
        """Inicializa un nuevo sistema de elementos estructurales."""
        self.nodes = {}  # Diccionario de nodos {node_id: Node}
        self.elements = {}  # Diccionario de elementos {element_id: Element}
        self.supports = {}  # Diccionario de apoyos {node_id: Support}
        self.loads = {}  # Diccionario de cargas {load_id: Load}
        self.node_id_counter = 0
        self.element_id_counter = 0
        self.load_id_counter = 0
        self.systems_of_equations = None  # Sistema de ecuaciones global
        self.displacements = None  # Vector de desplazamientos nodales
        self.reaction_forces = None  # Vector de fuerzas de reacción
        self.element_forces = {}  # Fuerzas internas en cada elemento
        self.is_solved = False  # Indicador de si la estructura ha sido resuelta

    def add_node(self, x, y, z=0):
        """
        Añade un nodo al sistema en las coordenadas especificadas.

        Args:
            x, y, z: Coordenadas del nodo en el espacio

        Returns:
            id: Identificador único del nodo creado
        """
        node_id = self.node_id_counter
        self.nodes[node_id] = Node(node_id, x, y, z)
        self.node_id_counter += 1
        return node_id

    def add_element(self, node_i, node_j, material_properties, section_properties, element_type="beam"):
        """
        Añade un elemento estructural entre dos nodos.

        Args:
            node_i: ID del nodo inicial
            node_j: ID del nodo final
            material_properties: Propiedades del material (objeto o diccionario)
            section_properties: Propiedades de la sección (objeto o diccionario)
            element_type: Tipo de elemento (barra, viga, etc.)

        Returns:
            id: Identificador único del elemento creado
        """
        if node_i not in self.nodes or node_j not in self.nodes:
            raise ValueError(
                "Los nodos especificados no existen en el sistema")

        element_id = self.element_id_counter

        if element_type == "truss":
            self.elements[element_id] = TrussElement(
                element_id,
                self.nodes[node_i],
                self.nodes[node_j],
                material_properties,
                section_properties
            )
        elif element_type == "beam":
            self.elements[element_id] = BeamElement(
                element_id,
                self.nodes[node_i],
                self.nodes[node_j],
                material_properties,
                section_properties
            )
        else:
            raise ValueError(f"Tipo de elemento no soportado: {element_type}")

        self.element_id_counter += 1
        return element_id

    def add_support(self, node_id, dx=False, dy=False, dz=False, rx=False, ry=False, rz=False):
        """
        Añade restricciones de apoyo a un nodo.

        Args:
            node_id: ID del nodo
            dx, dy, dz: Restricciones de desplazamiento en x, y, z
            rx, ry, rz: Restricciones de rotación en x, y, z

        Returns:
            node_id: ID del nodo con el apoyo definido
        """
        if node_id not in self.nodes:
            raise ValueError(f"El nodo {node_id} no existe")

        self.supports[node_id] = Support(node_id, dx, dy, dz, rx, ry, rz)
        return node_id

    def add_load(self, node_id, fx=0, fy=0, fz=0, mx=0, my=0, mz=0):
        """
        Añade cargas puntuales a un nodo.

        Args:
            node_id: ID del nodo
            fx, fy, fz: Componentes de la fuerza en x, y, z
            mx, my, mz: Componentes del momento en x, y, z

        Returns:
            load_id: Identificador único de la carga creada
        """
        if node_id not in self.nodes:
            raise ValueError(f"El nodo {node_id} no existe")

        load_id = self.load_id_counter
        self.loads[load_id] = NodalLoad(
            load_id, node_id, fx, fy, fz, mx, my, mz)
        self.load_id_counter += 1
        return load_id

    def add_distributed_load(self, element_id, q_x=0, q_y=0, q_z=0, start=0, end=1):
        """
        Añade una carga distribuida a un elemento.

        Args:
            element_id: ID del elemento
            q_x, q_y, q_z: Intensidad de la carga en las direcciones x, y, z
            start: Posición relativa de inicio de la carga (0 a 1)
            end: Posición relativa de fin de la carga (0 a 1)

        Returns:
            load_id: Identificador único de la carga creada
        """
        if element_id not in self.elements:
            raise ValueError(f"El elemento {element_id} no existe")

        load_id = self.load_id_counter
        self.loads[load_id] = DistributedLoad(
            load_id, element_id, q_x, q_y, q_z, start, end)
        self.load_id_counter += 1
        return load_id

    def assemble_stiffness_matrix(self):
        """
        Ensambla la matriz de rigidez global del sistema.

        Returns:
            K: Matriz de rigidez global
        """
        # Determinar tamaño de la matriz global basado en GDL
        ndof = self._count_dofs()
        K = np.zeros((ndof, ndof))

        # Ensamblar contribución de cada elemento
        for element_id, element in self.elements.items():
            # Obtener matriz de rigidez local del elemento
            k_local = element.get_stiffness_matrix()

            # Obtener matriz de transformación si es necesario
            if hasattr(element, 'get_transformation_matrix'):
                T = element.get_transformation_matrix()
                k_local = T.T @ k_local @ T

            # Obtener índices globales para este elemento
            dof_indices = self._get_element_dof_indices(element)

            # Ensamblar en la matriz global
            for i, dof_i in enumerate(dof_indices):
                if dof_i < 0:  # DOF restringido
                    continue
                for j, dof_j in enumerate(dof_indices):
                    if dof_j < 0:  # DOF restringido
                        continue
                    K[dof_i, dof_j] += k_local[i, j]

        return K

    def assemble_load_vector(self):
        """
        Ensambla el vector de cargas global del sistema.

        Returns:
            F: Vector de cargas global
        """
        ndof = self._count_dofs()
        F = np.zeros(ndof)

        # Procesar cargas nodales
        for load_id, load in self.loads.items():
            if isinstance(load, NodalLoad):
                node_id = load.node_id
                if node_id in self.nodes:
                    dof_indices = self._get_node_dof_indices(
                        self.nodes[node_id])
                    force_vector = [load.fx, load.fy,
                                    load.fz, load.mx, load.my, load.mz]

                    for i, dof in enumerate(dof_indices):
                        if dof >= 0:  # DOF no restringido
                            F[dof] += force_vector[i]

        # Procesar cargas distribuidas (convertir a cargas nodales equivalentes)
        for load_id, load in self.loads.items():
            if isinstance(load, DistributedLoad):
                element_id = load.element_id
                if element_id in self.elements:
                    element = self.elements[element_id]
                    # Obtener vector de cargas nodales equivalentes
                    f_eq = element.get_equivalent_nodal_loads(load)

                    # Transformar al sistema global si es necesario
                    if hasattr(element, 'get_transformation_matrix'):
                        T = element.get_transformation_matrix()
                        f_eq = T.T @ f_eq

                    # Ensamblar en el vector global
                    dof_indices = self._get_element_dof_indices(element)
                    for i, dof in enumerate(dof_indices):
                        if dof >= 0:  # DOF no restringido
                            F[dof] += f_eq[i]

        return F

    def solve(self):
        """
        Resuelve el sistema de ecuaciones para encontrar desplazamientos,
        reacciones y fuerzas internas.

        Returns:
            True si la solución es exitosa, False en caso contrario
        """
        try:
            # Ensamblar matriz de rigidez global y vector de cargas
            K = self.assemble_stiffness_matrix()
            F = self.assemble_load_vector()

            # Resolver el sistema de ecuaciones K*u = F
            u = np.linalg.solve(K, F)

            # Guardar desplazamientos
            self.displacements = u

            # Calcular reacciones en los apoyos
            self._calculate_reactions()

            # Calcular fuerzas internas en cada elemento
            self._calculate_element_forces()

            self.is_solved = True
            return True
        except Exception as e:
            print(f"Error al resolver el sistema: {e}")
            return False

    def _count_dofs(self):
        """
        Cuenta los grados de libertad (DOF) activos en el sistema.

        Returns:
            int: Número de DOF activos
        """
        dof_count = 0
        for node_id, node in self.nodes.items():
            # Comprobar si el nodo tiene restricciones
            if node_id in self.supports:
                support = self.supports[node_id]
                # Contar DOF no restringidos
                if not support.dx:
                    dof_count += 1
                if not support.dy:
                    dof_count += 1
                if not support.dz:
                    dof_count += 1
                if not support.rx:
                    dof_count += 1
                if not support.ry:
                    dof_count += 1
                if not support.rz:
                    dof_count += 1
            else:
                # Si no hay restricciones, todos los DOF están activos
                dof_count += 6  # 3 traslaciones + 3 rotaciones

        return dof_count

    def _get_node_dof_indices(self, node):
        """
        Obtiene los índices de los grados de libertad para un nodo.

        Args:
            node: Objeto Node

        Returns:
            list: Índices de DOF (valores negativos para DOF restringidos)
        """
        dof_indices = [-1] * 6  # Iniciar con valores negativos (restringidos)
        dof_counter = 0

        # Recorrer todos los nodos para asignar índices en orden
        for node_id in sorted(self.nodes.keys()):
            if node_id in self.supports:
                support = self.supports[node_id]
                if not support.dx:
                    if node_id == node.id:
                        dof_indices[0] = dof_counter
                    dof_counter += 1
                if not support.dy:
                    if node_id == node.id:
                        dof_indices[1] = dof_counter
                    dof_counter += 1
                if not support.dz:
                    if node_id == node.id:
                        dof_indices[2] = dof_counter
                    dof_counter += 1
                if not support.rx:
                    if node_id == node.id:
                        dof_indices[3] = dof_counter
                    dof_counter += 1
                if not support.ry:
                    if node_id == node.id:
                        dof_indices[4] = dof_counter
                    dof_counter += 1
                if not support.rz:
                    if node_id == node.id:
                        dof_indices[5] = dof_counter
                    dof_counter += 1
            else:
                # Sin restricciones
                if node_id == node.id:
                    dof_indices = list(range(dof_counter, dof_counter + 6))
                dof_counter += 6

        return dof_indices

    def _get_element_dof_indices(self, element):
        """
        Obtiene los índices de los grados de libertad para un elemento.

        Args:
            element: Objeto Element

        Returns:
            list: Índices de DOF para el elemento
        """
        node_i = element.node_i
        node_j = element.node_j

        dof_i = self._get_node_dof_indices(node_i)
        dof_j = self._get_node_dof_indices(node_j)

        return dof_i + dof_j  # Concatenar los DOF de ambos nodos

    def _calculate_reactions(self):
        """
        Calcula las fuerzas de reacción en los apoyos.
        """
        # Ensamblar matriz de rigidez completa (incluyendo DOF restringidos)
        K_full = self._assemble_full_stiffness_matrix()

        # Obtener desplazamientos completos (incluyendo ceros para DOF restringidos)
        u_full = self._get_full_displacement_vector()

        # R = K * u (producto matriz-vector para obtener todas las fuerzas)
        R_full = K_full @ u_full

        # Almacenar reacciones en apoyos
        self.reaction_forces = {}
        for node_id, node in self.nodes.items():
            if node_id in self.supports:
                support = self.supports[node_id]
                reactions = [0] * 6

                # Obtener índices globales completos para este nodo
                dof_indices = self._get_full_node_dof_indices(node)

                # Extraer fuerzas de reacción para DOF restringidos
                if support.dx:
                    reactions[0] = R_full[dof_indices[0]]
                if support.dy:
                    reactions[1] = R_full[dof_indices[1]]
                if support.dz:
                    reactions[2] = R_full[dof_indices[2]]
                if support.rx:
                    reactions[3] = R_full[dof_indices[3]]
                if support.ry:
                    reactions[4] = R_full[dof_indices[4]]
                if support.rz:
                    reactions[5] = R_full[dof_indices[5]]

                self.reaction_forces[node_id] = reactions

    def _calculate_element_forces(self):
        """
        Calcula las fuerzas internas en cada elemento.
        """
        for element_id, element in self.elements.items():
            # Obtener desplazamientos nodales de este elemento
            dof_indices = self._get_element_dof_indices(element)
            u_element = []
            for dof in dof_indices:
                if dof >= 0:
                    u_element.append(self.displacements[dof])
                else:
                    u_element.append(0)  # DOF restringido

            # Convertir a array numpy
            u_element = np.array(u_element)

            # Transformar desplazamientos al sistema local si es necesario
            if hasattr(element, 'get_transformation_matrix'):
                T = element.get_transformation_matrix()
                u_local = T @ u_element
            else:
                u_local = u_element

            # Calcular fuerzas internas
            element_forces = element.calculate_internal_forces(u_local)
            self.element_forces[element_id] = element_forces

    def get_nodal_displacements(self, node_id):
        """
        Obtiene los desplazamientos calculados para un nodo específico.

        Args:
            node_id: ID del nodo

        Returns:
            list: [dx, dy, dz, rx, ry, rz] desplazamientos y rotaciones
        """
        if not self.is_solved:
            raise ValueError(
                "El sistema aún no ha sido resuelto. Ejecute solve() primero.")

        if node_id not in self.nodes:
            raise ValueError(f"El nodo {node_id} no existe")

        node = self.nodes[node_id]
        dof_indices = self._get_node_dof_indices(node)

        displacements = [0] * 6
        for i, dof in enumerate(dof_indices):
            if dof >= 0:
                displacements[i] = self.displacements[dof]

        return displacements

    def get_element_forces(self, element_id):
        """
        Obtiene las fuerzas internas calculadas para un elemento específico.

        Args:
            element_id: ID del elemento

        Returns:
            dict: Fuerzas internas del elemento
        """
        if not self.is_solved:
            raise ValueError(
                "El sistema aún no ha sido resuelto. Ejecute solve() primero.")

        if element_id not in self.elements:
            raise ValueError(f"El elemento {element_id} no existe")

        return self.element_forces[element_id]

    def get_reaction_forces(self, node_id):
        """
        Obtiene las fuerzas de reacción en un nodo con apoyo.

        Args:
            node_id: ID del nodo

        Returns:
            list: [Rx, Ry, Rz, Mx, My, Mz] fuerzas y momentos de reacción
        """
        if not self.is_solved:
            raise ValueError(
                "El sistema aún no ha sido resuelto. Ejecute solve() primero.")

        if node_id not in self.supports:
            raise ValueError(f"El nodo {node_id} no tiene apoyos definidos")

        return self.reaction_forces[node_id]

    def plot_structure(self, show_nodes=True, show_elements=True, show_loads=True, show_supports=True):
        """
        Genera una representación gráfica de la estructura.

        Args:
            show_nodes: Mostrar los nodos
            show_elements: Mostrar los elementos
            show_loads: Mostrar las cargas
            show_supports: Mostrar los apoyos

        Returns:
            fig: Objeto de figura para visualización
        """
        # Esta función requeriría una implementación específica dependiendo
        # de la biblioteca gráfica utilizada (matplotlib, plotly, etc.)
        pass

    def plot_deformed_shape(self, scale_factor=1.0):
        """
        Genera una representación gráfica de la estructura deformada.

        Args:
            scale_factor: Factor de escala para los desplazamientos

        Returns:
            fig: Objeto de figura para visualización
        """
        if not self.is_solved:
            raise ValueError(
                "El sistema aún no ha sido resuelto. Ejecute solve() primero.")

        # Esta función requeriría una implementación específica dependiendo
        # de la biblioteca gráfica utilizada (matplotlib, plotly, etc.)
        pass

    def plot_bending_moment(self, element_id=None):
        """
        Genera un diagrama de momento flector para los elementos.

        Args:
            element_id: ID específico de elemento, o None para todos

        Returns:
            fig: Objeto de figura para visualización
        """
        if not self.is_solved:
            raise ValueError(
                "El sistema aún no ha sido resuelto. Ejecute solve() primero.")

        # Esta función requeriría una implementación específica dependiendo
        # de la biblioteca gráfica utilizada (matplotlib, plotly, etc.)
        pass

    def plot_shear_force(self, element_id=None):
        """
        Genera un diagrama de fuerza cortante para los elementos.

        Args:
            element_id: ID específico de elemento, o None para todos

        Returns:
            fig: Objeto de figura para visualización
        """
        if not self.is_solved:
            raise ValueError(
                "El sistema aún no ha sido resuelto. Ejecute solve() primero.")

        # Esta función requeriría una implementación específica dependiendo
        # de la biblioteca gráfica utilizada (matplotlib, plotly, etc.)
        pass

    def plot_axial_force(self, element_id=None):
        """
        Genera un diagrama de fuerza axial para los elementos.

        Args:
            element_id: ID específico de elemento, o None para todos

        Returns:
            fig: Objeto de figura para visualización
        """
        if not self.is_solved:
            raise ValueError(
                "El sistema aún no ha sido resuelto. Ejecute solve() primero.")

        # Esta función requeriría una implementación específica dependiendo
        # de la biblioteca gráfica utilizada (matplotlib, plotly, etc.)
        pass

    def to_json(self):
        """
        Exporta la estructura a formato JSON para su almacenamiento.

        Returns:
            str: Representación JSON del sistema de elementos
        """
        # Implementación para serializar la estructura
        pass

    @classmethod
    def from_json(cls, json_str):
        """
        Crea un sistema de elementos a partir de una representación JSON.

        Args:
            json_str: Representación JSON del sistema

        Returns:
            SystemElements: Nuevo objeto con la estructura cargada
        """
        # Implementación para deserializar la estructura
        pass


class AnalysisOptions:
    """
    Clase que contiene opciones de configuración para el análisis estructural.
    """

    def __init__(self,
                 analysis_type="linear_static",
                 dimension=3,
                 store_displacements=True,
                 store_reactions=True,
                 store_internal_forces=True,
                 store_stresses=False,
                 store_strains=False,
                 mass_type="consistent",
                 num_modes=10,
                 nonlinear_method="newton_raphson",
                 max_iterations=50,
                 convergence_tolerance=1e-6,
                 load_steps=1,
                 use_line_search=False):
        """
        Inicializa opciones para el análisis.

        Args:
            analysis_type: Tipo de análisis ("linear_static", "modal", "buckling", "nonlinear")
            dimension: Dimensión del análisis (2 o 3)
            store_displacements: Almacenar desplazamientos nodales
            store_reactions: Almacenar fuerzas de reacción en apoyos
            store_internal_forces: Almacenar fuerzas internas en elementos
            store_stresses: Almacenar tensiones en elementos
            store_strains: Almacenar deformaciones en elementos
            mass_type: Tipo de matriz de masa ("consistent" o "lumped")
            num_modes: Número de modos a calcular (para análisis modal o de pandeo)
            nonlinear_method: Método para análisis no lineal ("newton_raphson", "modified_newton", "arc_length")
            max_iterations: Número máximo de iteraciones para análisis no lineal
            convergence_tolerance: Tolerancia para convergencia en análisis no lineal
            load_steps: Número de pasos de carga para análisis no lineal
            use_line_search: Usar búsqueda lineal para mejorar convergencia
        """
        self.analysis_type = analysis_type
        self.dimension = dimension
        self.store_displacements = store_displacements
        self.store_reactions = store_reactions
        self.store_internal_forces = store_internal_forces
        self.store_stresses = store_stresses
        self.store_strains = store_strains
        self.mass_type = mass_type
        self.num_modes = num_modes
        self.nonlinear_method = nonlinear_method
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.load_steps = load_steps
        self.use_line_search = use_line_search

    def validate(self):
        """
        Valida la configuración de opciones.

        Returns:
            bool: True si las opciones son válidas, False en caso contrario
        """
        # Validar tipo de análisis
        valid_analysis_types = ["linear_static",
                                "modal", "buckling", "nonlinear"]
        if self.analysis_type not in valid_analysis_types:
            print(
                f"Tipo de análisis no válido: {self.analysis_type}. Valores permitidos: {valid_analysis_types}")
            return False

        # Validar dimensión
        if self.dimension not in [2, 3]:
            print(
                f"Dimensión no válida: {self.dimension}. Valores permitidos: 2, 3")
            return False

        # Validar tipo de matriz de masa
        valid_mass_types = ["consistent", "lumped"]
        if self.mass_type not in valid_mass_types:
            print(
                f"Tipo de matriz de masa no válido: {self.mass_type}. Valores permitidos: {valid_mass_types}")
            return False

        # Validar método no lineal
        if self.analysis_type == "nonlinear":
            valid_nonlinear_methods = [
                "newton_raphson", "modified_newton", "arc_length"]
            if self.nonlinear_method not in valid_nonlinear_methods:
                print(
                    f"Método no lineal no válido: {self.nonlinear_method}. Valores permitidos: {valid_nonlinear_methods}")
                return False

            if self.max_iterations <= 0:
                print(
                    f"Número máximo de iteraciones debe ser positivo: {self.max_iterations}")
                return False

            if self.convergence_tolerance <= 0:
                print(
                    f"Tolerancia de convergencia debe ser positiva: {self.convergence_tolerance}")
                return False

            if self.load_steps <= 0:
                print(
                    f"Número de pasos de carga debe ser positivo: {self.load_steps}")
                return False

        return True

    def to_dict(self):
        """
        Convierte las opciones a un diccionario.

        Returns:
            dict: Diccionario con las opciones de análisis
        """
        return {
            "analysis_type": self.analysis_type,
            "dimension": self.dimension,
            "store_displacements": self.store_displacements,
            "store_reactions": self.store_reactions,
            "store_internal_forces": self.store_internal_forces,
            "store_stresses": self.store_stresses,
            "store_strains": self.store_strains,
            "mass_type": self.mass_type,
            "num_modes": self.num_modes,
            "nonlinear_method": self.nonlinear_method,
            "max_iterations": self.max_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "load_steps": self.load_steps,
            "use_line_search": self.use_line_search
        }

    @classmethod
    def from_dict(cls, options_dict):
        """
        Crea un objeto de opciones a partir de un diccionario.

        Args:
            options_dict: Diccionario con opciones

        Returns:
            AnalysisOptions: Nuevo objeto con las opciones especificadas
        """
        return cls(**options_dict)


class ResultOptions:
    """
    Clase que define las opciones para el almacenamiento y procesamiento de resultados.
    """

    def __init__(self):
        """
        Inicializa las opciones de resultados con valores predeterminados.
        """
        # Opciones para análisis estático
        self.store_displacements = True
        self.store_reactions = True
        self.store_internal_forces = True
        self.store_stresses = True
        self.store_strains = True

        # Opciones para análisis modal
        self.store_frequencies = True
        self.store_mode_shapes = True
        self.store_participation_factors = True
        self.store_effective_masses = True
        self.num_modes = 10  # Número de modos a almacenar por defecto

        # Opciones para análisis de pandeo
        self.store_buckling_factors = True
        self.store_buckling_modes = True

        # Opciones para análisis no lineal
        self.store_load_steps = True
        self.store_convergence_history = True
        self.max_history_points = 100  # Número máximo de puntos de historia a almacenar

        # Opciones de visualización
        # Factor de escala para visualización de deformaciones
        self.deformation_scale = 100.0
        self.contour_resolution = 50  # Resolución para visualización de contornos

    def set_modal_options(self, store_frequencies=True, store_mode_shapes=True,
                          store_participation_factors=True, store_effective_masses=True,
                          num_modes=10):
        """
        Configura las opciones específicas para análisis modal.

        Args:
            store_frequencies: Si se deben almacenar las frecuencias naturales
            store_mode_shapes: Si se deben almacenar las formas modales
            store_participation_factors: Si se deben almacenar factores de participación
            store_effective_masses: Si se deben almacenar masas efectivas
            num_modes: Número de modos a almacenar
        """
        self.store_frequencies = store_frequencies
        self.store_mode_shapes = store_mode_shapes
        self.store_participation_factors = store_participation_factors
        self.store_effective_masses = store_effective_masses
        self.num_modes = num_modes

    def set_static_options(self, store_displacements=True, store_reactions=True,
                           store_internal_forces=True, store_stresses=True,
                           store_strains=True):
        """
        Configura las opciones específicas para análisis estático.

        Args:
            store_displacements: Si se deben almacenar los desplazamientos
            store_reactions: Si se deben almacenar las reacciones
            store_internal_forces: Si se deben almacenar las fuerzas internas
            store_stresses: Si se deben almacenar las tensiones
            store_strains: Si se deben almacenar las deformaciones
        """
        self.store_displacements = store_displacements
        self.store_reactions = store_reactions
        self.store_internal_forces = store_internal_forces
        self.store_stresses = store_stresses
        self.store_strains = store_strains

    def set_buckling_options(self, store_buckling_factors=True, store_buckling_modes=True,
                             num_modes=10):
        """
        Configura las opciones específicas para análisis de pandeo.

        Args:
            store_buckling_factors: Si se deben almacenar los factores de pandeo
            store_buckling_modes: Si se deben almacenar los modos de pandeo
            num_modes: Número de modos a almacenar
        """
        self.store_buckling_factors = store_buckling_factors
        self.store_buckling_modes = store_buckling_modes
        self.num_modes = num_modes

    def set_nonlinear_options(self, store_load_steps=True, store_convergence_history=True,
                              max_history_points=100):
        """
        Configura las opciones específicas para análisis no lineal.

        Args:
            store_load_steps: Si se deben almacenar los pasos de carga
            store_convergence_history: Si se debe almacenar el historial de convergencia
            max_history_points: Número máximo de puntos de historia a almacenar
        """
        self.store_load_steps = store_load_steps
        self.store_convergence_history = store_convergence_history
        self.max_history_points = max_history_points


class SolverOptions:
    """
    Clase que define las opciones para la solución de sistemas de ecuaciones
    en análisis estructural.
    """

    def __init__(self):
        """
        Inicializa las opciones del solucionador con valores predeterminados.
        """
        # Tipo de solucionador
        self.solver_type = "direct"  # Opciones: "direct", "iterative", "auto"

        # Opciones para solución directa
        self.direct_method = "lu"  # Opciones: "lu", "cholesky", "qr"

        # Opciones para solución iterativa
        self.iterative_method = "cg"  # Opciones: "cg", "gmres", "bicgstab"
        self.tolerance = 1e-6
        self.max_iterations = 1000
        self.preconditioner = "ilu"  # Opciones: "ilu", "jacobi", "ssor", "none"

        # Opciones para análisis modal
        self.modal_method = "subspace"  # Opciones: "subspace", "lanczos", "arnoldi"
        self.num_modes = 10
        self.shift_value = 0.0  # Para shift-invert en problemas con modos rígidos

        # Opciones para análisis de pandeo
        self.buckling_method = "subspace"  # Opciones: "subspace", "lanczos", "arnoldi"
        self.num_buckling_modes = 5

        # Opciones para análisis no lineal
        # "newton-raphson", "modified-newton", "arc-length"
        self.nonlinear_method = "newton-raphson"
        self.load_increments = 10
        self.max_iterations_nl = 25
        self.convergence_tolerance = 1e-4
        self.divergence_tolerance = 1e6
        self.line_search = False
        self.adaptive_load_stepping = True

        # Opciones de paralelización
        self.use_parallel = False
        self.num_threads = 4

        # Opciones de manejo de memoria
        self.use_sparse_matrices = True
        self.in_core_solution = True  # True: en memoria, False: solución out-of-core

    def set_direct_solver(self, method="lu"):
        """
        Configura el solucionador para usar método directo.

        Args:
            method: Método directo a utilizar ("lu", "cholesky", "qr")
        """
        self.solver_type = "direct"
        self.direct_method = method

    def set_iterative_solver(self, method="cg", tolerance=1e-6, max_iterations=1000,
                             preconditioner="ilu"):
        """
        Configura el solucionador para usar método iterativo.

        Args:
            method: Método iterativo a utilizar ("cg", "gmres", "bicgstab")
            tolerance: Tolerancia para convergencia
            max_iterations: Número máximo de iteraciones
            preconditioner: Precondicionador a utilizar ("ilu", "jacobi", "ssor", "none")
        """
        self.solver_type = "iterative"
        self.iterative_method = method
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.preconditioner = preconditioner

    def set_modal_solver(self, method="subspace", num_modes=10, shift_value=0.0):
        """
        Configura el solucionador para análisis modal.

        Args:
            method: Método para eigenvalores ("subspace", "lanczos", "arnoldi")
            num_modes: Número de modos a calcular
            shift_value: Valor de desplazamiento para shift-invert
        """
        self.modal_method = method
        self.num_modes = num_modes
        self.shift_value = shift_value

    def set_buckling_solver(self, method="subspace", num_modes=5):
        """
        Configura el solucionador para análisis de pandeo.

        Args:
            method: Método para eigenvalores ("subspace", "lanczos", "arnoldi")
            num_modes: Número de modos de pandeo a calcular
        """
        self.buckling_method = method
        self.num_buckling_modes = num_modes

    def set_nonlinear_solver(self, method="newton-raphson", load_increments=10,
                             max_iterations=25, tolerance=1e-4, adaptive_load_stepping=True,
                             line_search=False):
        """
        Configura el solucionador para análisis no lineal.

        Args:
            method: Método no lineal ("newton-raphson", "modified-newton", "arc-length")
            load_increments: Número de incrementos de carga
            max_iterations: Número máximo de iteraciones por incremento
            tolerance: Tolerancia para convergencia
            adaptive_load_stepping: Si se debe usar adaptación de incremento de carga
            line_search: Si se debe usar line search para mejorar convergencia
        """
        self.nonlinear_method = method
        self.load_increments = load_increments
        self.max_iterations_nl = max_iterations
        self.convergence_tolerance = tolerance
        self.adaptive_load_stepping = adaptive_load_stepping
        self.line_search = line_search

    def enable_parallelization(self, num_threads=4):
        """
        Habilita la paralelización para el solucionador.

        Args:
            num_threads: Número de hilos a utilizar
        """
        self.use_parallel = True
        self.num_threads = num_threads

    def disable_parallelization(self):
        """
        Deshabilita la paralelización para el solucionador.
        """
        self.use_parallel = False

    def use_sparse_matrix_format(self, use_sparse=True):
        """
        Configura si se deben usar matrices dispersas.

        Args:
            use_sparse: Si se deben usar matrices dispersas
        """
        self.use_sparse_matrices = use_sparse


class PlotterOptions:
    """
    Clase que define las opciones para la visualización de resultados
    de análisis estructural.
    """

    def __init__(self):
        """
        Inicializa las opciones de visualización con valores predeterminados.
        """
        # Opciones generales
        self.fig_width = 10
        self.fig_height = 8
        self.dpi = 100
        self.background_color = 'white'
        self.grid = True
        self.title_font_size = 12
        self.axis_font_size = 10
        self.legend_font_size = 10
        # 'default', 'dark_background', 'ggplot', etc.
        self.plot_style = 'default'

        # Opciones para visualización de estructura
        self.node_size = 6
        self.node_color = 'blue'
        self.element_line_width = 1.0
        self.element_color = 'black'
        self.support_size = 10
        self.highlight_selected = True
        self.selected_color = 'red'
        self.node_labels = False
        self.element_labels = False
        self.label_font_size = 8

        # Opciones para visualización de resultados
        self.deformation_scale = 100.0  # Factor de escala para deformaciones
        self.show_undeformed = True     # Mostrar estructura sin deformar
        self.undeformed_style = 'dashed'  # Estilo para estructura sin deformar
        self.undeformed_color = 'gray'   # Color para estructura sin deformar

        # Opciones para contornos y diagramas
        self.contour_type = 'filled'     # 'filled', 'lines', 'both'
        self.colormap = 'jet'            # 'jet', 'viridis', 'coolwarm', etc.
        self.colormap_reverse = False    # Invertir mapa de colores
        self.num_contours = 20           # Número de niveles en contornos
        self.show_colorbar = True        # Mostrar barra de colores
        self.colorbar_label = ''         # Etiqueta para barra de colores
        self.edge_color = 'black'        # Color de bordes en contornos
        self.alpha = 0.7                 # Transparencia para contornos

        # Opciones para diagramas de esfuerzos
        self.internal_forces_scale = 1.0  # Factor de escala para diagramas de esfuerzos
        self.shear_force_color = 'green'  # Color para diagramas de cortante
        self.bending_moment_color = 'red'  # Color para diagramas de momento
        self.axial_force_color = 'blue'   # Color para diagramas de axil
        self.fill_diagram = True          # Rellenar diagramas
        self.moment_on_tension_side = True  # Dibujar momentos en lado de tensión

        # Opciones para análisis modal y pandeo
        self.mode_animation = True        # Animar modos de vibración/pandeo
        self.mode_animation_fps = 20      # Frames por segundo en animación
        self.mode_animation_loops = 3     # Número de ciclos en animación
        self.mode_animation_amplitude = 1.0  # Amplitud para animación de modos
        self.mode_color = 'purple'        # Color para visualización de modos

        # Opciones para salida y guardado
        # Formato para guardar imágenes ('png', 'pdf', 'svg')
        self.save_format = 'png'
        self.save_dpi = 300               # DPI para guardar imágenes
        self.transparent_background = False  # Fondo transparente al guardar
        self.tight_layout = True          # Ajuste automático de layout

    def set_figure_options(self, width=10, height=8, dpi=100, background_color='white',
                           plot_style='default'):
        """
        Configura opciones para la figura.

        Args:
            width: Ancho de la figura en pulgadas
            height: Alto de la figura en pulgadas
            dpi: Resolución de la figura
            background_color: Color de fondo
            plot_style: Estilo general del plot
        """
        self.fig_width = width
        self.fig_height = height
        self.dpi = dpi
        self.background_color = background_color
        self.plot_style = plot_style

    def set_structure_display(self, node_size=6, node_color='blue', element_line_width=1.0,
                              element_color='black', support_size=10, node_labels=False,
                              element_labels=False):
        """
        Configura opciones para visualización de la estructura.

        Args:
            node_size: Tamaño de los nodos en visualización
            node_color: Color de los nodos
            element_line_width: Ancho de línea para elementos
            element_color: Color de los elementos
            support_size: Tamaño de los símbolos de apoyo
            node_labels: Si se muestran etiquetas en nodos
            element_labels: Si se muestran etiquetas en elementos
        """
        self.node_size = node_size
        self.node_color = node_color
        self.element_line_width = element_line_width
        self.element_color = element_color
        self.support_size = support_size
        self.node_labels = node_labels
        self.element_labels = element_labels

    def set_deformation_options(self, scale=100.0, show_undeformed=True,
                                undeformed_style='dashed', undeformed_color='gray'):
        """
        Configura opciones para visualización de deformaciones.

        Args:
            scale: Factor de escala para deformaciones
            show_undeformed: Si se muestra la estructura sin deformar
            undeformed_style: Estilo de línea para estructura sin deformar
            undeformed_color: Color para estructura sin deformar
        """
        self.deformation_scale = scale
        self.show_undeformed = show_undeformed
        self.undeformed_style = undeformed_style
        self.undeformed_color = undeformed_color

    def set_contour_options(self, contour_type='filled', colormap='jet',
                            num_contours=20, show_colorbar=True, colorbar_label='',
                            edge_color='black', alpha=0.7):
        """
        Configura opciones para visualización de contornos.

        Args:
            contour_type: Tipo de contorno ('filled', 'lines', 'both')
            colormap: Mapa de colores a utilizar
            num_contours: Número de niveles en contornos
            show_colorbar: Si se muestra barra de colores
            colorbar_label: Etiqueta para barra de colores
            edge_color: Color de bordes en contornos
            alpha: Transparencia para contornos
        """
        self.contour_type = contour_type
        self.colormap = colormap
        self.num_contours = num_contours
        self.show_colorbar = show_colorbar
        self.colorbar_label = colorbar_label
        self.edge_color = edge_color
        self.alpha = alpha

    def set_internal_forces_options(self, scale=1.0, shear_color='green',
                                    moment_color='red', axial_color='blue',
                                    fill_diagram=True, moment_on_tension_side=True):
        """
        Configura opciones para visualización de esfuerzos internos.

        Args:
            scale: Factor de escala para diagramas
            shear_color: Color para diagrama de cortante
            moment_color: Color para diagrama de momento
            axial_color: Color para diagrama de axil
            fill_diagram: Si se rellenan los diagramas
            moment_on_tension_side: Si se dibujan momentos en lado de tensión
        """
        self.internal_forces_scale = scale
        self.shear_force_color = shear_color
        self.bending_moment_color = moment_color
        self.axial_force_color = axial_color
        self.fill_diagram = fill_diagram
        self.moment_on_tension_side = moment_on_tension_side

    def set_mode_animation_options(self, animate=True, fps=20, loops=3,
                                   amplitude=1.0, color='purple'):
        """
        Configura opciones para animación de modos.

        Args:
            animate: Si se animan los modos
            fps: Frames por segundo en animación
            loops: Número de ciclos en animación
            amplitude: Amplitud para animación
            color: Color para visualización de modos
        """
        self.mode_animation = animate
        self.mode_animation_fps = fps
        self.mode_animation_loops = loops
        self.mode_animation_amplitude = amplitude
        self.mode_color = color

    def set_save_options(self, format='png', dpi=300, transparent=False, tight_layout=True):
        """
        Configura opciones para guardar figuras.

        Args:
            format: Formato para guardar ('png', 'pdf', 'svg', etc.)
            dpi: Resolución para archivos guardados
            transparent: Si se usa fondo transparente
            tight_layout: Si se ajusta automáticamente el layout
        """
        self.save_format = format
        self.save_dpi = dpi
        self.transparent_background = transparent
        self.tight_layout = tight_layout


class Results:
    """
    Clase que almacena y procesa los resultados del análisis estructural.
    """

    def __init__(self, system_elements, result_options=None):
        """
        Inicializa un objeto para almacenar resultados.

        Args:
            system_elements: Objeto SystemElements con la estructura analizada
            result_options: Opciones para el almacenamiento de resultados
        """
        self.system_elements = system_elements
        self.options = result_options if result_options else ResultOptions()

        # Resultados para análisis estático
        self.displacements = {}
        self.reactions = {}
        self.internal_forces = {}
        self.stresses = {}
        self.strains = {}

        # Resultados para análisis modal
        self.frequencies = None
        self.mode_shapes = None
        self.participation_factors = None
        self.effective_masses = None

        # Resultados para análisis de pandeo
        self.buckling_factors = None
        self.buckling_modes = None

        # Resultados para análisis no lineal
        self.load_steps = []
        self.displacement_history = []
        self.force_history = []
        self.convergence_history = []

    def process_results(self):
        """
        Procesa y almacena los resultados del análisis estático lineal.
        """
        # Procesar desplazamientos si se requiere
        if self.options.store_displacements:
            self._process_displacements()

        # Procesar reacciones si se requiere
        if self.options.store_reactions:
            self._process_reactions()

        # Procesar fuerzas internas si se requiere
        if self.options.store_internal_forces:
            self._process_internal_forces()

        # Procesar tensiones si se requiere
        if self.options.store_stresses:
            self._process_stresses()

        # Procesar deformaciones si se requiere
        if self.options.store_strains:
            self._process_strains()

    def _process_displacements(self):
        """
        Procesa y almacena los desplazamientos nodales.
        """
        for node_id in self.system_elements.nodes:
            self.displacements[node_id] = self.system_elements.get_nodal_displacements(
                node_id)

    def _process_reactions(self):
        """
        Procesa y almacena las reacciones en apoyos.
        """
        for node_id in self.system_elements.supports:
            self.reactions[node_id] = self.system_elements.get_reaction_forces(
                node_id)

    def _process_internal_forces(self):
        """
        Procesa y almacena las fuerzas internas en elementos.
        """
        for element_id in self.system_elements.elements:
            self.internal_forces[element_id] = self.system_elements.get_element_forces(
                element_id)

    def _process_stresses(self):
        """
        Procesa y almacena las tensiones en los elementos.
        """
        for element_id, element in self.system_elements.elements.items():
            # Verificar si el elemento implementa el cálculo de tensiones
            if hasattr(element, 'calculate_stresses'):
                # Obtener fuerzas internas
                forces = self.internal_forces.get(element_id, None)

                if forces is not None:
                    # Calcular tensiones
                    self.stresses[element_id] = element.calculate_stresses(
                        forces)

    def _process_strains(self):
        """
        Procesa y almacena las deformaciones en los elementos.
        """
        for element_id, element in self.system_elements.elements.items():
            # Verificar si el elemento implementa el cálculo de deformaciones
            if hasattr(element, 'calculate_strains'):
                # Obtener fuerzas internas
                forces = self.internal_forces.get(element_id, None)

                if forces is not None:
                    # Calcular deformaciones
                    self.strains[element_id] = element.calculate_strains(
                        forces)

    def store_modal_results(self, eigenvalues, eigenvectors):
        """
        Almacena los resultados del análisis modal.

        Args:
            eigenvalues: Vector de autovalores (cuadrados de frecuencias angulares)
            eigenvectors: Matriz de autovectores (modos de vibración)
        """
        if self.options.store_frequencies:
            # Convertir autovalores a frecuencias naturales en Hz
            self.frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

        if self.options.store_mode_shapes:
            # Almacenar formas modales
            self.mode_shapes = {}

            # Limitar al número de modos especificado
            num_modes = min(self.options.num_modes, eigenvectors.shape[1])

            for i in range(num_modes):
                # Extraer forma modal i
                mode_vector = eigenvectors[:, i]

                # Reorganizar en formato por nodo
                mode_by_node = {}

                for node_id, node in self.system_elements.nodes.items():
                    # Obtener índices de DOF para este nodo
                    dof_indices = self.system_elements._get_node_dof_indices(
                        node)

                    # Extraer desplazamientos modales
                    modal_displacements = [0] * 6
                    for j, dof in enumerate(dof_indices):
                        if dof >= 0 and dof < len(mode_vector):
                            modal_displacements[j] = mode_vector[dof]

                    mode_by_node[node_id] = modal_displacements

                self.mode_shapes[i] = mode_by_node

        if self.options.store_participation_factors:
            # Calcular factores de participación
            self._calculate_participation_factors(eigenvectors)

    def store_buckling_results(self, eigenvalues, eigenvectors):
        """
        Almacena los resultados del análisis de pandeo.

        Args:
            eigenvalues: Vector de autovalores (factores de pandeo)
            eigenvectors: Matriz de autovectores (modos de pandeo)
        """
        if self.options.store_buckling_factors:
            # Almacenar factores de pandeo
            self.buckling_factors = eigenvalues

        if self.options.store_buckling_modes:
            # Almacenar modos de pandeo
            self.buckling_modes = {}

            # Limitar al número de modos especificado
            num_modes = min(self.options.num_modes, eigenvectors.shape[1])

            for i in range(num_modes):
                # Extraer modo de pandeo i
                mode_vector = eigenvectors[:, i]


class Solver:
    """
    Clase para solucionar sistemas de ecuaciones en análisis estructural.
    Proporciona métodos para resolver diferentes tipos de problemas: estático,
    modal, pandeo y no lineal.
    """

    def __init__(self, system_elements, solver_options=None):
        """
        Inicializa el solucionador con el sistema estructural y opciones.

        Args:
            system_elements: Objeto SystemElements con la estructura a analizar
            solver_options: Opciones para el solucionador
        """
        self.system = system_elements
        self.options = solver_options if solver_options else SolverOptions()

        # Matrices del sistema
        self.K = None  # Matriz de rigidez global
        self.M = None  # Matriz de masa global
        self.C = None  # Matriz de amortiguamiento global
        self.Kg = None  # Matriz de rigidez geométrica para análisis de pandeo

        # Vectores del sistema
        self.F = None  # Vector de fuerzas
        self.u = None  # Vector de desplazamientos

        # Información de grados de libertad
        self.dofs = None  # Información de DOFs del sistema
        self.free_dofs = None  # DOFs libres
        self.constrained_dofs = None  # DOFs restringidos

        # Rendimiento y diagnóstico
        self.solution_time = 0.0
        self.assembly_time = 0.0
        self.iterations = 0
        self.converged = False

    def assemble_system(self):
        """
        Ensambla las matrices del sistema a partir de las contribuciones de los elementos.
        """
        start_time = time.time()

        # Obtener información de DOFs
        self.dofs = self.system.get_dof_map()
        self.free_dofs = self.system.get_free_dofs()
        self.constrained_dofs = self.system.get_constrained_dofs()

        # Ensamblar matriz de rigidez
        self.K = self._assemble_stiffness_matrix()

        # Ensamblar vector de fuerzas
        self.F = self._assemble_force_vector()

        # Ensamblar matriz de masa si es necesario
        if self.system.requires_mass_matrix():
            self.M = self._assemble_mass_matrix()

        # Ensamblar matriz de amortiguamiento si es necesario
        if self.system.requires_damping_matrix():
            self.C = self._assemble_damping_matrix()

        self.assembly_time = time.time() - start_time

    def _assemble_stiffness_matrix(self):
        """
        Ensambla la matriz de rigidez global del sistema.

        Returns:
            Matriz de rigidez global ensamblada
        """
        # Determinar tamaño del sistema
        ndofs = self.system.get_number_of_dofs()

        # Crear matriz de rigidez global (dispersa o densa)
        if self.options.use_sparse_matrices:
            from scipy.sparse import lil_matrix
            K = lil_matrix((ndofs, ndofs), dtype=float)
        else:
            K = np.zeros((ndofs, ndofs), dtype=float)

        # Ensamblar contribuciones de cada elemento
        for element_id, element in self.system.elements.items():
            # Obtener matriz de rigidez local
            K_elem = element.get_stiffness_matrix()

            # Obtener mapeo de DOFs locales a globales
            dof_map = self.system.get_element_dof_map(element_id)

            # Ensamblar en matriz global
            for i, dof_i in enumerate(dof_map):
                if dof_i < 0:
                    continue  # DOF restringido o no existente

                for j, dof_j in enumerate(dof_map):
                    if dof_j < 0:
                        continue  # DOF restringido o no existente

                    K[dof_i, dof_j] += K_elem[i, j]

        # Convertir a formato CSR si es matriz dispersa
        if self.options.use_sparse_matrices:
            K = K.tocsr()

        return K

    def _assemble_mass_matrix(self):
        """
        Ensambla la matriz de masa global del sistema.

        Returns:
            Matriz de masa global ensamblada
        """
        # Determinar tamaño del sistema
        ndofs = self.system.get_number_of_dofs()

        # Crear matriz de masa global (dispersa o densa)
        if self.options.use_sparse_matrices:
            from scipy.sparse import lil_matrix
            M = lil_matrix((ndofs, ndofs), dtype=float)
        else:
            M = np.zeros((ndofs, ndofs), dtype=float)

        # Ensamblar contribuciones de cada elemento
        for element_id, element in self.system.elements.items():
            # Verificar si el elemento implementa matriz de masa
            if hasattr(element, 'get_mass_matrix'):
                # Obtener matriz de masa local
                M_elem = element.get_mass_matrix()

                # Obtener mapeo de DOFs locales a globales
                dof_map = self.system.get_element_dof_map(element_id)

                # Ensamblar en matriz global
                for i, dof_i in enumerate(dof_map):
                    if dof_i < 0:
                        continue  # DOF restringido o no existente

                    for j, dof_j in enumerate(dof_map):
                        if dof_j < 0:
                            continue  # DOF restringido o no existente

                        M[dof_i, dof_j] += M_elem[i, j]

        # Convertir a formato CSR si es matriz dispersa
        if self.options.use_sparse_matrices:
            M = M.tocsr()

        return M

    def _assemble_damping_matrix(self):
        """
        Ensambla la matriz de amortiguamiento global del sistema.

        Returns:
            Matriz de amortiguamiento global ensamblada
        """
        # Determinar tamaño del sistema
        ndofs = self.system.get_number_of_dofs()

        # Verificar si se usa amortiguamiento de Rayleigh
        if self.system.use_rayleigh_damping and self.M is not None and self.K is not None:
            # Obtener coeficientes de Rayleigh
            alpha = self.system.rayleigh_alpha
            beta = self.system.rayleigh_beta

            # Calcular matriz de amortiguamiento de Rayleigh
            C = alpha * self.M + beta * self.K
            return C

        # Si no, ensamblar manualmente igual que las otras matrices
        if self.options.use_sparse_matrices:
            from scipy.sparse import lil_matrix
            C = lil_matrix((ndofs, ndofs), dtype=float)
        else:
            C = np.zeros((ndofs, ndofs), dtype=float)

        # Ensamblar contribuciones de cada elemento
        for element_id, element in self.system.elements.items():
            # Verificar si el elemento implementa matriz de amortiguamiento
            if hasattr(element, 'get_damping_matrix'):
                # Obtener matriz de amortiguamiento local
                C_elem = element.get_damping_matrix()

                # Obtener mapeo de DOFs locales a globales
                dof_map = self.system.get_element_dof_map(element_id)

                # Ensamblar en matriz global
                for i, dof_i in enumerate(dof_map):
                    if dof_i < 0:
                        continue  # DOF restringido o no existente

                    for j, dof_j in enumerate(dof_map):
                        if dof_j < 0:
                            continue  # DOF restringido o no existente

                        C[dof_i, dof_j] += C_elem[i, j]

        # Convertir a formato CSR si es matriz dispersa
        if self.options.use_sparse_matrices:
            C = C.tocsr()

        return C

    def _assemble_force_vector(self):
        """
        Ensambla el vector de fuerzas global del sistema.

        Returns:
            Vector de fuerzas global ensamblado
        """
        # Determinar tamaño del sistema
        ndofs = self.system.get_number_of_dofs()

        # Crear vector de fuerzas
        F = np.zeros(ndofs, dtype=float)

        # Ensamblar fuerzas nodales
        for node_id, node in self.system.nodes.items():
            # Obtener fuerzas aplicadas al nodo
            node_forces = self.system.get_nodal_forces(node_id)

            # Obtener mapeo de DOFs para este nodo
            dof_indices = self.system.get_node_dof_indices(node_id)

            # Ensamblar en vector global
            for i, dof in enumerate(dof_indices):
                if dof >= 0:  # DOF no restringido
                    F[dof] += node_forces[i]

        # Ensamblar cargas distribuidas en elementos
        for element_id, element in self.system.elements.items():
            # Verificar si el elemento tiene cargas distribuidas
            if hasattr(element, 'get_equivalent_nodal_forces'):
                # Obtener fuerzas nodales equivalentes
                equiv_forces = element.get_equivalent_nodal_forces()

                # Obtener mapeo de DOFs para este elemento
                dof_map = self.system.get_element_dof_map(element_id)

                # Ensamblar en vector global
                for i, dof in enumerate(dof_map):
                    if dof >= 0:  # DOF no restringido
                        F[dof] += equiv_forces[i]

        return F

    def _assemble_geometric_stiffness_matrix(self):
        """
        Ensambla la matriz de rigidez geométrica para análisis de pandeo.

        Returns:
            Matriz de rigidez geométrica global
        """
        # Determinar tamaño del sistema
        ndofs = self.system.get_number_of_dofs()

        # Crear matriz de rigidez geométrica
        if self.options.use_sparse_matrices:
            from scipy.sparse import lil_matrix
            Kg = lil_matrix((ndofs, ndofs), dtype=float)
        else:
            Kg = np.zeros((ndofs, ndofs), dtype=float)

        # Se necesitan las fuerzas internas actuales para la matriz de rigidez geométrica
        if self.u is None:
            # Si no se han calculado desplazamientos, resolver primero el sistema
            self.solve_static()

        # Ensamblar contribuciones de cada elemento
        for element_id, element in self.system.elements.items():
            # Verificar si el elemento implementa matriz de rigidez geométrica
            if hasattr(element, 'get_geometric_stiffness_matrix'):
                # Obtener fuerzas internas del elemento
                forces = self.system.get_element_forces(element_id, self.u)

                # Obtener matriz de rigidez geométrica local
                Kg_elem = element.get_geometric_stiffness_matrix(forces)

                # Obtener mapeo de DOFs locales a globales
                dof_map = self.system.get_element_dof_map(element_id)

                # Ensamblar en matriz global
                for i, dof_i in enumerate(dof_map):
                    if dof_i < 0:
                        continue  # DOF restringido o no existente

                    for j, dof_j in enumerate(dof_map):
                        if dof_j < 0:
                            continue  # DOF restringido o no existente

                        Kg[dof_i, dof_j] += Kg_elem[i, j]

        # Convertir a formato CSR si es matriz dispersa
        if self.options.use_sparse_matrices:
            Kg = Kg.tocsr()

        return Kg

    def apply_boundary_conditions(self):
        """
        Aplica condiciones de contorno al sistema.
        """
        # Implementar diferentes métodos para aplicar condiciones de contorno
        # En este caso se usa el método de eliminación de filas/columnas

        # Extraer submatrices y subvectores para los DOFs libres
        self.K_free = self.K[np.ix_(self.free_dofs, self.free_dofs)]
        self.F_free = self.F[self.free_dofs]

        # Si hay desplazamientos prescritos no nulos
        prescribed_disps = self.system.get_prescribed_displacements()
        if any(prescribed_disps):
            # Ajustar el vector de fuerzas para considerar desplazamientos prescritos
            F_prescribed = self.K[np.ix_(
                self.free_dofs, self.constrained_dofs)] @ prescribed_disps
            self.F_free -= F_prescribed

    def solve_static(self):
        """
        Resuelve el sistema para análisis estático lineal.

        Returns:
            Vector de desplazamientos y objeto Results
        """
        start_time = time.time()

        # Ensamblar el sistema si aún no se ha hecho
        if self.K is None:
            self.assemble_system()

        # Aplicar condiciones de contorno
        self.apply_boundary_conditions()

        # Resolver sistema de ecuaciones Ku = F
        self.u_free = self._solve_linear_system(self.K_free, self.F_free)

        # Reconstruir vector de desplazamientos completo
        self.u = np.zeros(len(self.F))
        self.u[self.free_dofs] = self.u_free

        # Si hay desplazamientos prescritos, agregarlos
        prescribed_disps = self.system.get_prescribed_displacements()
        self.u[self.constrained_dofs] = prescribed_disps

        self.solution_time = time.time() - start_time

        # Crear y procesar resultados
        result_options = ResultOptions()
        results = Results(self.system, result_options)
        results.process_results()

        return self.u, results

    def _solve_linear_system(self, A, b):
        """
        Resuelve un sistema de ecuaciones lineales Ax = b según el método seleccionado.

        Args:
            A: Matriz del sistema
            b: Vector del lado derecho

        Returns:
            Vector solución x
        """
        # Elegir método de solución según opciones
        if self.options.solver_type == "direct":
            return self._solve_direct(A, b)
        elif self.options.solver_type == "iterative":
            return self._solve_iterative(A, b)
        else:  # Auto: elegir basado en tamaño del sistema
            if A.shape[0] > 10000 and self.options.use_sparse_matrices:
                return self._solve_iterative(A, b)
            else:
                return self._solve_direct(A, b)

    def _solve_direct(self, A, b):
        """
        Resuelve el sistema usando método directo.

        Args:
            A: Matriz del sistema
            b: Vector del lado derecho

        Returns:
            Vector solución x
        """
        if self.options.use_sparse_matrices:
            # Para matrices dispersas
            if self.options.direct_method == "lu":
                return sparse_linalg.spsolve(A, b)
            elif self.options.direct_method == "cholesky":
                from sksparse.cholmod import cholesky
                factor = cholesky(A)
                return factor(b)
            else:  # QR o cualquier otro
                return sparse_linalg.spsolve(A, b, use_umfpack=True)
        else:
            # Para matrices densas
            if self.options.direct_method == "lu":
                return np.linalg.solve(A, b)
            elif self.options.direct_method == "cholesky":
                from scipy.linalg import cholesky, cho_solve
                c, low = cho_factor(A)
                return cho_solve((c, low), b)
            elif self.options.direct_method == "qr":
                from scipy.linalg import qr, solve_triangular
                Q, R = qr(A)
                y = np.dot(Q.T, b)
                return solve_triangular(R, y)

    def _solve_iterative(self, A, b):
        """
        Resuelve el sistema usando método iterativo.

        Args:
            A: Matriz del sistema
            b: Vector del lado derecho

        Returns:
            Vector solución x
        """
        # Configurar precondicionador
        if self.options.preconditioner == "ilu":
            from scipy.sparse.linalg import spilu
            precond = spilu(A.tocsc())
            M = sparse_linalg.LinearOperator(A.shape, precond.solve)
        elif self.options.preconditioner == "jacobi":
            # Precondicionador de Jacobi (diagonal)
            diag = A.diagonal()
            M = sparse_linalg.LinearOperator(A.shape, lambda x: x / diag)
        elif self.options.preconditioner == "ssor":
            # SSOR no está disponible directamente, usar alternativa
            from pyamg import smoothed_aggregation_solver
            ml = smoothed_aggregation_solver(A)
            M = ml.aspreconditioner()
        else:  # Sin precondicionador
            M = None

        # Elegir método iterativo
        if self.options.iterative_method == "cg":
            x, info = sparse_linalg.cg(A, b, tol=self.options.tolerance,
                                       maxiter=self.options.max_iterations, M=M)
        elif self.options.iterative_method == "gmres":
            x, info = sparse_linalg.gmres(A, b, tol=self.options.tolerance,
                                          maxiter=self.options.max_iterations, M=M)
        elif self.options.iterative_method == "bicgstab":
            x, info = sparse_linalg.bicgstab(A, b, tol=self.options.tolerance,
                                             maxiter=self.options.max_iterations, M=M)
        else:
            raise ValueError(
                f"Método iterativo no soportado: {self.options.iterative_method}")

        # Guardar información de iteraciones
        self.iterations = info
        self.converged = (info == 0)

        return x

    def solve_modal(self):
        """
        Resuelve el problema de autovalores para análisis modal.

        Returns:
            Eigenvalores (frecuencias al cuadrado), eigenvectores (modos) y objeto Results
        """
        start_time = time.time()

        # Ensamblar matrices si aún no se ha hecho
        if self.K is None:
            self.assemble_system()

        # Aplicar condiciones de contorno
        self.apply_boundary_conditions()

        # Resolver problema de autovalores
        if self.options.modal_method == "subspace":
            vals, vecs = sparse_linalg.eigsh(
                self.K_free, k=self.options.num_modes, sigma=self.options.shift_value)
        elif self.options.modal_method == "lanczos":
            vals, vecs = sparse_linalg.eigsh(
                self.K_free, k=self.options.num_modes, which='SM')
        elif self.options.modal_method == "arnoldi":
            vals, vecs = sparse_linalg.eigs(
                self.K_free, k=self.options.num_modes, which='SM')
        else:
            raise ValueError(
                f"Método de autovalores no soportado: {self.options.modal_method}")

        # Ordenar autovalores y autovectores
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]

        # Crear y procesar resultados
        result_options = ResultOptions()
        result_options.set_modal_options(num_modes=self.options.num_modes)
        results = Results(self.system, result_options)
        results.process_results(vals, vecs)


class Analysis:
    """
    Clase que gestiona el proceso de análisis estructural completo,
    incluyendo la configuración, ejecución y post-procesamiento.
    """

    def __init__(self, system_elements, analysis_options=None):
        """
        Inicializa un nuevo análisis estructural.

        Args:
            system_elements: Objeto SystemElements con la estructura a analizar
            analysis_options: Opciones de análisis (objeto AnalysisOptions)
        """
        self.system_elements = system_elements
        self.analysis_options = analysis_options if analysis_options else AnalysisOptions()
        self.results = None
        self.is_analyzed = False

    def execute(self):
        """
        Ejecuta el análisis estructural según las opciones configuradas.

        Returns:
            Results: Objeto con los resultados del análisis
        """
        if self.analysis_options.analysis_type == "linear_static":
            self._execute_linear_static()
        elif self.analysis_options.analysis_type == "modal":
            self._execute_modal()
        elif self.analysis_options.analysis_type == "buckling":
            self._execute_buckling()
        elif self.analysis_options.analysis_type == "nonlinear":
            self._execute_nonlinear()
        else:
            raise ValueError(
                f"Tipo de análisis no soportado: {self.analysis_options.analysis_type}")

        self.is_analyzed = True
        return self.results

    def _execute_linear_static(self):
        """
        Ejecuta un análisis estático lineal.
        """
        # Verificar que la estructura tenga suficientes restricciones
        if not self._check_stability():
            raise ValueError(
                "La estructura no tiene suficientes restricciones para ser estable")

        # Resolver el sistema de ecuaciones
        solved = self.system_elements.solve()

        if not solved:
            raise RuntimeError("No se pudo resolver el sistema de ecuaciones")

        # Crear objeto de resultados
        result_options = ResultOptions(
            store_displacements=self.analysis_options.store_displacements,
            store_reactions=self.analysis_options.store_reactions,
            store_internal_forces=self.analysis_options.store_internal_forces,
            store_stresses=self.analysis_options.store_stresses,
            store_strains=self.analysis_options.store_strains
        )

        self.results = Results(self.system_elements, result_options)
        self.results.process_results()

    def _execute_modal(self):
        """
        Ejecuta un análisis modal (frecuencias naturales y modos de vibración).
        """
        # Verificar que la estructura tenga suficientes restricciones
        if not self._check_stability():
            raise ValueError(
                "La estructura no tiene suficientes restricciones para ser estable")

        # Ensamblar matriz de rigidez
        K = self.system_elements.assemble_stiffness_matrix()

        # Ensamblar matriz de masa
        M = self._assemble_mass_matrix()

        # Calcular frecuencias naturales y modos de vibración
        # Resolver problema de autovalores generalizado: K*phi = w^2*M*phi
        eigenvalues, eigenvectors = self._solve_generalized_eigenvalue_problem(
            K, M)

        # Crear objeto de resultados
        result_options = ResultOptions(
            store_frequencies=True,
            store_mode_shapes=True,
            num_modes=self.analysis_options.num_modes
        )

        self.results = Results(self.system_elements, result_options)
        self.results.store_modal_results(eigenvalues, eigenvectors)

    def _execute_buckling(self):
        """
        Ejecuta un análisis de pandeo.
        """
        # Verificar que la estructura tenga suficientes restricciones
        if not self._check_stability():
            raise ValueError(
                "La estructura no tiene suficientes restricciones para ser estable")

        # Realizar análisis estático lineal primero
        self._execute_linear_static()

        # Ensamblar matriz de rigidez geométrica
        K_g = self._assemble_geometric_stiffness_matrix()

        # Ensamblar matriz de rigidez
        K = self.system_elements.assemble_stiffness_matrix()

        # Calcular factores de pandeo y modos
        # Resolver problema de autovalores: K*phi = lambda*K_g*phi
        eigenvalues, eigenvectors = self._solve_generalized_eigenvalue_problem(
            K, K_g)

        # Crear objeto de resultados
        result_options = ResultOptions(
            store_buckling_factors=True,
            store_buckling_modes=True,
            num_modes=self.analysis_options.num_modes
        )

        self.results = Results(self.system_elements, result_options)
        self.results.store_buckling_results(eigenvalues, eigenvectors)

    def _execute_nonlinear(self):
        """
        Ejecuta un análisis no lineal.
        """
        # Implementar método iterativo (Newton-Raphson, Arc-Length, etc.)
        # según las opciones configuradas
        method = self.analysis_options.nonlinear_method
        max_iterations = self.analysis_options.max_iterations
        tolerance = self.analysis_options.convergence_tolerance

        if method == "newton_raphson":
            self._newton_raphson(max_iterations, tolerance)
        elif method == "modified_newton":
            self._modified_newton(max_iterations, tolerance)
        elif method == "arc_length":
            self._arc_length(max_iterations, tolerance)
        else:
            raise ValueError(f"Método no lineal no soportado: {method}")

    def _newton_raphson(self, max_iterations, tolerance):
        """
        Implementa el método de Newton-Raphson para análisis no lineal.

        Args:
            max_iterations: Número máximo de iteraciones
            tolerance: Tolerancia para convergencia
        """
        # Vector de cargas externas
        F_ext = self.system_elements.assemble_load_vector()

        # Vector de desplazamientos inicial
        u = np.zeros_like(F_ext)

        # Vector de fuerzas internas
        F_int = np.zeros_like(F_ext)

        for iteration in range(max_iterations):
            # Ensamblar matriz de rigidez tangente
            K_t = self._assemble_tangent_stiffness_matrix(u)

            # Calcular vector residual
            R = F_ext - F_int

            # Comprobar convergencia
            if np.linalg.norm(R) < tolerance:
                break

            # Resolver incremento de desplazamiento
            du = np.linalg.solve(K_t, R)

            # Actualizar desplazamiento
            u += du

            # Actualizar fuerzas internas
            F_int = self._calculate_internal_forces(u)

        # Crear objeto de resultados
        self.results = Results(self.system_elements, ResultOptions())
        self.results.store_nonlinear_results(u, F_int)

    def _check_stability(self):
        """
        Verifica que la estructura tenga suficientes restricciones para ser estable.

        Returns:
            bool: True si la estructura es estable, False en caso contrario
        """
        # Contar grados de libertad restringidos
        restrained_dofs = 0
        for node_id, support in self.system_elements.supports.items():
            if support.dx:
                restrained_dofs += 1
            if support.dy:
                restrained_dofs += 1
            if support.dz:
                restrained_dofs += 1
            if support.rx:
                restrained_dofs += 1
            if support.ry:
                restrained_dofs += 1
            if support.rz:
                restrained_dofs += 1

        # Comprobar estabilidad según el tipo de análisis
        if self.analysis_options.dimension == 2:  # Análisis 2D
            min_constraints = 3  # Mínimo para equilibrio estático en 2D
        else:  # Análisis 3D
            min_constraints = 6  # Mínimo para equilibrio estático en 3D

        return restrained_dofs >= min_constraints

    def _assemble_mass_matrix(self):
        """
        Ensambla la matriz de masa global del sistema.

        Returns:
            M: Matriz de masa global
        """
        # Determinar tamaño de la matriz global
        ndof = self.system_elements._count_dofs()
        M = np.zeros((ndof, ndof))

        # Ensamblar contribución de cada elemento
        for element_id, element in self.system_elements.elements.items():
            # Verificar si el elemento tiene implementado el método para matriz de masa
            if hasattr(element, 'get_mass_matrix'):
                # Obtener matriz de masa local del elemento
                m_local = element.get_mass_matrix()

                # Obtener matriz de transformación si es necesario
                if hasattr(element, 'get_transformation_matrix'):
                    T = element.get_transformation_matrix()
                    m_local = T.T @ m_local @ T

                # Obtener índices globales para este elemento
                dof_indices = self.system_elements._get_element_dof_indices(
                    element)

                # Ensamblar en la matriz global
                for i, dof_i in enumerate(dof_indices):
                    if dof_i < 0:  # DOF restringido
                        continue
                    for j, dof_j in enumerate(dof_indices):
                        if dof_j < 0:  # DOF restringido
                            continue
                        M[dof_i, dof_j] += m_local[i, j]

        # Aplicar tipo de matriz de masa según opciones
        if self.analysis_options.mass_type == "lumped":
            # Convertir a matriz de masa concentrada
            M_lumped = np.zeros_like(M)
            for i in range(M.shape[0]):
                M_lumped[i, i] = sum(M[i, :])
            return M_lumped

        return M

    def _assemble_geometric_stiffness_matrix(self):
        """
        Ensambla la matriz de rigidez geométrica global del sistema.

        Returns:
            K_g: Matriz de rigidez geométrica global
        """
        # Determinar tamaño de la matriz global
        ndof = self.system_elements._count_dofs()
        K_g = np.zeros((ndof, ndof))

        # Ensamblar contribución de cada elemento
        for element_id, element in self.system_elements.elements.items():
            # Verificar si el elemento tiene implementado el método para matriz de rigidez geométrica
            if hasattr(element, 'get_geometric_stiffness_matrix'):
                # Obtener fuerzas internas del elemento
                element_forces = self.system_elements.element_forces.get(
                    element_id, None)

                if element_forces is None:
                    continue

                # Obtener matriz de rigidez geométrica local del elemento
                kg_local = element.get_geometric_stiffness_matrix(
                    element_forces)

                # Obtener matriz de transformación si es necesario
                if hasattr(element, 'get_transformation_matrix'):
                    T = element.get_transformation_matrix()
                    kg_local = T.T @ kg_local @ T

                # Obtener índices globales para este elemento
                dof_indices = self.system_elements._get_element_dof_indices(
                    element)

                # Ensamblar en la matriz global
                for i, dof_i in enumerate(dof_indices):
                    if dof_i < 0:  # DOF restringido
                        continue
                    for j, dof_j in enumerate(dof_indices):
                        if dof_j < 0:  # DOF restringido
                            continue
                        K_g[dof_i, dof_j] += kg_local[i, j]

        return K_g

    def _assemble_tangent_stiffness_matrix(self, u):
        """
        Ensambla la matriz de rigidez tangente para análisis no lineal.

        Args:
            u: Vector de desplazamientos actual

        Returns:
            K_t: Matriz de rigidez tangente
        """
        # Determinar tamaño de la matriz global
        ndof = self.system_elements._count_dofs()
        K_t = np.zeros((ndof, ndof))

        # Distribuir desplazamientos globales a cada elemento
        for element_id, element in self.system_elements.elements.items():
            # Verificar si el elemento tiene implementado el método para matriz de rigidez tangente
            if hasattr(element, 'get_tangent_stiffness_matrix'):
                # Obtener índices globales para este elemento
                dof_indices = self.system_elements._get_element_dof_indices(
                    element)

                # Extraer desplazamientos del elemento
                u_element = []
                for dof in dof_indices:
                    if dof >= 0:
                        u_element.append(u[dof])
                    else:
                        u_element.append(0)  # DOF restringido

                # Convertir a array numpy
                u_element = np.array(u_element)

                # Transformar desplazamientos al sistema local si es necesario
                if hasattr(element, 'get_transformation_matrix'):
                    T = element.get_transformation_matrix()
                    u_local = T @ u_element
                else:
                    u_local = u_element

                # Obtener matriz de rigidez tangente local del elemento
                kt_local = element.get_tangent_stiffness_matrix(u_local)

                # Transformar al sistema global si es necesario
                if hasattr(element, 'get_transformation_matrix'):
                    kt_local = T.T @ kt_local @ T

                # Ensamblar en la matriz global
                for i, dof_i in enumerate(dof_indices):
                    if dof_i < 0:  # DOF restringido
                        continue
                    for j, dof_j in enumerate(dof_indices):
                        if dof_j < 0:  # DOF restringido
                            continue
                        K_t[dof_i, dof_j] += kt_local[i, j]

        return K_t

    def _calculate_internal_forces(self, u):
        """
        Calcula el vector de fuerzas internas para un vector de desplazamientos dado.

        Args:
            u: Vector de desplazamientos

        Returns:
            F_int: Vector de fuerzas internas
        """
        # Determinar tamaño del vector
        ndof = self.system_elements._count_dofs()
        F_int = np.zeros(ndof)

        # Calcular fuerzas internas de cada elemento
        for element_id, element in self.system_elements.elements.items():
            # Obtener índices globales para este elemento
            dof_indices = self.system_elements._get_element_dof_indices(
                element)

            # Extraer desplazamientos del elemento
            u_element = []
            for dof in dof_indices:
                if dof >= 0:
                    u_element.append(u[dof])
                else:
                    u_element.append(0)  # DOF restringido

            # Convertir a array numpy
            u_element = np.array(u_element)

            # Transformar desplazamientos al sistema local si es necesario
            if hasattr(element, 'get_transformation_matrix'):
                T = element.get_transformation_matrix()
                u_local = T @ u_element
            else:
                u_local = u_element

            # Calcular fuerzas internas
            f_element = element.calculate_internal_forces(u_local)

            # Transformar fuerzas al sistema global si es necesario
            if hasattr(element, 'get_transformation_matrix'):
                f_element = T.T @ f_element

            # Ensamblar en el vector global
            for i, dof in enumerate(dof_indices):
                if dof >= 0:  # DOF no restringido
                    F_int[dof] += f_element[i]

        return F_int

    def _solve_generalized_eigenvalue_problem(self, A, B):
        """
        Resuelve el problema de autovalores generalizado Ax = lambda*Bx.

        Args:
            A: Primera matriz
            B: Segunda matriz

        Returns:
            eigenvalues: Vector de autovalores
            eigenvectors: Matriz de autovectores
        """
        try:
            # Resolver problema de autovalores generalizado
            eigenvalues, eigenvectors = scipy.linalg.eigh(A, B)

            # Ordenar por autovalores de menor a mayor
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Limitar el número de modos según las opciones
            num_modes = min(self.analysis_options.num_modes, len(eigenvalues))
            eigenvalues = eigenvalues[:num_modes]
            eigenvectors = eigenvectors[:, :num_modes]

            return eigenvalues, eigenvectors

        except Exception as e:
            raise RuntimeError(
                f"Error al resolver problema de autovalores: {e}")


class PlotterValues:
    """
    Clase para almacenar y procesar valores para visualización 
    de resultados de análisis estructural.
    """

    def __init__(self, system_elements, results=None):
        """
        Inicializa los valores para visualización.

        Args:
            system_elements: Objeto SystemElements con la estructura
            results: Objeto Results con resultados de análisis
        """
        self.system = system_elements
        self.results = results

        # Datos de la geometría
        self.node_coordinates = {}  # Coordenadas originales de los nodos
        self.element_connectivity = {}  # Conectividad de elementos
        self.supports = {}  # Nodos con apoyos y tipos

        # Datos para visualización de resultados
        self.deformed_coordinates = {}  # Coordenadas deformadas
        self.displacement_values = {}   # Valores de desplazamientos por nodo
        self.stress_values = {}         # Valores de tensiones por elemento
        self.strain_values = {}         # Valores de deformaciones por elemento
        self.internal_forces = {}       # Valores de esfuerzos internos por elemento

        # Datos para análisis modal/pandeo
        self.mode_shapes = []           # Formas modales
        self.frequencies = []           # Frecuencias naturales
        self.buckling_factors = []      # Factores de pandeo

        # Límites para colormaps
        self.contour_min_value = 0
        self.contour_max_value = 0

        # Metadatos para leyendas, títulos, etc.
        self.analysis_type = "Static"   # Tipo de análisis realizado
        self.load_case_name = "Default"  # Nombre del caso de carga
        self.result_type = ""           # Tipo de resultado visualizado
        self.result_component = ""      # Componente específica visualizada
        self.load_step = 0              # Paso de carga (análisis no lineal)

        # Extraer datos básicos del modelo
        self._extract_geometry()

        # Si hay resultados, procesarlos
        if results:
            self._process_results()

    def _extract_geometry(self):
        """
        Extrae información geométrica del modelo estructural.
        """
        # Extraer coordenadas de nodos
        for node_id, node in self.system.nodes.items():
            self.node_coordinates[node_id] = node.coordinates

        # Extraer conectividad de elementos
        for element_id, element in self.system.elements.items():
            self.element_connectivity[element_id] = element.nodes

        # Extraer información de apoyos
        for node_id, support in self.system.supports.items():
            self.supports[node_id] = support

    def _process_results(self):
        """
        Procesa los resultados para visualización.
        """
        if not self.results:
            return

        # Determinar tipo de análisis según los resultados disponibles
        if self.results.frequencies is not None:
            self.analysis_type = "Modal"
            self._process_modal_results()
        elif self.results.buckling_factors is not None:
            self.analysis_type = "Buckling"
            self._process_buckling_results()
        elif self.results.load_steps:
            self.analysis_type = "Nonlinear"
            self._process_nonlinear_results()
        else:
            self.analysis_type = "Static"
            self._process_static_results()

    def _process_static_results(self):
        """
        Procesa resultados para análisis estático.
        """
        # Procesar desplazamientos
        if self.results.displacements:
            for node_id, disp in self.results.displacements.items():
                # Almacenar magnitud de desplazamiento para contornos
                self.displacement_values[node_id] = np.linalg.norm(
                    disp[:3])  # Solo translaciones

                # Calcular coordenadas deformadas
                orig_coords = self.node_coordinates[node_id]
                self.deformed_coordinates[node_id] = [
                    orig_coords[0] + disp[0],
                    orig_coords[1] + disp[1],
                    orig_coords[2] + disp[2] if len(orig_coords) > 2 else 0
                ]

        # Procesar tensiones
        if self.results.stresses:
            self.stress_values = self.results.stresses

            # Calcular rangos para colormap
            all_values = []
            for element_id, stress in self.stress_values.items():
                if isinstance(stress, dict):  # Si es un diccionario con componentes
                    for component, value in stress.items():
                        if isinstance(value, (int, float, np.number)):
                            all_values.append(value)
                elif isinstance(stress, (list, np.ndarray)):  # Si es una lista/array
                    all_values.extend(
                        [v for v in stress if isinstance(v, (int, float, np.number))])

            if all_values:
                self.contour_min_value = min(all_values)
                self.contour_max_value = max(all_values)

        # Procesar esfuerzos internos
        if self.results.internal_forces:
            self.internal_forces = self.results.internal_forces

    def _process_modal_results(self):
        """
        Procesa resultados para análisis modal.
        """
        if self.results.frequencies is not None:
            self.frequencies = self.results.frequencies

        if self.results.mode_shapes:
            self.mode_shapes = self.results.mode_shapes

    def _process_buckling_results(self):
        """
        Procesa resultados para análisis de pandeo.
        """
        if self.results.buckling_factors is not None:
            self.buckling_factors = self.results.buckling_factors

        if self.results.buckling_modes:
            self.mode_shapes = self.results.buckling_modes

    def _process_nonlinear_results(self):
        """
        Procesa resultados para análisis no lineal.
        """
        # Tomar el último paso de carga por defecto
        self.load_step = len(self.results.load_steps) - 1

        # Procesar desplazamientos del último paso
        if self.results.displacement_history and self.load_step < len(self.results.displacement_history):
            displacements = self.results.displacement_history[self.load_step]

            for node_id, disp in displacements.items():
                # Almacenar magnitud de desplazamiento para contornos
                self.displacement_values[node_id] = np.linalg.norm(
                    disp[:3])  # Solo translaciones

                # Calcular coordenadas deformadas
                orig_coords = self.node_coordinates[node_id]
                self.deformed_coordinates[node_id] = [
                    orig_coords[0] + disp[0],
                    orig_coords[1] + disp[1],
                    orig_coords[2] + disp[2] if len(orig_coords) > 2 else 0
                ]

    def set_result_type(self, result_type, component=None):
        """
        Establece el tipo de resultado a visualizar.

        Args:
            result_type: Tipo de resultado ('displacement', 'stress', 'strain', 'internal_force')
            component: Componente específica a visualizar
        """
        self.result_type = result_type
        self.result_component = component

        # Actualizar valores y límites según el tipo de resultado
        if result_type == 'stress' and component:
            # Extraer valores de la componente específica de tensión
            values = {}
            for element_id, stress in self.stress_values.items():
                if isinstance(stress, dict) and component in stress:
                    values[element_id] = stress[component]
                elif isinstance(stress, (list, np.ndarray)) and len(stress) > 0:
                    # Asumiendo un orden específico de componentes
                    comp_index = {'xx': 0, 'yy': 1, 'zz': 2,
                                  'xy': 3, 'yz': 4, 'xz': 5}.get(component, 0)
                    if comp_index < len(stress):
                        values[element_id] = stress[comp_index]

            if values:
                self.contour_min_value = min(values.values())
                self.contour_max_value = max(values.values())

        elif result_type == 'displacement' and component:
            # Extraer valores de la componente específica de desplazamiento
            values = {}
            for node_id, disp in self.results.displacements.items():
                comp_index = {'ux': 0, 'uy': 1, 'uz': 2,
                              'rx': 3, 'ry': 4, 'rz': 5}.get(component, 0)
                if comp_index < len(disp):
                    values[node_id] = disp[comp_index]

            if values:
                self.contour_min_value = min(values.values())
                self.contour_max_value = max(values.values())

    def set_load_step(self, step):
        """
        Establece el paso de carga para visualización de resultados no lineales.

        Args:
            step: Índice del paso de carga
        """
        if self.analysis_type == "Nonlinear" and step < len(self.results.load_steps):
            self.load_step = step

            # Actualizar datos de desplazamientos
            if self.results.displacement_history and self.load_step < len(self.results.displacement_history):
                displacements = self.results.displacement_history[self.load_step]

                self.deformed_coordinates = {}
                self.displacement_values = {}

                for node_id, disp in displacements.items():
                    # Almacenar magnitud de desplazamiento para contornos
                    self.displacement_values[node_id] = np.linalg.norm(
                        disp[:3])  # Solo translaciones

                    # Calcular coordenadas deformadas
                    orig_coords = self.node_coordinates[node_id]
                    self.deformed_coordinates[node_id] = [
                        orig_coords[0] + disp[0],
                        orig_coords[1] + disp[1],
                        orig_coords[2] + disp[2] if len(orig_coords) > 2 else 0
                    ]

    def set_mode_number(self, mode_num):
        """
        Establece el número de modo para visualización de resultados modales/pandeo.

        Args:
            mode_num: Número de modo a visualizar
        """
        if (self.analysis_type in ["Modal", "Buckling"] and
                mode_num < len(self.mode_shapes)):

            # Actualizar deformaciones para el modo seleccionado
            mode_shape = self.mode_shapes[mode_num]

            self.deformed_coordinates = {}
            self.displacement_values = {}

            # Factor de escala para normalizar el modo
            max_disp = 0
            for node_id, modal_disp in mode_shape.items():
                disp_magnitude = np.linalg.norm(
                    modal_disp[:3])  # Solo translaciones
                max_disp = max(max_disp, disp_magnitude)

            scale = 1.0 / max_disp if max_disp > 0 else 1.0

            for node_id, modal_disp in mode_shape.items():
                # Almacenar magnitud de desplazamiento para contornos
                self.displacement_values[node_id] = np.linalg.norm(
                    modal_disp[:3])  # Solo translaciones

                # Calcular coordenadas deformadas
                orig_coords = self.node_coordinates[node_id]
                self.deformed_coordinates[node_id] = [
                    orig_coords[0] + modal_disp[0] * scale,
                    orig_coords[1] + modal_disp[1] * scale,
                    orig_coords[2] + modal_disp[2] *
                    scale if len(orig_coords) > 2 else 0
                ]

            # Actualizar metadatos
            if self.analysis_type == "Modal" and self.frequencies:
                self.result_type = f"Mode {mode_num+1} (f={self.frequencies[mode_num]:.2f} Hz)"
            elif self.analysis_type == "Buckling" and self.buckling_factors:
                self.result_type = f"Buckling Mode {mode_num+1} (λ={self.buckling_factors[mode_num]:.2f})"

    def get_global_min_max(self):
        """
        Obtiene valores mínimo y máximo globales para el tipo de resultado actual.

        Returns:
            Tupla con (valor_mínimo, valor_máximo)
        """
        return self.contour_min_value, self.contour_max_value


class Plotter:
    """
    Clase para visualizar resultados de análisis estructural.
    """

    def __init__(self, system_elements, results=None, options=None):
        """
        Inicializa el objeto para visualización.

        Args:
            system_elements: Objeto SystemElements con la estructura
            results: Objeto Results con resultados de análisis
            options: Objeto PlotterOptions con opciones de visualización
        """
        self.system = system_elements
        self.values = PlotterValues(system_elements, results)
        self.options = options if options else PlotterOptions()

        # Atributos para figuras y ejes
        self.fig = None
        self.ax = None
        self.colorbar = None
        self.animation = None

        # Determinar dimensionalidad del problema
        self.dimension = self._determine_dimension()

        # Inicializar figura si se proporcionan resultados
        if results:
            self.initialize_figure()

    def _determine_dimension(self):
        """
        Determina si el modelo es 2D o 3D.

        Returns:
            Dimensión del modelo (2 o 3)
        """
        # Verificar si algún nodo tiene coordenada z no nula
        for node_id, coords in self.values.node_coordinates.items():
            if len(coords) > 2 and abs(coords[2]) > 1e-6:
                return 3

        return 2

    def initialize_figure(self):
        """
        Inicializa la figura para visualización.
        """
        # Configurar estilo global
        if self.options.plot_style in plt.style.available:
            plt.style.use(self.options.plot_style)

        # Crear figura
        self.fig = plt.figure(figsize=(self.options.fig_width, self.options.fig_height),
                              dpi=self.options.dpi,
                              facecolor=self.options.background_color)

        # Crear eje según dimensionalidad
        if self.dimension == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        # Configurar ejes
        if self.options.grid:
            self.ax.grid(True, linestyle='--', alpha=0.5)

        self.ax.set_xlabel('X', fontsize=self.options.axis_font_size)
        self.ax.set_ylabel('Y', fontsize=self.options.axis_font_size)

        if self.dimension == 3:
            self.ax.set_zlabel('Z', fontsize=self.options.axis_font_size)

        # Ajustar layout
        if self.options.tight_layout:
            self.fig.tight_layout()

    def plot_structure(self):
        """
        Visualiza la estructura sin resultados.
        """
        if self.fig is None:
            self.initialize_figure()

        # Limpiar ejes
        self.ax.clear()

        # Dibujar elementos
        self._plot_elements()

        # Dibujar nodos
        self._plot_nodes()

        # Dibujar apoyos
        self._plot_supports()

        # Ajustar visualización
        self._set_plot_limits()

        # Añadir título
        self.ax.set_title('Estructura', fontsize=self.options.title_font_size)

        # Mostrar etiquetas si se requiere
        if self.options.node_labels:
            self._add_node_labels()

        if self.options.element_labels:
            self._add_element_labels()

        # Ajustar layout
        if self.options.tight_layout:
            self.fig.tight_layout()

        return self.fig

    def _plot_nodes(self):
        """
        Dibuja los nodos de la estructura.
        """
        if self.dimension == 3:
            x = [coords[0] for coords in self.values.node_coordinates.values()]
            y = [coords[1] for coords in self.values.node_coordinates.values()]
            z = [coords[2] if len(
                coords) > 2 else 0 for coords in self.values.node_coordinates.values()]

            self.ax.scatter(x, y, z, c=self.options.node_color,
                            s=self.options.node_size, marker='o')
        else:
            x = [coords[0] for coords in self.values.node_coordinates.values()]
            y = [coords[1] for coords in self.values.node_coordinates.values()]

            self.ax.scatter(x, y, c=self.options.node_color,
                            s=self.options.node_size, marker='o')

    def _plot_elements(self):
        """
        Dibuja los elementos de la estructura.
        """
        for element_id, nodes in self.values.element_connectivity.items():
            # Obtener coordenadas de los nodos del elemento
            coords = [self.values.node_coordinates[node_id]
                      for node_id in nodes]

            # Dibujar elemento según su dimensionalidad
            if self.dimension == 3:
                x = [coord[0] for coord in coords]
                y = [coord[1] for coord in coords]
                z = [coord[2] if len(coord) > 2 else 0 for coord in coords]

                self.ax.plot(x, y, z, color=self.options.element_color,
                             linewidth=self.options.element_line_width)
            else:
                x = [coord[0] for coord in coords]
                y = [coord[1] for coord in coords]

                self.ax.plot(x, y, color=self.options.element_color,
                             linewidth=self.options.element_line_width)

# Continuación de la clase Plotter


    def _plot_supports(self):
        """
        Dibuja los apoyos de la estructura.
        """
        for node_id, support in self.values.supports.items():
            # Obtener coordenadas del nodo
            coords = self.values.node_coordinates[node_id]

            # Dibujar apoyo según su tipo
            if self.dimension == 3:
                x, y, z = coords[0], coords[1], coords[2] if len(coords) > 2 else 0

                # Dibujar símbolo de apoyo según restricciones
                # Esto es simplificado; en una implementación real, se dibujarían
                # símbolos distintos según el tipo de apoyo
                self.ax.scatter([x], [y], [z], marker='s',
                                c='green', s=self.options.support_size)
            else:
                x, y = coords[0], coords[1]

                # Determinar tipo de apoyo y dibujar símbolo apropiado
                if support.is_fixed():
                    # Apoyo empotrado (triángulo)
                    marker = '^'
                    color = 'darkred'
                elif support.is_pinned():
                    # Apoyo articulado (círculo)
                    marker = 'o'
                    color = 'darkgreen'
                elif support.is_roller():
                    # Apoyo deslizante (triángulo invertido)
                    marker = 'v'
                    color = 'darkblue'
                else:
                    # Apoyo genérico (cuadrado)
                    marker = 's'
                    color = 'orange'

                self.ax.scatter([x], [y], marker=marker,
                                c=color, s=self.options.support_size,
                                edgecolors='black', zorder=3)


    def _add_node_labels(self):
        """
        Añade etiquetas a los nodos.
        """
        for node_id, coords in self.values.node_coordinates.items():
            if self.dimension == 3:
                x, y, z = coords[0], coords[1], coords[2] if len(coords) > 2 else 0
                self.ax.text(x, y, z, f"{node_id}", fontsize=self.options.label_font_size,
                            ha='center', va='bottom', color='black', zorder=4)
            else:
                x, y = coords[0], coords[1]
                self.ax.text(x, y, f"{node_id}", fontsize=self.options.label_font_size,
                            ha='center', va='bottom', color='black', zorder=4)


    def _add_element_labels(self):
        """
        Añade etiquetas a los elementos.
        """
        for element_id, nodes in self.values.element_connectivity.items():
            # Obtener coordenadas de los nodos del elemento
            coords = [self.values.node_coordinates[node_id] for node_id in nodes]

            # Calcular centro del elemento para colocar la etiqueta
            if len(coords) == 2:  # Elemento lineal
                # Para un elemento de 2 nodos, el centro es el punto medio
                mid_x = (coords[0][0] + coords[1][0]) / 2
                mid_y = (coords[0][1] + coords[1][1]) / 2
                if self.dimension == 3:
                    mid_z = (coords[0][2] + coords[1][2]) / \
                        2 if len(coords[0]) > 2 and len(coords[1]) > 2 else 0
                    self.ax.text(mid_x, mid_y, mid_z, f"{element_id}", fontsize=self.options.label_font_size,
                                ha='center', va='bottom', color='darkblue', zorder=4)
                else:
                    self.ax.text(mid_x, mid_y, f"{element_id}", fontsize=self.options.label_font_size,
                                ha='center', va='bottom', color='darkblue', zorder=4)
            else:  # Elemento con más nodos (triángulos, cuadriláteros, etc.)
                # Calcular centroide
                n = len(coords)
                mid_x = sum(c[0] for c in coords) / n
                mid_y = sum(c[1] for c in coords) / n
                if self.dimension == 3:
                    mid_z = sum(c[2] if len(c) > 2 else 0 for c in coords) / n
                    self.ax.text(mid_x, mid_y, mid_z, f"{element_id}", fontsize=self.options.label_font_size,
                                ha='center', va='center', color='darkblue', zorder=4)
                else:
                    self.ax.text(mid_x, mid_y, f"{element_id}", fontsize=self.options.label_font_size,
                                ha='center', va='center', color='darkblue', zorder=4)


    def _set_plot_limits(self):
        """
        Establece los límites del gráfico para visualizar toda la estructura.
        """
        # Recopilar todas las coordenadas
        all_x = []
        all_y = []
        all_z = []

        for coords in self.values.node_coordinates.values():
            all_x.append(coords[0])
            all_y.append(coords[1])
            if len(coords) > 2:
                all_z.append(coords[2])

        # Si hay coordenadas deformadas, incluirlas también
        if self.values.deformed_coordinates:
            for coords in self.values.deformed_coordinates.values():
                all_x.append(coords[0])
                all_y.append(coords[1])
                if len(coords) > 2:
                    all_z.append(coords[2])

        # Calcular límites con un margen
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)

            # Añadir margen de 10%
            # Al menos 1 unidad de margen
            x_margin = max(0.1 * (x_max - x_min), 1.0)
            y_margin = max(0.1 * (y_max - y_min), 1.0)

            # Establecer límites
            self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
            self.ax.set_ylim(y_min - y_margin, y_max + y_margin)

            # Si es 3D, configurar límites en z
            if self.dimension == 3 and all_z:
                z_min, z_max = min(all_z), max(all_z)
                z_margin = max(0.1 * (z_max - z_min), 1.0)
                self.ax.set_zlim(z_min - z_margin, z_max + z_margin)

        # Establecer aspecto igual para visualización proporcional
        if self.dimension == 2:
            self.ax.set_aspect('equal')


    def plot_deformed(self, scale=None):
        """
        Visualiza la estructura deformada.

        Args:
            scale: Factor de escala para deformaciones (opcional, usa el valor de options si no se proporciona)
        """
        if self.fig is None:
            self.initialize_figure()

        # Usar escala proporcionada o la de las opciones
        deformation_scale = scale if scale is not None else self.options.deformation_scale

        # Limpiar ejes
        self.ax.clear()

        # Dibujar estructura original si se requiere
        if self.options.show_undeformed:
            self._plot_elements()

            # Cambiar estilo para distinguirla
            for line in self.ax.lines:
                line.set_color(self.options.undeformed_color)
                line.set_linestyle(self.options.undeformed_style)
                line.set_alpha(0.5)
                line.set_zorder(1)

        # Dibujar estructura deformada
        if self.values.deformed_coordinates:
            for element_id, nodes in self.values.element_connectivity.items():
                # Obtener coordenadas deformadas de los nodos del elemento
                coords = []
                for node_id in nodes:
                    if node_id in self.values.deformed_coordinates:
                        coords.append(self.values.deformed_coordinates[node_id])
                    else:  # Usar original si no hay deformación
                        orig = self.values.node_coordinates[node_id]
                        coords.append(orig)

                # Dibujar elemento deformado
                if self.dimension == 3:
                    x = [coord[0] for coord in coords]
                    y = [coord[1] for coord in coords]
                    z = [coord[2] if len(coord) > 2 else 0 for coord in coords]

                    self.ax.plot(x, y, z, color='red',
                                linewidth=self.options.element_line_width,
                                zorder=2)
                else:
                    x = [coord[0] for coord in coords]
                    y = [coord[1] for coord in coords]

                    self.ax.plot(x, y, color='red',
                                linewidth=self.options.element_line_width,
                                zorder=2)

            # Dibujar nodos deformados
            if self.dimension == 3:
                x = [coords[0]
                    for coords in self.values.deformed_coordinates.values()]
                y = [coords[1]
                    for coords in self.values.deformed_coordinates.values()]
                z = [coords[2] if len(
                    coords) > 2 else 0 for coords in self.values.deformed_coordinates.values()]

                self.ax.scatter(x, y, z, c='red',
                                s=self.options.node_size, marker='o', zorder=3)
            else:
                x = [coords[0]
                    for coords in self.values.deformed_coordinates.values()]
                y = [coords[1]
                    for coords in self.values.deformed_coordinates.values()]

                self.ax.scatter(
                    x, y, c='red', s=self.options.node_size, marker='o', zorder=3)

        # Dibujar apoyos
        self._plot_supports()

        # Ajustar visualización
        self._set_plot_limits()

        # Añadir título
        scale_text = f" (Escala x{deformation_scale})" if deformation_scale != 1.0 else ""
        self.ax.set_title(
            f'Estructura Deformada{scale_text}', fontsize=self.options.title_font_size)

        # Mostrar etiquetas si se requiere
        if self.options.node_labels:
            self._add_node_labels()

        if self.options.element_labels:
            self._add_element_labels()

        # Ajustar layout
        if self.options.tight_layout:
            self.fig.tight_layout()

        return self.fig


    def plot_contour(self, result_type='displacement', component=None):
        """
        Visualiza resultados mediante contornos de color.

        Args:
            result_type: Tipo de resultado ('displacement', 'stress', 'strain')
            component: Componente específica a visualizar
        """
        if self.fig is None:
            self.initialize_figure()

        # Establecer tipo de resultado en valores
        self.values.set_result_type(result_type, component)

        # Limpiar ejes
        self.ax.clear()

        # Determinar si visualizamos deformada o indeformada
        use_deformed = self.values.deformed_coordinates and result_type != 'mode'

        # Preparar datos según tipo de resultado
        if result_type == 'displacement':
            self._plot_displacement_contour(component, use_deformed)
        elif result_type == 'stress':
            self._plot_stress_contour(component, use_deformed)
        elif result_type == 'strain':
            self._plot_strain_contour(component, use_deformed)
        elif result_type == 'mode':
            self._plot_mode_shape(component)

        # Dibujar apoyos
        self._plot_supports()

        # Ajustar visualización
        self._set_plot_limits()

        # Añadir título
        component_text = f" - {component}" if component else ""
        self.ax.set_title(f'{result_type.capitalize()}{component_text}',
                        fontsize=self.options.title_font_size)

        # Mostrar etiquetas si se requiere
        if self.options.node_labels:
            self._add_node_labels()

        if self.options.element_labels:
            self._add_element_labels()

        # Ajustar layout
        if self.options.tight_layout:
            self.fig.tight_layout()

        return self.fig


    def _plot_displacement_contour(self, component=None, use_deformed=True):
        """
        Visualiza contorno de desplazamientos.

        Args:
            component: Componente de desplazamiento específica
            use_deformed: Si se usa la geometría deformada
        """
        # Obtener coordenadas y valores
        coords = self.values.deformed_coordinates if use_deformed else self.values.node_coordinates

        # Recopilar coordenadas y valores para triangulación
        points = []
        values = []
        node_ids = []

        for node_id, coord in coords.items():
            if node_id in self.values.displacement_values:
                points.append([coord[0], coord[1]])
                values.append(self.values.displacement_values[node_id])
                node_ids.append(node_id)

        if not points:
            return

        # Convertir a arrays numpy
        points = np.array(points)
        values = np.array(values)

        # Crear triangulación (para elementos 2D)
        if self.dimension == 2:
            # Crear triangulación a partir de elementos
            triangles = []
            for element_id, nodes in self.values.element_connectivity.items():
                if len(nodes) == 3:  # Elemento triangular
                    # Verificar que todos los nodos tienen valores
                    if all(node_id in node_ids for node_id in nodes):
                        # Convertir IDs de nodos a índices en la lista node_ids
                        indices = [node_ids.index(node_id) for node_id in nodes]
                        triangles.append(indices)
                elif len(nodes) == 4:  # Elemento cuadrilátero, dividir en 2 triángulos
                    if all(node_id in node_ids for node_id in nodes):
                        indices = [node_ids.index(node_id) for node_id in nodes]
                        # Dividir cuadrilátero en dos triángulos
                        triangles.append([indices[0], indices[1], indices[2]])
                        triangles.append([indices[0], indices[2], indices[3]])

            if triangles:
                # Crear objeto de triangulación
                triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

                # Colormap
                cmap = plt.cm.get_cmap(self.options.colormap)
                if self.options.colormap_reverse:
                    cmap = plt.cm.get_cmap(self.options.colormap + '_r')

                # Dibujar contorno
                if self.options.contour_type in ['filled', 'both']:
                    contour = self.ax.tripcolor(triang, values, cmap=cmap,
                                                alpha=self.options.alpha,
                                                edgecolors=self.options.edge_color if self.options.contour_type == 'both' else None,
                                                shading='gouraud')
                else:  # 'lines'
                    contour = self.ax.tricontour(triang, values,
                                                self.options.num_contours,
                                                cmap=cmap,
                                                linewidths=1.0)

                # Colorbar
                if self.options.show_colorbar:
                    cbar_label = self.options.colorbar_label
                    if not cbar_label and component:
                        cbar_label = f"Desplazamiento {component}"
                    elif not cbar_label:
                        cbar_label = "Magnitud de desplazamiento"

                    self.colorbar = self.fig.colorbar(
                        contour, ax=self.ax, label=cbar_label)
                    self.colorbar.ax.tick_params(
                        labelsize=self.options.legend_font_size)
                    self.colorbar.set_label(
                        cbar_label, size=self.options.legend_font_size)
            else:
                # Si no hay triangulación, dibujar nodos con colores
                scatter = self.ax.scatter(points[:, 0], points[:, 1], c=values,
                                        cmap=self.options.colormap,
                                        s=50,
                                        edgecolor='black')

                # Colorbar
                if self.options.show_colorbar:
                    cbar_label = self.options.colorbar_label
                    if not cbar_label and component:
                        cbar_label = f"Desplazamiento {component}"
                    elif not cbar_label:
                        cbar_label = "Magnitud de desplazamiento"

                    self.colorbar = self.fig.colorbar(
                        scatter, ax=self.ax, label=cbar_label)
                    self.colorbar.ax.tick_params(
                        labelsize=self.options.legend_font_size)
                    self.colorbar.set_label(
                        cbar_label, size=self.options.legend_font_size)
        else:
            # Para 3D, usamos scatter con colores
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z = [coords[node_ids[i]][2] if len(
                coords[node_ids[i]]) > 2 else 0 for i in range(len(points))]

            scatter = self.ax.scatter(x, y, z, c=values,
                                    cmap=self.options.colormap,
                                    s=50,
                                    edgecolor='black')

            # Colorbar
            if self.options.show_colorbar:
                cbar_label = self.options.colorbar_label
                if not cbar_label and component:
                    cbar_label = f"Desplazamiento {component}"
                elif not cbar_label:
                    cbar_label = "Magnitud de desplazamiento"

                self.colorbar = self.fig.colorbar(
                    scatter, ax=self.ax, label=cbar_label)
                self.colorbar.ax.tick_params(
                    labelsize=self.options.legend_font_size)
                self.colorbar.set_label(
                    cbar_label, size=self.options.legend_font_size)


    def _plot_stress_contour(self, component=None, use_deformed=True):
        """
        Visualiza contorno de tensiones.

        Args:
            component: Componente de tensión específica
            use_deformed: Si se usa la geometría deformada
        """
        if not self.values.stress_values:
            return

        # Obtener coordenadas
        coords = self.values.deformed_coordinates if use_deformed else self.values.node_coordinates

        # Crear triangulación (para elementos 2D)
        if self.dimension == 2:
            # Recopilar elementos triangulares o convertir a triangulares
            triangles = []
            element_values = []

            for element_id, nodes in self.values.element_connectivity.items():
                if element_id in self.values.stress_values:
                    # Obtener valor de tensión para este elemento
                    stress_value = 0

                    if component and isinstance(self.values.stress_values[element_id], dict):
                        if component in self.values.stress_values[element_id]:
                            stress_value = self.values.stress_values[element_id][component]
                    elif isinstance(self.values.stress_values[element_id], (int, float, np.number)):
                        stress_value = self.values.stress_values[element_id]
                    elif isinstance(self.values.stress_values[element_id], (list, np.ndarray)) and component:
                        # Asumiendo un orden específico de componentes
                        comp_index = {'xx': 0, 'yy': 1, 'zz': 2,
                                    'xy': 3, 'yz': 4, 'xz': 5}.get(component, 0)
                        if comp_index < len(self.values.stress_values[element_id]):
                            stress_value = self.values.stress_values[element_id][comp_index]

                    # Crear patches para los elementos
                    if len(nodes) == 3:  # Elemento triangular
                        if all(node_id in coords for node_id in nodes):
                            # Obtener coordenadas de los vértices
                            triangle_coords = [coords[node_id]
                                            for node_id in nodes]
                            xy = [(c[0], c[1]) for c in triangle_coords]

                            # Crear polígono
                            triangle = Polygon(xy, closed=True)
                            triangles.append(triangle)
                            element_values.append(stress_value)

                    elif len(nodes) == 4:  # Elemento cuadrilátero
                        if all(node_id in coords for node_id in nodes):
                            # Obtener coordenadas de los vértices
                            quad_coords = [coords[node_id] for node_id in nodes]
                            xy = [(c[0], c[1]) for c in quad_coords]

                            # Crear polígono
                            quad = Polygon(xy, closed=True)
                            triangles.append(quad)
                            element_values.append(stress_value)

            if triangles:
                # Normalizar valores para colormap
                min_val, max_val = min(element_values), max(element_values)
                norm = plt.Normalize(min_val, max_val)

                # Colormap
                cmap = plt.cm.get_cmap(self.options.colormap)
                if self.options.colormap_reverse:
                    cmap = plt.cm.get_cmap(self.options.colormap + '_r')

                # Crear colección de polígonos
                collection = PolyCollection(
                    [p.get_xy() for p in triangles],
                    edgecolor=self.options.edge_color,
                    linewidth=0.5,
                    cmap=cmap,
                    norm=norm,
                    alpha=self.options.alpha
                )
                collection.set_array(np.array(element_values))

                # Añadir a ejes
                self.ax.add_collection(collection)

                # Colorbar
                if self.options.show_colorbar:
                    cbar_label = self.options.colorbar_label
                    if not cbar_label and component:
                        cbar_label = f"Tensión {component}"
                    elif not cbar_label:
                        cbar_label = "Tensión"

                    self.colorbar = self.fig.colorbar(
                        collection, ax=self.ax, label=cbar_label)
                    self.colorbar.ax.tick_params(
                        labelsize=self.options.legend_font_size)
                    self.colorbar.set_label(
                        cbar_label, size=self.options.legend_font_size)
        else:
            # Para 3D, usamos scatter con colores en los centroides de elementos
            centroids = []
            element_values = []

            for element_id, nodes in self.values.element_connectivity.items():
                if element_id in self.values.stress_values and all(node_id in coords for node_id in nodes):
                    # Calcular centroide
                    x_sum = sum(coords[node_id][0] for node_id in nodes)
                    y_sum = sum(coords[node_id][1] for node_id in nodes)
                    z_sum = sum(coords[node_id][2] if len(
                        coords[node_id]) > 2 else 0 for node_id in nodes)

                    n = len(nodes)
                    centroid = [x_sum/n, y_sum/n, z_sum/n]
                    centroids.append(centroid)

                    # Obtener valor de tensión
                    stress_value = 0

                    if component and isinstance(self.values.stress_values[element_id], dict):
                        if component in self.values.stress_values[element_id]:
                            stress_value = self.values.stress_values[element_id][component]
                    elif isinstance(self.values.stress_values[element_id], (int, float, np.number)):
                        stress_value = self.values.stress_values[element_id]

                    element_values.append(stress_value)

            if centroids:
                x = [c[0] for c in centroids]
                y = [c[1] for c in centroids]
                z = [c[2] for c in centroids]

                scatter = self.ax.scatter(x, y, z, c=element_values,
                                        cmap=self.options.colormap,
                                        s=100,
                                        alpha=0.7,
                                        edgecolor='black')

                # Colorbar
                if self.options.show_colorbar:
                    cbar_label = self.options.colorbar_label
                    if not cbar_label and component:
                        cbar_label = f"Tensión {component}"
                    elif not cbar_label:
                        cbar_label = "Tensión"

                    self.colorbar = self.fig.colorbar(
                        scatter, ax=self.ax, label=cbar_label)
                    self.colorbar.ax.tick_params(
                        labelsize=self.options.legend_font_size)
                    self.colorbar.set_label(
                        cbar_label, size=self.options.legend_font_size)


    def _plot_strain_contour(self, component=None, use_deformed=True):
        """
        Visualiza contorno de deformaciones unitarias.
        Similar a _plot_stress_contour pero con valores de deformación.
        """
        # Implementación similar a _plot_stress_contour pero usando self.values.strain_values
        # Este método se implementaría de manera similar al de tensiones
        pass


    def _plot_mode_shape(self, mode_num=0):
        """
        Visualiza forma modal.

        Args:
            mode_num: Número de modo a visualizar
        """
        # Establecer modo en valores
        self.values.set_mode_number(int(mode_num) if mode_num is not None else 0)

        # Dibujar estructura deformada (modo)
        if self.values.deformed_coordinates:
            for element_id, nodes in self.values.element_connectivity.items():
                # Obtener coordenadas deformadas de los nodos del elemento
                coords = []
                for node_id in nodes:
                    if node_id in self.values.deformed_coordinates:
                        coords.append(self.values.deformed_coordinates[node_id])
                    else:  # Usar original si no hay deformación
                        orig = self.values.node_coordinates[node_id]
                        coords.append(orig)

                # Dibujar elemento deformado
                if self.dimension == 3:
                    x = [coord[0] for coord in coords]
                    y = [coord[1] for coord in coords]
                    z = [coord[2] if len(coord) > 2 else 0 for coord in coords]

                    self.ax.plot(x, y, z, color=self.options.mode_color,
                                linewidth=self.options.element_line_width * 1.5)
                else:
                    x = [coord[0] for coord in coords]
                    y = [coord[1] for coord in coords]

                    self.ax.plot(x, y, color=self.options.mode_color,
                                linewidth=self.options.element_line_width * 1.5)

            # Dibujar nodos deformados
            if self.dimension == 3:
                x = [coords[0]
                    for coords in self.values.deformed_coordinates.values()]
                y = [coords[1]
                    for coords in self.values.deformed_coordinates.values()]
                z = [coords[2] if len(
                    coords) > 2 else 0 for coords in self.values.deformed_coordinates.values()]

                self.ax.scatter(x, y, z, c=self.options.mode_color,
                                s=self.options.node_size * 1.5, marker='o')
            else:
                x = [coords[0]
                    for coords in self.values.deformed_coordinates.values()]
                y = [coords[1]
                    for coords in self.values.deformed_coordinates.values()]

                self.ax.scatter(x, y, c=self.options.mode_color,
                                s=self.options.node_size * 1.5, marker='o')

        # Dibujar estructura original en fondo
        if self.options.show_undeformed:
            # Dibujar elementos originales
            for element_id, nodes in self.values.element_connectivity.items():
                # Obtener coordenadas originales de los nodos del elemento
                coords = [self.values.node_coordinates[node_id]
                        for node_id in nodes]

                # Dibujar elemento original
                if self.dimension == 3:
                    x = [coord[0] for coord in coords]
                    y = [coord[1] for coord in coords]
                    z = [coord[2] if len(coord) > 2 else 0 for coord in coords]

                    self.ax.plot(x, y, z, color=self.options.undeformed_color,
                                linestyle=self.options.undeformed_style,
                                linewidth=self.options.element_line_width * 0.8,
                                alpha=0.5)
                else:
                    x = [coord[0] for coord in coords]
                    y = [coord[1] for coord in coords]

                    self.ax.plot(x, y, color=self.options.undeformed_color,
                                linestyle=self.options.undeformed_style,
                                linewidth=self.options.element_line_width * 0.8,
                                alpha=0.5)

            # Dibujar nodos originales
            if self.dimension == 3:
                x = [coords[0] for coords in self.values.node_coordinates.values()]
                y = [coords[1] for coords in self.values.node_coordinates.values()]
                z = [coords[2] if len(
                    coords) > 2 else 0 for coords in self.values.node_coordinates.values()]

                self.ax.scatter(x, y, z, c=self.options.undeformed_color,
                                s=self.options.node_size, marker='o', alpha=0.5)
            else:
                x = [coords[0] for coords in self.values.node_coordinates.values()]
                y = [coords[1] for coords in self.values.node_coordinates.values()]

                self.ax.scatter(x, y, c=self.options.undeformed_color,
                                s=self.options.node_size, marker='o', alpha=0.5)

        # Ajustar visualización
        self._set_plot_limits()

        # Añadir título
        self.ax.set_title(f'Modo {mode_num}',
                        fontsize=self.options.title_font_size)

        # Mostrar etiquetas si se requiere
        if self.options.node_labels:
            self._add_node_labels()

        if self.options.element_labels:
            self._add_element_labels()

        # Ajustar layout
        if self.options.tight_layout:
            self.fig.tight_layout()

        return self.fig

import numpy as np
from enum import Enum

class MaterialType(Enum):
    """Enumeración para los diferentes tipos de materiales."""
    ISOTROPIC = "isotropic"
    ORTHOTROPIC = "orthotropic"
    ANISOTROPIC = "anisotropic"

class Material:
    """
    Clase para representar las propiedades del material.
    Permite definir materiales con diferentes características para el análisis estructural.
    """
    def __init__(self, material_id, name=None, material_type=MaterialType.ISOTROPIC, **kwargs):
        """
        Constructor para el material.
        
        Parámetros:
        -----------
        material_id : str o int
            Identificador único del material
        name : str, opcional
            Nombre descriptivo del material
        material_type : MaterialType, opcional
            Tipo de material (isotropic, orthotropic, anisotropic)
        **kwargs : dict
            Propiedades del material dependiendo del tipo:
            - Para ISOTROPIC: E, nu, rho, alpha, G (opcional)
            - Para ORTHOTROPIC: E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, rho, alpha1, alpha2, alpha3
            - Para ANISOTROPIC: matriz de rigidez C directamente
        """
        self.material_id = material_id
        self.name = name if name is not None else f"Material-{material_id}"
        self.material_type = material_type
        
        # Inicializar propiedades estándar
        self.rho = kwargs.get('rho', 0.0)  # Densidad
        self.alpha = kwargs.get('alpha', 0.0)  # Coeficiente de expansión térmica (para isotropic)
        
        # Inicializar propiedades según el tipo de material
        if material_type == MaterialType.ISOTROPIC:
            self._init_isotropic(**kwargs)
        elif material_type == MaterialType.ORTHOTROPIC:
            self._init_orthotropic(**kwargs)
        elif material_type == MaterialType.ANISOTROPIC:
            self._init_anisotropic(**kwargs)
        else:
            raise ValueError(f"Tipo de material no reconocido: {material_type}")
            
    def _init_isotropic(self, **kwargs):
        """Inicializa propiedades para material isótropo."""
        # Propiedades obligatorias
        if 'E' not in kwargs:
            raise ValueError("Se requiere el módulo de elasticidad (E) para materiales isótropos")
        
        self.E = kwargs.get('E')  # Módulo de elasticidad
        self.nu = kwargs.get('nu', 0.0)  # Coeficiente de Poisson
        
        # Módulo de corte (G), calculado si no se proporciona
        if 'G' in kwargs:
            self.G = kwargs.get('G')
        else:
            self.G = self.E / (2 * (1 + self.nu))
    
    def _init_orthotropic(self, **kwargs):
        """Inicializa propiedades para material ortótropo."""
        required_props = ['E1', 'E2', 'E3', 'nu12', 'nu13', 'nu23', 'G12', 'G13', 'G23']
        for prop in required_props:
            if prop not in kwargs:
                raise ValueError(f"Se requiere la propiedad {prop} para materiales ortótropos")
        
        # Módulos de elasticidad en las tres direcciones principales
        self.E1 = kwargs.get('E1')
        self.E2 = kwargs.get('E2')
        self.E3 = kwargs.get('E3')
        
        # Coeficientes de Poisson
        self.nu12 = kwargs.get('nu12')
        self.nu13 = kwargs.get('nu13')
        self.nu23 = kwargs.get('nu23')
        
        # Módulos de corte
        self.G12 = kwargs.get('G12')
        self.G13 = kwargs.get('G13')
        self.G23 = kwargs.get('G23')
        
        # Coeficientes de expansión térmica
        self.alpha1 = kwargs.get('alpha1', 0.0)
        self.alpha2 = kwargs.get('alpha2', 0.0)
        self.alpha3 = kwargs.get('alpha3', 0.0)
        
        # Calculamos los coeficientes de Poisson recíprocos por relaciones de simetría
        self.nu21 = self.nu12 * self.E2 / self.E1
        self.nu31 = self.nu13 * self.E3 / self.E1
        self.nu32 = self.nu23 * self.E3 / self.E2
    
    def _init_anisotropic(self, **kwargs):
        """Inicializa propiedades para material anisótropo."""
        if 'C' not in kwargs:
            raise ValueError("Se requiere la matriz de rigidez (C) para materiales anisótropos")
        
        self.C = kwargs.get('C')  # Matriz de rigidez
        
        # Coeficientes de expansión térmica
        if 'alpha_vector' in kwargs:
            self.alpha_vector = kwargs.get('alpha_vector')
        else:
            self.alpha_vector = np.zeros(6)  # Vector para expansión térmica [a11, a22, a33, a12, a13, a23]
    
    def get_stiffness_matrix(self):
        """
        Obtiene la matriz de rigidez del material.
        
        Retorna:
        --------
        numpy.ndarray
            Matriz de rigidez (6x6 para 3D)
        """
        if self.material_type == MaterialType.ISOTROPIC:
            return self._get_isotropic_stiffness_matrix()
        elif self.material_type == MaterialType.ORTHOTROPIC:
            return self._get_orthotropic_stiffness_matrix()
        elif self.material_type == MaterialType.ANISOTROPIC:
            return self.C
        else:
            raise ValueError(f"Tipo de material no soportado: {self.material_type}")
    
    def _get_isotropic_stiffness_matrix(self):
        """
        Calcula la matriz de rigidez para material isótropo.
        
        Retorna:
        --------
        numpy.ndarray
            Matriz de rigidez 6x6 para 3D
        """
        # Factores para matriz de rigidez
        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        
        # Construir matriz
        C = np.zeros((6, 6))
        
        # Términos diagonales
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu  # términos normales
        C[3, 3] = C[4, 4] = C[5, 5] = mu  # términos cortantes
        
        # Términos fuera de la diagonal (solo términos de acoplamiento normal-normal)
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
        
        return C
    
    def _get_orthotropic_stiffness_matrix(self):
        """
        Calcula la matriz de rigidez para material ortótropo.
        
        Retorna:
        --------
        numpy.ndarray
            Matriz de rigidez 6x6 para 3D
        """
        # Cálculo de denominador para términos de la matriz
        denom = (1 - self.nu12 * self.nu21 - self.nu23 * self.nu32 - self.nu13 * self.nu31 - 
                 2 * self.nu21 * self.nu32 * self.nu13)
        
        # Construir matriz
        C = np.zeros((6, 6))
        
        # Términos de la matriz de rigidez
        C[0, 0] = self.E1 * (1 - self.nu23 * self.nu32) / denom
        C[1, 1] = self.E2 * (1 - self.nu13 * self.nu31) / denom
        C[2, 2] = self.E3 * (1 - self.nu12 * self.nu21) / denom
        
        C[0, 1] = C[1, 0] = self.E1 * (self.nu21 + self.nu31 * self.nu23) / denom
        C[0, 2] = C[2, 0] = self.E1 * (self.nu31 + self.nu21 * self.nu32) / denom
        C[1, 2] = C[2, 1] = self.E2 * (self.nu32 + self.nu12 * self.nu31) / denom
        
        # Términos de cortante
        C[3, 3] = self.G12
        C[4, 4] = self.G13
        C[5, 5] = self.G23
        
        return C
    
    def get_compliance_matrix(self):
        """
        Obtiene la matriz de flexibilidad (inversa de la matriz de rigidez).
        
        Retorna:
        --------
        numpy.ndarray
            Matriz de flexibilidad
        """
        stiffness = self.get_stiffness_matrix()
        return np.linalg.inv(stiffness)
    
    def get_thermal_strain_vector(self, delta_t):
        """
        Calcula el vector de deformaciones térmicas.
        
        Parámetros:
        -----------
        delta_t : float
            Cambio de temperatura
            
        Retorna:
        --------
        numpy.ndarray
            Vector de deformaciones térmicas [e11, e22, e33, e12, e13, e23]
        """
        if self.material_type == MaterialType.ISOTROPIC:
            strain = np.zeros(6)
            strain[0:3] = self.alpha * delta_t  # Solo deformaciones normales
            return strain
        elif self.material_type == MaterialType.ORTHOTROPIC:
            strain = np.zeros(6)
            strain[0] = self.alpha1 * delta_t
            strain[1] = self.alpha2 * delta_t
            strain[2] = self.alpha3 * delta_t
            return strain
        elif self.material_type == MaterialType.ANISOTROPIC:
            return self.alpha_vector * delta_t
        else:
            raise ValueError(f"Tipo de material no soportado: {self.material_type}")
    
    def get_elastic_moduli(self):
        """
        Obtiene los módulos elásticos del material.
        
        Retorna:
        --------
        dict
            Diccionario con los módulos elásticos relevantes
        """
        if self.material_type == MaterialType.ISOTROPIC:
            return {
                'E': self.E,
                'G': self.G,
                'nu': self.nu
            }
        elif self.material_type == MaterialType.ORTHOTROPIC:
            return {
                'E1': self.E1, 'E2': self.E2, 'E3': self.E3,
                'G12': self.G12, 'G13': self.G13, 'G23': self.G23,
                'nu12': self.nu12, 'nu13': self.nu13, 'nu23': self.nu23,
                'nu21': self.nu21, 'nu31': self.nu31, 'nu32': self.nu32
            }
        elif self.material_type == MaterialType.ANISOTROPIC:
            # Para materiales anisótropos, devolver componentes de la matriz de rigidez
            return {'C': self.C}
    
    def get_poisson_ratio(self, direction1=None, direction2=None):
        """
        Obtiene el coeficiente de Poisson en las direcciones especificadas.
        
        Parámetros:
        -----------
        direction1, direction2 : int, opcional
            Direcciones para las cuales calcular el coeficiente (1, 2, 3)
            
        Retorna:
        --------
        float
            Coeficiente de Poisson
        """
        if self.material_type == MaterialType.ISOTROPIC:
            return self.nu
        elif self.material_type == MaterialType.ORTHOTROPIC:
            if direction1 == 1 and direction2 == 2:
                return self.nu12
            elif direction1 == 1 and direction2 == 3:
                return self.nu13
            elif direction1 == 2 and direction2 == 3:
                return self.nu23
            elif direction1 == 2 and direction2 == 1:
                return self.nu21
            elif direction1 == 3 and direction2 == 1:
                return self.nu31
            elif direction1 == 3 and direction2 == 2:
                return self.nu32
            else:
                raise ValueError("Direcciones no válidas para coeficiente de Poisson")
        else:
            raise ValueError(f"Operación no soportada para material tipo {self.material_type}")
    
    def get_strain_from_stress(self, stress_vector):
        """
        Calcula el vector de deformaciones a partir del vector de tensiones.
        
        Parámetros:
        -----------
        stress_vector : numpy.ndarray
            Vector de tensiones [s11, s22, s33, s12, s13, s23]
            
        Retorna:
        --------
        numpy.ndarray
            Vector de deformaciones [e11, e22, e33, e12, e13, e23]
        """
        compliance = self.get_compliance_matrix()
        return np.dot(compliance, stress_vector)
    
    def get_stress_from_strain(self, strain_vector):
        """
        Calcula el vector de tensiones a partir del vector de deformaciones.
        
        Parámetros:
        -----------
        strain_vector : numpy.ndarray
            Vector de deformaciones [e11, e22, e33, e12, e13, e23]
            
        Retorna:
        --------
        numpy.ndarray
            Vector de tensiones [s11, s22, s33, s12, s13, s23]
        """
        stiffness = self.get_stiffness_matrix()
        return np.dot(stiffness, strain_vector)
    
    def get_strain_energy_density(self, strain_vector):
        """
        Calcula la densidad de energía de deformación.
        
        Parámetros:
        -----------
        strain_vector : numpy.ndarray
            Vector de deformaciones [e11, e22, e33, e12, e13, e23]
            
        Retorna:
        --------
        float
            Densidad de energía de deformación
        """
        stress_vector = self.get_stress_from_strain(strain_vector)
        return 0.5 * np.dot(stress_vector, strain_vector)
    
    def clone(self, new_material_id=None, new_name=None):
        """
        Crea una copia del material.
        
        Parámetros:
        -----------
        new_material_id : str o int, opcional
            Nuevo ID para el material clonado
        new_name : str, opcional
            Nuevo nombre para el material clonado
            
        Retorna:
        --------
        Material
            Copia del material actual
        """
        material_id = new_material_id if new_material_id is not None else f"{self.material_id}_copy"
        name = new_name if new_name is not None else f"{self.name} (copy)"
        
        # Crear diccionario de propiedades según el tipo de material
        props = {'rho': self.rho}
        
        if self.material_type == MaterialType.ISOTROPIC:
            props.update({
                'E': self.E,
                'nu': self.nu,
                'G': self.G,
                'alpha': self.alpha
            })
        elif self.material_type == MaterialType.ORTHOTROPIC:
            props.update({
                'E1': self.E1, 'E2': self.E2, 'E3': self.E3,
                'nu12': self.nu12, 'nu13': self.nu13, 'nu23': self.nu23,
                'G12': self.G12, 'G13': self.G13, 'G23': self.G23,
                'alpha1': self.alpha1, 'alpha2': self.alpha2, 'alpha3': self.alpha3
            })
        elif self.material_type == MaterialType.ANISOTROPIC:
            props.update({
                'C': self.C.copy(),
                'alpha_vector': self.alpha_vector.copy() if hasattr(self, 'alpha_vector') else None
            })
        
        return Material(material_id, name, self.material_type, **props)
    
    @classmethod
    def create_steel(cls, material_id="steel", name="Structural Steel", grade="A36"):
        """
        Crea un material de acero estructural predefinido.
        
        Parámetros:
        -----------
        material_id : str o int, opcional
            ID del material
        name : str, opcional
            Nombre del material
        grade : str, opcional
            Grado del acero (A36, A572, etc.)
            
        Retorna:
        --------
        Material
            Material de acero con propiedades predeterminadas
        """
        # Propiedades según el grado
        props = {
            "A36": {
                "E": 200e9,      # 200 GPa
                "nu": 0.3,
                "rho": 7850,     # 7850 kg/m³
                "alpha": 12e-6,  # 12e-6 /°C
                "yield_stress": 250e6  # 250 MPa
            },
            "A572": {
                "E": 200e9,      # 200 GPa
                "nu": 0.3,
                "rho": 7850,     # 7850 kg/m³
                "alpha": 12e-6,  # 12e-6 /°C
                "yield_stress": 345e6  # 345 MPa (grado 50)
            },
            "A992": {
                "E": 200e9,      # 200 GPa
                "nu": 0.3,
                "rho": 7850,     # 7850 kg/m³
                "alpha": 12e-6,  # 12e-6 /°C
                "yield_stress": 345e6,  # 345 MPa
                "ultimate_stress": 450e6  # 450 MPa
            }
        }
        
        # Usar A36 como predeterminado si no se encuentra el grado
        grade_props = props.get(grade, props["A36"])
        
        # Crear el material
        material = cls(material_id, name, MaterialType.ISOTROPIC, **grade_props)
        material.grade = grade
        
        return material
    
    @classmethod
    def create_concrete(cls, material_id="concrete", name="Concrete", fc=25e6):
        """
        Crea un material de concreto predefinido.
        
        Parámetros:
        -----------
        material_id : str o int, opcional
            ID del material
        name : str, opcional
            Nombre del material
        fc : float, opcional
            Resistencia a la compresión (Pa)
            
        Retorna:
        --------
        Material
            Material de concreto con propiedades predeterminadas
        """
        # Módulo de elasticidad según ACI 318
        E = 4700 * np.sqrt(fc)  # E en MPa si fc en MPa
        
        # Propiedades
        props = {
            "E": E,
            "nu": 0.2,
            "rho": 2400,     # 2400 kg/m³
            "alpha": 10e-6,  # 10e-6 /°C
            "compression_strength": fc,
            "tension_strength": 0.1 * fc  # Estimación aproximada
        }
        
        # Crear el material
        material = cls(material_id, name, MaterialType.ISOTROPIC, **props)
        material.fc = fc
        
        return material
    
    @classmethod
    def create_aluminum(cls, material_id="aluminum", name="Aluminum Alloy", alloy="6061-T6"):
        """
        Crea un material de aluminio predefinido.
        
        Parámetros:
        -----------
        material_id : str o int, opcional
            ID del material
        name : str, opcional
            Nombre del material
        alloy : str, opcional
            Aleación de aluminio (6061-T6, 7075-T6, etc.)
            
        Retorna:
        --------
        Material
            Material de aluminio con propiedades predeterminadas
        """
        # Propiedades según la aleación
        props = {
            "6061-T6": {
                "E": 69e9,       # 69 GPa
                "nu": 0.33,
                "rho": 2700,     # 2700 kg/m³
                "alpha": 23.4e-6,# 23.4e-6 /°C
                "yield_stress": 240e6,  # 240 MPa
                "ultimate_stress": 290e6  # 290 MPa
            },
            "7075-T6": {
                "E": 71.7e9,     # 71.7 GPa
                "nu": 0.33,
                "rho": 2810,     # 2810 kg/m³
                "alpha": 23.1e-6,# 23.1e-6 /°C
                "yield_stress": 470e6,  # 470 MPa
                "ultimate_stress": 540e6  # 540 MPa
            }
        }
        
        # Usar 6061-T6 como predeterminado si no se encuentra la aleación
        alloy_props = props.get(alloy, props["6061-T6"])
        
        # Crear el material
        material = cls(material_id, name, MaterialType.ISOTROPIC, **alloy_props)
        material.alloy = alloy
        
        return material
    
    @classmethod
    def create_timber(cls, material_id="timber", name="Structural Timber", species="Pine"):
        """
        Crea un material de madera predefinido.
        
        Parámetros:
        -----------
        material_id : str o int, opcional
            ID del material
        name : str, opcional
            Nombre del material
        species : str, opcional
            Especie de madera
            
        Retorna:
        --------
        Material
            Material de madera con propiedades ortótropas predeterminadas
        """
        # Propiedades aproximadas según la especie (valores simplificados)
        props = {
            "Pine": {
                "E1": 12e9,      # 12 GPa paralelo a la fibra
                "E2": 1.0e9,     # 1.0 GPa perpendicular a la fibra (radial)
                "E3": 0.8e9,     # 0.8 GPa perpendicular a la fibra (tangencial)
                "G12": 0.9e9,    # 0.9 GPa
                "G13": 0.8e9,    # 0.8 GPa
                "G23": 0.3e9,    # 0.3 GPa
                "nu12": 0.3,
                "nu13": 0.3,
                "nu23": 0.4,
                "rho": 550,      # 550 kg/m³
                "alpha1": 5e-6,  # 5e-6 /°C paralelo a la fibra
                "alpha2": 30e-6, # 30e-6 /°C perpendicular a la fibra (radial)
                "alpha3": 40e-6  # 40e-6 /°C perpendicular a la fibra (tangencial)
            },
            "Oak": {
                "E1": 13e9,      # 13 GPa paralelo a la fibra
                "E2": 1.3e9,     # 1.3 GPa perpendicular a la fibra (radial)
                "E3": 1.0e9,     # 1.0 GPa perpendicular a la fibra (tangencial)
                "G12": 1.0e9,    # 1.0 GPa
                "G13": 0.9e9,    # 0.9 GPa
                "G23": 0.35e9,   # 0.35 GPa
                "nu12": 0.35,
                "nu13": 0.35,
                "nu23": 0.45,
                "rho": 700,      # 700 kg/m³
                "alpha1": 5e-6,  # 5e-6 /°C paralelo a la fibra
                "alpha2": 35e-6, # 35e-6 /°C perpendicular a la fibra (radial)
                "alpha3": 45e-6  # 45e-6 /°C perpendicular a la fibra (tangencial)
            }
        }
        
        # Usar Pine como predeterminado si no se encuentra la especie
        species_props = props.get(species, props["Pine"])
        
        # Crear el material ortótropo
        material = cls(material_id, name, MaterialType.ORTHOTROPIC, **species_props)
        material.species = species
        
        return material
    
    def __str__(self):
        """Representación en string del material."""
        if self.material_type == MaterialType.ISOTROPIC:
            return f"Material: {self.name} (ID: {self.material_id})\n" \
                   f"Tipo: Isótropo\n" \
                   f"E = {self.E:.3e} Pa, nu = {self.nu:.3f}, rho = {self.rho:.1f} kg/m³"
        elif self.material_type == MaterialType.ORTHOTROPIC:
            return f"Material: {self.name} (ID: {self.material_id})\n" \
                   f"Tipo: Ortótropo\n" \
                   f"E1 = {self.E1:.3e} Pa, E2 = {self.E2:.3e} Pa, E3 = {self.E3:.3e} Pa\n" \
                   f"nu12 = {self.nu12:.3f}, nu13 = {self.nu13:.3f}, nu23 = {self.nu23:.3f}\n" \
                   f"rho = {self.rho:.1f} kg/m³"
        elif self.material_type == MaterialType.ANISOTROPIC:
            return f"Material: {self.name} (ID: {self.material_id})\n" \
                   f"Tipo: Anisótropo\n" \
                   f"rho = {self.rho:.1f} kg/m³"


import numpy as np
from enum import Enum

class SectionType(Enum):
    """Enumeración para los diferentes tipos de sección."""
    CUSTOM = "custom"          # Sección personalizada
    RECTANGLE = "rectangle"    # Sección rectangular
    CIRCLE = "circle"          # Sección circular
    I_SECTION = "i_section"    # Sección I/H
    T_SECTION = "t_section"    # Sección T
    L_SECTION = "l_section"    # Sección L (ángulo)
    C_SECTION = "c_section"    # Sección C (canal)
    TUBE = "tube"              # Sección tubular
    BOX = "box"                # Sección cajón
    
class Section:
    """
    Clase para representar las propiedades geométricas de una sección transversal.
    """
    def __init__(self, section_id, name=None, section_type=SectionType.CUSTOM, **kwargs):
        """
        Constructor para una sección transversal.
        
        Parámetros:
        -----------
        section_id : str o int
            Identificador único de la sección
        name : str, opcional
            Nombre descriptivo de la sección
        section_type : SectionType, opcional
            Tipo de sección
        **kwargs : dict
            Propiedades geométricas de la sección dependiendo del tipo
        """
        self.section_id = section_id
        self.name = name if name is not None else f"Section-{section_id}"
        self.section_type = section_type
        
        # Inicializar propiedades básicas
        self.area = kwargs.get('area', 0.0)  # Área
        self.Iy = kwargs.get('Iy', 0.0)      # Momento de inercia respecto al eje y
        self.Iz = kwargs.get('Iz', 0.0)      # Momento de inercia respecto al eje z
        self.J = kwargs.get('J', 0.0)        # Constante torsional
        self.Iyz = kwargs.get('Iyz', 0.0)    # Producto de inercia
        self.Wy = kwargs.get('Wy', 0.0)      # Módulo resistente elástico respecto al eje y
        self.Wz = kwargs.get('Wz', 0.0)      # Módulo resistente elástico respecto al eje z
        self.Sy = kwargs.get('Sy', 0.0)      # Módulo plástico respecto al eje y
        self.Sz = kwargs.get('Sz', 0.0)      # Módulo plástico respecto al eje z
        self.ry = kwargs.get('ry', 0.0)      # Radio de giro respecto al eje y
        self.rz = kwargs.get('rz', 0.0)      # Radio de giro respecto al eje z
        self.Ay = kwargs.get('Ay', 0.0)      # Área de cortante en dirección y
        self.Az = kwargs.get('Az', 0.0)      # Área de cortante en dirección z
        
        # Propiedades geométricas para tipos de sección específicos
        self.dimensions = kwargs.get('dimensions', {})  # Dimensiones para generar la sección
        
        # Propiedades para cálculos avanzados
        self.warping_constant = kwargs.get('warping_constant', 0.0)  # Constante de alabeo
        self.shear_center_y = kwargs.get('shear_center_y', 0.0)      # Coordenada y del centro de cortante
        self.shear_center_z = kwargs.get('shear_center_z', 0.0)      # Coordenada z del centro de cortante
        
        # Propiedades para análisis de pandeo
        self.buckling_params = kwargs.get('buckling_params', {})
        
        # Si se proporciona dimensiones pero no propiedades calculadas, calcularlas
        if self.dimensions and self.section_type != SectionType.CUSTOM and self.area == 0.0:
            self.calculate_section_properties()
    
    def calculate_section_properties(self):
        """
        Calcula las propiedades de la sección basándose en sus dimensiones y tipo.
        """
        if self.section_type == SectionType.RECTANGLE:
            self._calculate_rectangle_properties()
        elif self.section_type == SectionType.CIRCLE:
            self._calculate_circle_properties()
        elif self.section_type == SectionType.I_SECTION:
            self._calculate_i_section_properties()
        elif self.section_type == SectionType.T_SECTION:
            self._calculate_t_section_properties()
        elif self.section_type == SectionType.L_SECTION:
            self._calculate_l_section_properties()
        elif self.section_type == SectionType.C_SECTION:
            self._calculate_c_section_properties()
        elif self.section_type == SectionType.TUBE:
            self._calculate_tube_properties()
        elif self.section_type == SectionType.BOX:
            self._calculate_box_properties()
        # Para CUSTOM, se asume que las propiedades ya fueron proporcionadas
    
    def _calculate_rectangle_properties(self):
        """Calcula propiedades para una sección rectangular."""
        h = self.dimensions.get('height', 0.0)
        b = self.dimensions.get('width', 0.0)
        
        self.area = b * h
        self.Iy = (b * h**3) / 12.0
        self.Iz = (h * b**3) / 12.0
        self.J = (b * h * (b**2 + h**2)) / 12.0
        self.Wy = (b * h**2) / 6.0
        self.Wz = (h * b**2) / 6.0
        # Para secciones rectangulares macizas, factor 5/6 para áreas de cortante
        self.Ay = (5.0/6.0) * self.area
        self.Az = (5.0/6.0) * self.area
        self.ry = np.sqrt(self.Iy / self.area) if self.area > 0 else 0.0
        self.rz = np.sqrt(self.Iz / self.area) if self.area > 0 else 0.0
        # Módulos plásticos (aproximación para sección rectangular)
        self.Sy = (b * h**2) / 4.0
        self.Sz = (h * b**2) / 4.0
    
    def _calculate_circle_properties(self):
        """Calcula propiedades para una sección circular."""
        r = self.dimensions.get('radius', 0.0)
        
        self.area = np.pi * r**2
        self.Iy = (np.pi * r**4) / 4.0
        self.Iz = self.Iy
        self.J = (np.pi * r**4) / 2.0
        self.Wy = (np.pi * r**3) / 4.0
        self.Wz = self.Wy
        # Para secciones circulares macizas, factor 0.9 para áreas de cortante
        self.Ay = 0.9 * self.area
        self.Az = 0.9 * self.area
        self.ry = r / 2.0
        self.rz = self.ry
        # Módulos plásticos (para sección circular)
        self.Sy = (4.0 * r**3) / 3.0
        self.Sz = self.Sy
    
    def _calculate_i_section_properties(self):
        """Calcula propiedades para una sección I/H."""
        h = self.dimensions.get('height', 0.0)
        b_f = self.dimensions.get('flange_width', 0.0)
        t_f = self.dimensions.get('flange_thickness', 0.0)
        t_w = self.dimensions.get('web_thickness', 0.0)
        
        # Área total
        self.area = 2 * b_f * t_f + (h - 2 * t_f) * t_w
        
        # Momento de inercia respecto al eje y (eje fuerte)
        self.Iy = (b_f * h**3) / 12.0 - ((b_f - t_w) * (h - 2 * t_f)**3) / 12.0
        
        # Momento de inercia respecto al eje z (eje débil)
        self.Iz = (2 * t_f * b_f**3) / 12.0 + ((h - 2 * t_f) * t_w**3) / 12.0
        
        # Constante torsional (aproximación)
        self.J = (1.0/3.0) * (2 * b_f * t_f**3 + (h - 2 * t_f) * t_w**3)
        
        # Módulos resistentes elásticos
        self.Wy = (2 * self.Iy) / h
        self.Wz = (2 * self.Iz) / b_f
        
        # Áreas de cortante (aproximaciones para perfiles I)
        self.Ay = (h - 2 * t_f) * t_w  # Área del alma
        self.Az = 2 * b_f * t_f        # Área de las alas
        
        # Radios de giro
        self.ry = np.sqrt(self.Iy / self.area) if self.area > 0 else 0.0
        self.rz = np.sqrt(self.Iz / self.area) if self.area > 0 else 0.0
        
        # Módulos plásticos (cálculo aproximado)
        h_w = h - 2 * t_f  # Altura del alma
        self.Sy = t_w * h_w**2 / 4.0 + b_f * t_f * (h - t_f)
        self.Sz = (t_w**2 * h_w / 4.0) + (b_f**2 * t_f / 2.0)
        
        # Constante de alabeo (aproximación)
        self.warping_constant = (t_f * b_f**3 * h**2) / 24.0
    
    # Los métodos para los otros tipos de secciones seguirían estructura similar...
    
    def _calculate_t_section_properties(self):
        """Calcula propiedades para una sección T."""
        h = self.dimensions.get('height', 0.0)
        b_f = self.dimensions.get('flange_width', 0.0)
        t_f = self.dimensions.get('flange_thickness', 0.0)
        t_w = self.dimensions.get('web_thickness', 0.0)
        
        # Implementación similar a la sección I pero adaptada para T
        # ...
        pass
    
    def _calculate_l_section_properties(self):
        """Calcula propiedades para una sección L (ángulo)."""
        # ...
        pass
    
    def _calculate_c_section_properties(self):
        """Calcula propiedades para una sección C (canal)."""
        # ...
        pass
    
    def _calculate_tube_properties(self):
        """Calcula propiedades para una sección tubular."""
        # ...
        pass
    
    def _calculate_box_properties(self):
        """Calcula propiedades para una sección cajón."""
        # ...
        pass
    
    def calculate_stresses(self, axial_force=0.0, my=0.0, mz=0.0, vz=0.0, vy=0.0, torque=0.0):
        """
        Calcula los esfuerzos en la sección debido a las fuerzas y momentos aplicados.
        
        Parámetros:
        -----------
        axial_force : float
            Fuerza axial (N o kN)
        my : float
            Momento flector alrededor del eje y (N·m o kN·m)
        mz : float
            Momento flector alrededor del eje z (N·m o kN·m)
        vz : float
            Fuerza cortante en dirección z (N o kN)
        vy : float
            Fuerza cortante en dirección y (N o kN)
        torque : float
            Momento torsor (N·m o kN·m)
            
        Retorna:
        --------
        dict
            Diccionario con los esfuerzos calculados
        """
        stresses = {}
        
        # Esfuerzo axial
        if self.area > 0:
            stresses['axial'] = axial_force / self.area
        else:
            stresses['axial'] = 0.0
        
        # Esfuerzos de flexión
        if self.Wy > 0:
            stresses['flexion_y'] = my / self.Wy
        else:
            stresses['flexion_y'] = 0.0
            
        if self.Wz > 0:
            stresses['flexion_z'] = mz / self.Wz
        else:
            stresses['flexion_z'] = 0.0
        
        # Esfuerzos cortantes
        if self.Ay > 0:
            stresses['shear_y'] = vy / self.Ay
        else:
            stresses['shear_y'] = 0.0
            
        if self.Az > 0:
            stresses['shear_z'] = vz / self.Az
        else:
            stresses['shear_z'] = 0.0
        
        # Esfuerzo de torsión
        if self.J > 0:
            # Simplificación para torsión en secciones circulares
            if self.section_type == SectionType.CIRCLE:
                r = self.dimensions.get('radius', 0.0)
                stresses['torsion'] = (torque * r) / self.J
            else:
                # Para otras secciones se necesitaría un análisis más detallado
                stresses['torsion'] = 0.0  # Aquí se podría implementar un cálculo más preciso
        else:
            stresses['torsion'] = 0.0
        
        # Esfuerzo combinado (von Mises)
        sigma_x = stresses['axial'] + stresses['flexion_y'] + stresses['flexion_z']
        tau_yz = np.sqrt(stresses['shear_y']**2 + stresses['shear_z']**2 + stresses['torsion']**2)
        stresses['von_mises'] = np.sqrt(sigma_x**2 + 3 * tau_yz**2)
        
        return stresses
    
    def check_section_capacity(self, material, axial_force=0.0, my=0.0, mz=0.0, vz=0.0, vy=0.0, torque=0.0):
        """
        Verifica la capacidad de la sección considerando el material y las fuerzas aplicadas.
        
        Parámetros:
        -----------
        material : Material
            Objeto material con las propiedades del material
        axial_force, my, mz, vz, vy, torque : float
            Fuerzas y momentos aplicados
            
        Retorna:
        --------
        dict
            Diccionario con los ratios de utilización
        """
        stresses = self.calculate_stresses(axial_force, my, mz, vz, vy, torque)
        utilization = {}
        
        # Verificación de resistencia
        if hasattr(material, 'yield_stress'):
            # Verificación axial
            utilization['axial'] = abs(stresses['axial']) / material.yield_stress
            
            # Verificación flexión
            utilization['flexion_y'] = abs(stresses['flexion_y']) / material.yield_stress
            utilization['flexion_z'] = abs(stresses['flexion_z']) / material.yield_stress
            
            # Verificación cortante
            if hasattr(material, 'shear_yield_stress'):
                shear_yield = material.shear_yield_stress
            else:
                # Aproximación del esfuerzo de fluencia a cortante
                shear_yield = material.yield_stress / np.sqrt(3)
                
            utilization['shear_y'] = abs(stresses['shear_y']) / shear_yield
            utilization['shear_z'] = abs(stresses['shear_z']) / shear_yield
            utilization['torsion'] = abs(stresses['torsion']) / shear_yield
            
            # Verificación combinada (von Mises)
            utilization['von_mises'] = stresses['von_mises'] / material.yield_stress
            
            # Utilización máxima
            utilization['max'] = max(utilization.values())
        
        return utilization
    
    def rotate(self, angle_degrees):
        """
        Rota la sección alrededor del origen un ángulo dado.
        Esto afecta a los momentos de inercia y productos de inercia.
        
        Parámetros:
        -----------
        angle_degrees : float
            Ángulo de rotación en grados
        """
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        cos_a2 = cos_a**2
        sin_a2 = sin_a**2
        sin_2a = np.sin(2 * angle_rad)
        
        # Rotación de momentos de inercia
        Iy_new = self.Iy * cos_a2 + self.Iz * sin_a2 - self.Iyz * sin_2a
        Iz_new = self.Iy * sin_a2 + self.Iz * cos_a2 + self.Iyz * sin_2a
        Iyz_new = (self.Iy - self.Iz) * sin_a * cos_a + self.Iyz * (cos_a2 - sin_a2)
        
        self.Iy = Iy_new
        self.Iz = Iz_new
        self.Iyz = Iyz_new
        
        # Recalcular otras propiedades que dependen de los momentos de inercia
        if self.area > 0:
            self.ry = np.sqrt(self.Iy / self.area)
            self.rz = np.sqrt(self.Iz / self.area)
        
        # Nota: Para una rotación completa, se debería recalcular también
        # los módulos resistentes y otras propiedades
    
    def get_principal_axes(self):
        """
        Calcula los ejes principales de inercia y sus momentos de inercia.
        
        Retorna:
        --------
        tuple
            (ángulo en radianes, I1, I2) donde I1, I2 son los momentos principales de inercia
        """
        if abs(self.Iyz) < 1e-10:  # Si el producto de inercia es despreciable
            if self.Iy >= self.Iz:
                return 0.0, self.Iy, self.Iz
            else:
                return np.pi/2, self.Iz, self.Iy
        
        # Ángulo de los ejes principales
        theta = 0.5 * np.arctan2(2 * self.Iyz, self.Iy - self.Iz)
        
        # Momentos principales de inercia
        I_avg = (self.Iy + self.Iz) / 2
        I_diff = np.sqrt(((self.Iy - self.Iz) / 2)**2 + self.Iyz**2)
        I1 = I_avg + I_diff  # Mayor momento de inercia
        I2 = I_avg - I_diff  # Menor momento de inercia
        
        return theta, I1, I2
    
    def copy(self):
        """Crea una copia de la sección."""
        import copy
        return copy.deepcopy(self)
    
    def __str__(self):
        """Representación de cadena de la sección."""
        return f"Section(id={self.section_id}, type={self.section_type.value}, A={self.area:.6f})"
    
    def __repr__(self):
        """Representación detallada de la sección."""
        return f"Section(id={self.section_id}, name={self.name}, type={self.section_type.value}, " + \
               f"A={self.area:.6f}, Iy={self.Iy:.6f}, Iz={self.Iz:.6f}, J={self.J:.6f})"