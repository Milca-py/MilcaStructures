

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
            status = "Restringido" if self.constraints.get(dof, False) else "Libre"
            index = self.indices.get(dof, -1)
            value = self.prescribed_values.get(dof, 0.0)
            
            if self.constraints.get(dof, False):
                result += f"  {dof}: {status}, Valor={value:.6f}\n"
            else:
                result += f"  {dof}: {status}, Índice Global={index}\n"
        
        return result

if __name__ == "__main__":
    # Ejemplo de uso
    dof = DegreesOfFreedom(node_id=1, dimension=2)
    print(dof)
    
    dof.apply_constraint('UY', value=0)
    dof.set_global_index('UX', 0)
    print(dof)