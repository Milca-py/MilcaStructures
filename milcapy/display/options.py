from typing import TYPE_CHECKING, Tuple, Optional
import numpy as np

if TYPE_CHECKING:
    from milcapy.elements.system import SystemMilcaModel

class GraphicOption:
    """
    Clase que gestiona las opciones gráficas para representar un sistema estructural.
    
    Esta clase calcula varias propiedades necesarias para la visualización adecuada
    de los elementos del sistema, como escalas para fuerzas, cargas, y tamaños de
    elementos gráficos.
    """
    
    def __init__(self, system: "SystemMilcaModel") -> None:
        """
        Inicializa una instancia de GraphicOption.
        
        Args:
            system: Instancia del modelo del sistema estructural a representar.
        """
        self.system = system
        # Valores cacheados para mejorar rendimiento
        self._cached_length_mean: Optional[float] = None
        self._cached_values = {}
    
    def _get_cached_mean(self, key: str, value_getter, filter_condition):
        """
        Método auxiliar para calcular y cachear valores medios.
        
        Args:
            key: Clave para identificar el valor en la caché.
            value_getter: Función para obtener el valor de cada elemento.
            filter_condition: Función para filtrar elementos válidos.
            
        Returns:
            float: Valor medio calculado o recuperado de la caché.
        """
        if key not in self._cached_values:
            values = [
                value_getter(item) 
                for item in filter_condition()
                if value_getter(item) != 0
            ]
            
            if not values:
                self._cached_values[key] = 1.0  # Valor predeterminado para evitar divisiones por cero
            else:
                self._cached_values[key] = np.mean(np.abs(values))
                
        return self._cached_values[key]

    @property
    def _length_mean(self) -> float:
        """
        Calcula la longitud media de todos los elementos del sistema.
        
        Returns:
            float: Longitud media.
        """
        if self._cached_length_mean is None:
            elements = list(self.system.element_map.values())
            if not elements:
                self._cached_length_mean = 1.0  # Valor predeterminado
            else:
                self._cached_length_mean = np.mean([element.length for element in elements])
        
        return self._cached_length_mean

    @property
    def _qi_mean(self) -> float:
        """
        Calcula el valor medio absoluto de las cargas distribuidas iniciales.
        
        Returns:
            float: Valor medio de q_i.
        """
        return self._get_cached_mean(
            "qi_mean",
            lambda e: e.distributed_load.q_i,
            lambda: self.system.element_map.values()
        )
    
    @property
    def _qj_mean(self) -> float:
        """
        Calcula el valor medio absoluto de las cargas distribuidas finales.
        
        Returns:
            float: Valor medio de q_j.
        """
        return self._get_cached_mean(
            "qj_mean",
            lambda e: e.distributed_load.q_j,
            lambda: self.system.element_map.values()
        )
    
    @property
    def _pi_mean(self) -> float:
        """
        Calcula el valor medio absoluto de las cargas axiales iniciales.
        
        Returns:
            float: Valor medio de p_i.
        """
        return self._get_cached_mean(
            "pi_mean",
            lambda e: e.distributed_load.p_i,
            lambda: self.system.element_map.values()
        )
    
    @property
    def _pj_mean(self) -> float:
        """
        Calcula el valor medio absoluto de las cargas axiales finales.
        
        Returns:
            float: Valor medio de p_j.
        """
        return self._get_cached_mean(
            "pj_mean",
            lambda e: e.distributed_load.p_j,
            lambda: self.system.element_map.values()
        )
    
    @property
    def _fx_mean(self) -> float:
        """
        Calcula el valor medio absoluto de las fuerzas en x aplicadas a los nodos.
        
        Returns:
            float: Valor medio de fx.
        """
        return self._get_cached_mean(
            "fx_mean",
            lambda n: n.forces.fx,
            lambda: self.system.node_map.values()
        )
    
    @property
    def _fy_mean(self) -> float:
        """
        Calcula el valor medio absoluto de las fuerzas en y aplicadas a los nodos.
        
        Returns:
            float: Valor medio de fy.
        """
        return self._get_cached_mean(
            "fy_mean",
            lambda n: n.forces.fy,
            lambda: self.system.node_map.values()
        )
    
    @property
    def _mz_mean(self) -> float:
        """
        Calcula el valor medio absoluto de los momentos en z aplicados a los nodos.
        
        Returns:
            float: Valor medio de mz.
        """
        return self._get_cached_mean(
            "mz_mean",
            lambda n: n.forces.mz,
            lambda: self.system.node_map.values()
        )
    
    @property
    def ratio_scale_force(self) -> float:
        """
        Calcula la escala para representar fuerzas nodales.
        
        Returns:
            float: Factor de escala para fuerzas.
        """
        force_mean = self._fx_mean + self._fy_mean
        if force_mean < 1e-10:
            return 0.15 * self._length_mean
        return 0.15 * self._length_mean * (2 / force_mean)

    @property
    def ratio_scale_load(self) -> float:
        """
        Calcula la escala para representar cargas distribuidas.
        
        Returns:
            float: Factor de escala para cargas distribuidas.
        """
        load_mean = self._qi_mean + self._qj_mean
        if load_mean < 1e-10:
            return 0.15 * self._length_mean
        return 0.15 * self._length_mean * (2 / load_mean)
    
    @property
    def ratio_scale_axial(self) -> float:
        """
        Calcula la escala para representar cargas axiales.
        
        Returns:
            float: Factor de escala para cargas axiales.
        """
        axial_mean = self._pi_mean + self._pj_mean
        if axial_mean < 1e-10:
            return 0.15 * self._length_mean
        return 0.15 * self._length_mean * (2 / axial_mean)
    
    @property
    def nrof_arrows(self) -> int:
        """
        Determina el número de flechas a utilizar para representar cargas.
        
        Returns:
            int: Número de flechas.
        """
        return 10

    @property
    def support_size(self) -> float:
        """
        Calcula el tamaño adecuado para los apoyos en la visualización.
        
        Returns:
            float: Tamaño para los apoyos.
        """
        return 0.1 * self._length_mean
    
    @property
    def figsize(self) -> Tuple[int, int]:
        """
        Define el tamaño de la figura para la visualización.
        
        Returns:
            Tuple[int, int]: Dimensiones (ancho, alto) de la figura.
        """
        return (10, 10)