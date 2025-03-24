from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel


class PlotterOptions:
    """
    Clase que define las opciones para la visualización de resultados
    de análisis estructural.
    """

    def __init__(self):
        """
        Inicializa las opciones de visualización con valores predeterminados.
        """
        # self.values_calculator: GraphicOptionCalculator = graphic_option_calculator
        # Opciones
        # GENERALES
        self.figure_size = (10, 8)      ##
        self.dpi = 100                  ##
        self.UI_background_color = 'white' ##
        self.grid = False                ##
        # 'default', 'dark_background', 'ggplot', etc.
        self.plot_style = 'default'     ##

        # Opciones para visualización de estructura
        # NODOS:
        self.UI_show_nodes = True           #####
        self.node_size = 4              #####
        self.node_color = 'blue'            #####
        # ELEMENTOS:
        self.UI_show_elements = True        #####
        self.element_line_width = 1.0   #####
        self.element_color = 'blue'    #####
        # APOYOS:
        self.support_size = 0.1        #####
        self.support_color = 'green'    #####
        # OPCIONES DE SELECCIÓN:
        self.highlight_selected = True
        self.selected_color = 'red'
        # ETIQUETAS DE NODOS Y ELEMENTOS:
        self.UI_node_labels = False              ###########
        self.UI_element_labels = False           ###########
        self.label_font_size = 8                ##########
        self.node_label_color = '#ed0808'        ###########
        self.element_label_color = '#26a699'     ###########

        # CARGAS PUNTUALES
        self.UI_point_load = False               ###########
        self.point_load_color = '#282e3e'           ###########
        self.point_load_length_arrow = 0.2 ############
        self.point_moment_length_arrow = 0.5 ########
        # ETIQUETAS DE CARGAS PUNTUALES
        self.UI_point_load_label = True                    ##########
        self.point_load_label_color = '#ff0892'         ###########
        self.point_load_label_font_size = 8             ###########

        # CARGAS DISTRIBUIDAS
        self.ratio_scale_load = 0.1                     ####
        self.nrof_arrows = 10                           #######
        self.distributed_load_color = '#831f7a'         ########
        # ETIQUETAS DE CARGAS DISTRIBUIDAS
        self.distributed_load_label = True              #######
        self.distributed_load_label_color = '#511f74'   #######
        self.distributed_load_label_font_size = 8       #######

        # DEFORMADA
        self.UI_deformation_scale = 40 ########### # Factor de escala para deformaciones
        self.deformation_line_width = 1.0 ######### # Ancho de línea para deformaciones
        self.deformation_color = '#007acc' ########### # Color para deformaciones
        # CON ESTOS DATOS DE ACTUALIZA DE FORMA SIN DEFORMAR automatixcamente, y se REVIERTE CON EL BOTON DE DEFORMADA
        self.show_undeformed = False    ########## # Mostrar estructura sin deformar
        self.undeformed_style = 'dashed'  # Estilo para estructura sin deformar
        self.undeformed_color = '#aeacad'   ######### Color para estructura sin deformar

        # ANOTACIONES DE LOS DEZPLAZAMIENTOS EN NODOS
        self.disp_nodes = True    ########## # Mostrar desplazamientos en nodos
        self.disp_nodes_color = 'black'  ########## # Color para desplazamientos en nodos
        self.disp_nodes_font_size = 8    ########## # Tamaño de fuente para desplazamientos en nodos

        # FUERZAS INTERNAS
        # Dibujar momentos en lado de tensión (C| ---|Ɔ)
        self.UI_internal_forces_scale = 0.03  ###### # Factor de escala para diagramas de esfuerzos
        self.moment_on_tension_side = True
        self.axial_force_color = 'blue'    # Color para diagrama de axial
        self.shear_force_color = 'green'   # Color para diagrama de cortante
        self.bending_moment_color = 'red'  # Color para diagrama de momento
        self.fi_line_width = 1.0           # Ancho de línea de contorno para diagramas de esfuerzos
        self.UI_dscale_s_d = 0.03           # Factor de escala para slope y deflection
        self.slope = "red"                 # Color para diagrama de giros
        self.deflection = "blue"           # Color para diagrama de deflexiones

        # RELLENOS Y CONTORNOS
        self.UI_fill_diagram = True         # Rellenar diagramas
        self.UI_filling_type = 'solid'      # 'solid', 'barcolor'
        self.alpha_filling = 0.7         # Transparencia para relleno
        self.UI_colormap = 'jet'            # 'jet', 'viridis', 'coolwarm', etc.
        self.UI_show_colorbar = True        # Mostrar barra de colores
        self.border_contour = True          # Mostrar contornos
        self.edge_color = 'black'        # Color de bordes en contornos
        self.alpha = 0.7                 # Transparencia para contornos
        self.num_contours = 20           # Número de niveles en contornos

        # OPCIONES DE GUARDADO
        self.save_dpi = 300               # DPI para guardar imágenes
        self.tight_layout = True          # Ajuste automático de layout


        # OTROS:
        self.label_size = 8               # Tamaño de fuente para etiquetas
        self.relsult_label_size = 8       # Tamaño de fuente para etiquetas de resultados
    def reset(self):
        """
        Reinicia todas las opciones a sus valores predeterminados.
        """
        self.__init__()
        self.UI_label_font_size = self.label_size
        self.point_load_label_font_size = self.label_size
        self.distributed_load_label_font_size = self.label_size
        self.disp_nodes_font_size = self.relsult_label_size








class GraphicOptionCalculator:
    """
    Clase que calcula los parametros maximos y medios para la visualizacion de la estructura.
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
                # Valor predeterminado para evitar divisiones por cero
                self._cached_values[key] = 1.0
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
                self._cached_length_mean = np.mean(
                    [element.length for element in elements])

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


























































# class PlotterOptions:
#     """
#     Clase que define las opciones para la visualización de resultados
#     de análisis estructural.
#     """

#     def __init__(self):
#         """
#         Inicializa las opciones de visualización con valores predeterminados.
#         """
#         # Opciones
#         # GENERALES
#         self.UI_background_color = 'white' ##

#         # Opciones para visualización de estructura
#         # NODOS:
#         self.UI_show_nodes = True           #####
#         # ELEMENTOS:
#         self.UI_show_elements = True        #####
#         # ETIQUETAS DE NODOS Y ELEMENTOS:
#         self.UI_node_labels = False              ###########
#         self.UI_element_labels = False           ###########

#         # CARGAS PUNTUALES
#         self.UI_point_load = False               ###########

#         # ETIQUETAS DE CARGAS PUNTUALES
#         self.UI_point_load_label = True                    ##########

#         # DEFORMADA
#         self.UI_deformation_scale = 40 ########### # Factor de escala para deformaciones

#         # FUERZAS INTERNAS
#         self.UI_internal_forces_scale = 0.03  ###### # Factor de escala para diagramas de esfuerzos
#         self.UI_dscale_s_d = 0.03           # Factor de escala para slope y deflection

#         # RELLENOS Y CONTORNOS
#         self.UI_fill_diagram = True         # Rellenar diagramas
#         self.UI_filling_type = 'solid'      # 'solid', 'barcolor'
#         self.UI_colormap = 'jet'            # 'jet', 'viridis', 'coolwarm', etc.
#         self.UI_show_colorbar = True        # Mostrar barra de colores
