from typing import Dict, Tuple, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
from enum import Enum, auto

if TYPE_CHECKING:
    from milcapy.elements.system import SystemMilcaModel
    from milcapy.elements.load_pattern import LoadPattern


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


"""
VALORES EN PlotterValues

*** ESTRUCTURA ***
1. Nodos                    : {id_node: (x, y)}
2. Elementos                : {id_element: [(x1, y1), (x2, y2)]}
3. Restricciones            : {id_node: (restricciones)}
4. Cargas_distribuidas      : {id_element: {q_i, q_j, p_i, p_j, m_i, m_j}}
5. Cargas_puntuales         : {id_node: {fx, fy, mz}}

*** RESULTADOS DE ANALISIS MATRICIAL***
1. Desplazamientos nodales  : {id_node: (ux, vy, wz)}
2. Reacciones               : {id_node: (rx, ry, rz)}
3. Fuerzas internas         : {id_element: {axial, shear, moment}}

*** RESULTADOS DE POST-PROCESSING***
1. Fuerzas Axiales          : {id_element: (np.ndarray, np.ndarray)}
2. Fuerzas Cortantes        : {id_element: (np.ndarray, np.ndarray)}
3. Momentos Flectores       : {id_element: (np.ndarray, np.ndarray)}
4. Giros                    : {id_element: (np.ndarray, np.ndarray)}
5. Deflexiones              : {id_element: (np.ndarray, np.ndarray)}
6. Deformada                : {id_element: (np.ndarray, np.ndarray)}
7. Deformada Rígida         : {id_element: (np.ndarray, np.ndarray)}

"""


class PlotterOptions:
    """
    Clase que define las opciones para la visualización de resultados
    de análisis estructural.
    """

    def __init__(self, graphic_option_calculator: GraphicOptionCalculator):
        """
        Inicializa las opciones de visualización con valores predeterminados.
        """
        self.values_calculator: GraphicOptionCalculator = graphic_option_calculator
        # Opciones generales
        self.figure_size = (10, 8)      ##
        self.dpi = 100                  ##
        self.background_color = 'white' ##
        self.grid = False                ##
        self.title_font_size = 12       
        self.legend_font_size = 10      ##
        # 'default', 'dark_background', 'ggplot', etc.
        self.plot_style = 'default'     ##

        # Opciones para visualización de estructura
        self.node_size = 4              #####
        self.node_color = 'blue'            #####
        self.element_line_width = 1.0   #####
        self.element_color = 'blue'    #####
        self.support_size = self.values_calculator.support_size          #####
        self.support_color = 'green'    #####
        self.highlight_selected = True
        self.selected_color = 'red'
        self.node_labels = False              ###########
        self.element_labels = False           ###########
        self.label_font_size = 8                ##########
        self.node_label_color = '#ed0808'        ###########
        self.element_label_color = '#26a699'     ###########
        
        # CARGAS
        self.point_load = False               ###########
        self.point_load_color = '#282e3e'           ###########
        self.point_load_label = True                 ###########
        self.point_load_label_color = '#ff0892'                 ###########
        self.point_load_label_font_size = 8                 ###########
        self.point_load_length_arrow = 0.2 * self.values_calculator._length_mean ############
        self.point_moment_length_arrow = 0.05 * self.values_calculator._length_mean ########

        self.ratio_scale_load = self.values_calculator.ratio_scale_load ####
        self.nrof_arrows = self.values_calculator.nrof_arrows#######
        self.distributed_load_color = '#831f7a'########
        self.distributed_load_label = True #######
        self.distributed_load_label_color = '#511f74'#######
        self.distributed_load_label_font_size = 8#######

        # Opciones para visualización de resultados
        self.deformation_scale = 40 ########### # Factor de escala para deformaciones
        self.deformation_line_width = 1.0 ######### # Ancho de línea para deformaciones
        self.deformation_color = 'red' ########### # Color para deformaciones
        
        self.show_undeformed = True    ########## # Mostrar estructura sin deformar
        self.undeformed_style = 'dashed'  # Estilo para estructura sin deformar
        self.undeformed_color = '#aeacad'   ######### Color para estructura sin deformar

        # Opciones para diagramas de esfuerzos
        self.internal_forces_scale = 40  ###### # Factor de escala para diagramas de esfuerzos
        # Dibujar momentos en lado de tensión (C| ---|Ɔ)
        self.moment_on_tension_side = True
        self.axial_force_color = 'blue'    # Color para diagrama de axil
        self.shear_force_color = 'green'   # Color para diagrama de cortante
        self.bending_moment_color = 'red'  # Color para diagrama de momento
        self.slope = "red"                 # Color para diagrama de giros
        self.deflection = "blue"           # Color para diagrama de deflexiones

        # Opciones para contornos y diagramas
        self.fill_diagram = True         # Rellenar diagramas
        self.contour_type = 'filled'     # 'filled', 'lines', 'both'
        self.filling_type = 'solid'      # 'solid', 'color'
        # Transparencia para relleno (corregido: apha_filling -> alpha_filling)
        self.alpha_filling = 0.7
        self.colormap = 'jet'            # 'jet', 'viridis', 'coolwarm', etc.
        self.colormap_reverse = False    # Invertir mapa de colores
        self.show_colorbar = True        # Mostrar barra de colores
        self.colorbar_label = ''         # Etiqueta para barra de colores
        self.edge_color = 'black'        # Color de bordes en contornos
        self.alpha = 0.7                 # Transparencia para contornos
        self.num_contours = 20           # Número de niveles en contornos

        # Opciones para salida y guardado
        # Formato para guardar imágenes ('png', 'pdf', 'svg')
        self.save_format = 'png'
        self.save_dpi = 300               # DPI para guardar imágenes
        self.transparent_background = False  # Fondo transparente al guardar
        self.tight_layout = True ######         # Ajuste automático de layout

    def reset(self):
        """
        Reinicia todas las opciones a sus valores predeterminados.
        """
        self.__init__()

    def set_figure_options(self, width=None, height=None, dpi=None, background_color=None,
                           plot_style=None, grid=None, title_font_size=None,
                           axis_font_size=None, legend_font_size=None):
        """
        Configura opciones para la figura.

        Args:
            width (float, opcional): Ancho de la figura en pulgadas
            height (float, opcional): Alto de la figura en pulgadas
            dpi (int, opcional): Resolución de la figura
            background_color (str, opcional): Color de fondo
            plot_style (str, opcional): Estilo general del plot
            grid (bool, opcional): Si se muestra la cuadrícula
            title_font_size (int, opcional): Tamaño de fuente del título
            axis_font_size (int, opcional): Tamaño de fuente de los ejes
            legend_font_size (int, opcional): Tamaño de fuente de la leyenda
        """
        if width is not None:
            self.fig_width = width
        if height is not None:
            self.fig_height = height
        if dpi is not None:
            self.dpi = dpi
        if background_color is not None:
            self.background_color = background_color
        if plot_style is not None:
            self.plot_style = plot_style
        if grid is not None:
            self.grid = grid
        if title_font_size is not None:
            self.title_font_size = title_font_size
        if axis_font_size is not None:
            self.axis_font_size = axis_font_size
        if legend_font_size is not None:
            self.legend_font_size = legend_font_size

    def set_structure_display(self, node_size=None, node_color=None, element_line_width=None,
                              element_color=None, support_size=None, node_labels=None,
                              element_labels=None, highlight_selected=None, selected_color=None,
                              label_font_size=None):
        """
        Configura opciones para visualización de la estructura.

        Args:
            node_size (float, opcional): Tamaño de los nodos en visualización
            node_color (str, opcional): Color de los nodos
            element_line_width (float, opcional): Ancho de línea para elementos
            element_color (str, opcional): Color de los elementos
            support_size (float, opcional): Tamaño de los símbolos de apoyo
            node_labels (bool, opcional): Si se muestran etiquetas en nodos
            element_labels (bool, opcional): Si se muestran etiquetas en elementos
            highlight_selected (bool, opcional): Si se resaltan los elementos seleccionados
            selected_color (str, opcional): Color para elementos seleccionados
            label_font_size (int, opcional): Tamaño de fuente para etiquetas
        """
        if node_size is not None:
            self.node_size = node_size
        if node_color is not None:
            self.node_color = node_color
        if element_line_width is not None:
            self.element_line_width = element_line_width
        if element_color is not None:
            self.element_color = element_color
        if support_size is not None:
            self.support_size = support_size
        if node_labels is not None:
            self.node_labels = node_labels
        if element_labels is not None:
            self.element_labels = element_labels
        if highlight_selected is not None:
            self.highlight_selected = highlight_selected
        if selected_color is not None:
            self.selected_color = selected_color
        if label_font_size is not None:
            self.label_font_size = label_font_size

    def set_deformation_options(self, scale=None, show_undeformed=None,
                                undeformed_style=None, undeformed_color=None):
        """
        Configura opciones para visualización de deformaciones.

        Args:
            scale (float, opcional): Factor de escala para deformaciones
            show_undeformed (bool, opcional): Si se muestra la estructura sin deformar
            undeformed_style (str, opcional): Estilo de línea para estructura sin deformar
            undeformed_color (str, opcional): Color para estructura sin deformar
        """
        if scale is not None:
            self.deformation_scale = scale
        if show_undeformed is not None:
            self.show_undeformed = show_undeformed
        if undeformed_style is not None:
            self.undeformed_style = undeformed_style
        if undeformed_color is not None:
            self.undeformed_color = undeformed_color

    def set_contour_options(self, contour_type=None, colormap=None,
                            num_contours=None, show_colorbar=None, colorbar_label=None,
                            edge_color=None, alpha=None, filling_type=None,
                            alpha_filling=None, colormap_reverse=None):
        """
        Configura opciones para visualización de contornos.

        Args:
            contour_type (str, opcional): Tipo de contorno ('filled', 'lines', 'both')
            colormap (str, opcional): Mapa de colores a utilizar
            num_contours (int, opcional): Número de niveles en contornos
            show_colorbar (bool, opcional): Si se muestra barra de colores
            colorbar_label (str, opcional): Etiqueta para barra de colores
            edge_color (str, opcional): Color de bordes en contornos
            alpha (float, opcional): Transparencia para contornos
            filling_type (str, opcional): Tipo de relleno ('solid', 'color')
            alpha_filling (float, opcional): Transparencia para relleno
            colormap_reverse (bool, opcional): Si se invierte el mapa de colores
        """
        if contour_type is not None:
            self.contour_type = contour_type
        if colormap is not None:
            self.colormap = colormap
        if num_contours is not None:
            self.num_contours = num_contours
        if show_colorbar is not None:
            self.show_colorbar = show_colorbar
        if colorbar_label is not None:
            self.colorbar_label = colorbar_label
        if edge_color is not None:
            self.edge_color = edge_color
        if alpha is not None:
            self.alpha = alpha
        if filling_type is not None:
            self.filling_type = filling_type
        if alpha_filling is not None:
            self.alpha_filling = alpha_filling
        if colormap_reverse is not None:
            self.colormap_reverse = colormap_reverse

    def set_internal_forces_options(self, scale=None, shear_color=None,
                                    moment_color=None, axial_color=None,
                                    slope=None, deflection=None,
                                    fill_diagram=None, moment_on_tension_side=None
                                    ):
        """
        Configura opciones para visualización de esfuerzos internos.

        Args:
            scale (float, opcional): Factor de escala para diagramas
            shear_color (str, opcional): Color para diagrama de cortante
            moment_color (str, opcional): Color para diagrama de momento
            axial_color (str, opcional): Color para diagrama de axil
            fill_diagram (bool, opcional): Si se rellenan los diagramas
            moment_on_tension_side (bool, opcional): Si se dibujan momentos en lado de tensión
        """
        if scale is not None:
            self.internal_forces_scale = scale
        if shear_color is not None:
            self.shear_force_color = shear_color
        if moment_color is not None:
            self.bending_moment_color = moment_color
        if axial_color is not None:
            self.axial_force_color = axial_color
        if fill_diagram is not None:
            self.fill_diagram = fill_diagram
        if moment_on_tension_side is not None:
            self.moment_on_tension_side = moment_on_tension_side
        if slope is not None:
            self.slope = slope
        if deflection is not None:
            self.deflection = deflection

    def set_save_options(self, format=None, dpi=None, transparent=None, tight_layout=None):
        """
        Configura opciones para guardar figuras.

        Args:
            format (str, opcional): Formato para guardar ('png', 'pdf', 'svg', etc.)
            dpi (int, opcional): Resolución para archivos guardados
            transparent (bool, opcional): Si se usa fondo transparente
            tight_layout (bool, opcional): Si se ajusta automáticamente el layout
        """
        if format is not None:
            self.save_format = format
        if dpi is not None:
            self.save_dpi = dpi
        if transparent is not None:
            self.transparent_background = transparent
        if tight_layout is not None:
            self.tight_layout = tight_layout

    def copy(self):
        """
        Crea una copia independiente de las opciones actuales.

        Returns:
            PlotterOptions: Nueva instancia con las mismas opciones que la actual
        """
        new_options = PlotterOptions()
        for attr_name, attr_value in self.__dict__.items():
            setattr(new_options, attr_name, attr_value)
        return new_options
