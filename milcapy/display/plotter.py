from typing import TYPE_CHECKING, Optional, Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt

from milcapy.utils import rotate_xy, traslate_xy
from milcapy.core.post_processing import (
    values_axial_force,
    values_shear_force,
    values_bending_moment,
    values_slope,
    values_deflection,
    values_deformed,
    values_rigid_deformed
)
from milcapy.display.options import GraphicOption
from milcapy.display.suports import (
    support_ttt,
    support_ttf,
    support_tft,
    support_ftt,
    support_tff,
    support_ftf,
    support_fff,
    support_fft
)
from milcapy.display.load import (
    graphic_n_arrow,
    graphic_one_arrow,
    moment_fancy_arrow,
    moment_n_arrow
)

if TYPE_CHECKING:
    from milcapy.core.system import SystemMilcaModel
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from milcapy.core.element import Element


class PlotterValues:
    """Clase que proporciona los valores necesarios para graficar la estructura."""
    
    def __init__(self, system: 'SystemMilcaModel'):
        """
        Inicializa un objeto PlotterValues.
        
        Args:
            system: Sistema de estructura a graficar.
        """
        self.system = system

    def structure(self) -> Tuple[Dict[int, Tuple[float, float]], 
                                Dict[int, List[Tuple[float, float]]], 
                                Dict[int, Dict], 
                                Dict[int, Dict], 
                                Dict[int, Tuple[bool, bool, bool]]]:
        """
        Devuelve los valores necesarios para graficar la estructura.

        Returns:
            Tuple que contiene:
            - node_values: Diccionario de nodos {id: (x, y)}
            - element_values: Diccionario de elementos {id: [(x1, y1), (x2, y2)]}
            - load_elements: Diccionario de cargas distribuidas {id_element: {q_i, q_j, p_i, p_j, m_i, m_j}}
            - load_nodes: Diccionario de cargas puntuales {id_node: {fx, fy, mz}}
            - restrained_nodes: Diccionario de nodos restringidos {id: (restricciones)}
        """
        # Obtener los valores para graficar los nodos {id: (x, y)}
        node_values = {node.id: (node.vertex.x, node.vertex.y) 
                      for node in self.system.node_map.values()}

        # Obtener los valores para graficar los elementos {id: [(x1, y1), (x2, y2)]}
        element_values = {}
        for element in self.system.element_map.values():
            node_i, node_j = element.node_i, element.node_j
            element_values[element.id] = [
                (node_i.vertex.x, node_i.vertex.y), 
                (node_j.vertex.x, node_j.vertex.y)
            ]

        # Obtener los elementos cargados {id: {q_i, q_j, p_i, p_j, m_i, m_j}}
        load_elements = {}
        for load_pattern in self.system.load_pattern_map.values():
            for id_element, load in load_pattern.distributed_loads_map.items():
                load_elements[id_element] = load.to_dict()

        # Obtener nodos cargados {id: {fx, fy, mz}}
        load_nodes = {}
        for load_pattern in self.system.load_pattern_map.values():
            for id_node, load in load_pattern.point_loads_map.items():
                load_nodes[id_node] = load.to_dict()

        # Obtener los nodos restringidos {id: (restricciones)}
        restrained_nodes = {}
        for node in self.system.node_map.values():
            if node.restraints != (False, False, False):
                restrained_nodes[node.id] = node.restraints

        return node_values, element_values, load_elements, load_nodes, restrained_nodes

    def axial_force(self):
        """Método a implementar para obtener valores de fuerzas axiales."""
        pass

    def shear_force(self):
        """Método a implementar para obtener valores de fuerzas cortantes."""
        pass

    def bending_moment(self):
        """Método a implementar para obtener valores de momentos flectores."""
        pass

    def displacements(self):
        """Método a implementar para obtener valores de desplazamientos."""
        pass


class Plotter:
    """Clase para graficar estructuras y resultados de análisis."""
    
    def __init__(self, system: 'SystemMilcaModel'):
        """
        Inicializa un objeto Plotter.
        
        Args:
            system: Sistema de estructura a graficar.
        """
        self.system = system
        self.options = GraphicOption(self.system)
        self.values = PlotterValues(self.system)
        self.axes: List["Axes"] = []
        self.fig: Optional["Figure"] = None
        self.__start_plot(self.options.figsize)

    def __start_plot(self, figsize: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Inicia la gráfica creando una ventana de Matplotlib con el tamaño especificado.

        Args:
            figsize: Tamaño de la figura (ancho, alto).

        Returns:
            Tamaño de la figura (ancho, alto).
        """
        plt.close("all")
        self.fig = plt.figure(figsize=figsize)
        self.axes = [self.fig.add_subplot(111)]
        plt.tight_layout()
        plt.axis('equal')
        return (self.fig.get_figwidth(), self.fig.get_figheight())

    def plot_structure(
        self,
        axes_i: int = 0,
        labels_nodes: bool = False,
        labels_elements: bool = False,
        color_nodes: str = "red",
        color_elements: str = "blue",
        color_labels_node: str = "red",
        color_labels_element: str = "red",
        labels_point_loads: bool = False,
        labels_distributed_loads: bool = False,
        color_point_loads: str = "blue",
        color_distributed_loads: str = "blue",
        show: bool = True
    ) -> None:
        """
        Grafica la estructura completa.
        
        Args:
            axes_i: Índice del eje donde graficar.
            labels_nodes: Mostrar etiquetas de nodos.
            labels_elements: Mostrar etiquetas de elementos.
            color_nodes: Color de los nodos.
            color_elements: Color de los elementos.
            color_labels_node: Color de las etiquetas de nodos.
            color_labels_element: Color de las etiquetas de elementos.
            labels_point_loads: Mostrar etiquetas de cargas puntuales.
            labels_distributed_loads: Mostrar etiquetas de cargas distribuidas.
            color_point_loads: Color de las cargas puntuales.
            color_distributed_loads: Color de las cargas distribuidas.
            show: Mostrar el gráfico inmediatamente.
        """
        # Ploteo de los nodos y elementos
        self._plot_nodes(
            axes_i=axes_i, 
            labels=labels_nodes,
            color_node=color_nodes, 
            color_labels=color_labels_node
        )
        self._plot_element(
            axes_i=axes_i, 
            labels=labels_elements,
            color_element=color_elements, 
            color_labels=color_labels_element
        )
        # Ploteo de los apoyos
        self._plot_supports(axes_i=axes_i, color="green")
        # Ploteo de las cargas puntuales
        self._plot_point_loads(
            axes_i=axes_i, 
            color=color_point_loads, 
            label=labels_point_loads
        )
        # Ploteo de las cargas distribuidas
        self._plot_distributed_loads(
            axes_i=axes_i, 
            color=color_distributed_loads, 
            label=labels_distributed_loads
        )
        
        if show:
            plt.show()

    def axial_force(self):
        """Grafica las fuerzas axiales."""
        pass

    def bending_moment(self):
        """Grafica los momentos flectores."""
        pass

    def shear_force(self):
        """Grafica las fuerzas cortantes."""
        pass

    def reaction_force(self):
        """Grafica las fuerzas de reacción."""
        pass

    def displacements(self):
        """Grafica las deformaciones de la estructura."""
        pass

    def results_plot(self):
        """Grafica la estructura, DFA, DFC, DMF, DG, DEFORMADA."""
        pass

    def _plot_element(
        self,
        labels: bool = False,
        axes_i: int = 0,
        color_element: str = "blue",
        color_labels: str = "red"
    ) -> None:
        """
        Grafica los elementos de la estructura.
        
        Args:
            labels: Mostrar etiquetas de elementos.
            axes_i: Índice del eje donde graficar.
            color_element: Color de los elementos.
            color_labels: Color de las etiquetas.
        """
        # Ploteo de los elementos
        structure_values = self.values.structure()
        for id_element, element in structure_values[1].items():
            x_coords = [element[0][0], element[1][0]]
            y_coords = [element[0][1], element[1][1]]
            self.axes[axes_i].plot(
                x_coords, 
                y_coords,
                color=color_element,
                lw=1
            )

            # Ploteo de las etiquetas de los elementos
            if labels:
                self.axes[axes_i].text(
                    (element[0][0] + element[1][0]) / 2,
                    (element[0][1] + element[1][1]) / 2,
                    f"{id_element}",
                    fontsize=8,
                    color=color_labels
                )

    def _plot_nodes(
        self,
        labels: bool = False,
        axes_i: int = 0,
        color_node: str = "red",
        color_labels: str = "red"
    ) -> None:
        """
        Grafica los nodos de la estructura.
        
        Args:
            labels: Mostrar etiquetas de nodos.
            axes_i: Índice del eje donde graficar.
            color_node: Color de los nodos.
            color_labels: Color de las etiquetas.
        """
        # Ploteo de los nodos
        structure_values = self.values.structure()
        for id_node, node in structure_values[0].items():
            self.axes[axes_i].plot(
                node[0], 
                node[1],
                color=color_node
            )

            # Ploteo de las etiquetas de los nodos
            if labels:
                self.axes[axes_i].text(
                    node[0], 
                    node[1],
                    f"{id_node}",
                    fontsize=8,
                    color=color_labels
                )

    def _plot_supports(
        self,
        axes_i: int = 0,
        color: str = "green",
    ) -> None:
        """
        Grafica los apoyos de la estructura.
        
        Args:
            axes_i: Índice del eje donde graficar.
            color: Color de los apoyos.
            
        Raises:
            ValueError: Si las restricciones no son válidas.
        """
        # Diccionario de funciones de soporte para optimizar el código
        support_functions = {
            (True, True, True): support_ttt,
            (False, False, True): support_fft,
            (False, True, False): support_ftf,
            (True, False, False): support_tff,
            (False, True, True): support_ftt,
            (True, False, True): support_tft,
            (True, True, False): support_ttf,
            (False, False, False): None
        }
        
        structure_values = self.values.structure()
        for id_node, restrains in structure_values[4].items():
            node_coords = structure_values[0][id_node]
            support_func = support_functions.get(restrains)
            
            if support_func:
                support_func(
                    self.axes[axes_i],
                    node_coords[0],
                    node_coords[1],
                    self.options.support_size,
                    color
                )
            elif restrains != (False, False, False):
                raise ValueError("Restricciones no válidas, no se puede plotear el apoyo.")

    def _plot_point_loads(
        self,
        axes_i: int = 0,
        color: str = "blue",
        label: bool = False
    ) -> None:
        """
        Grafica las cargas puntuales.
        
        Args:
            axes_i: Índice del eje donde graficar.
            color: Color de las cargas.
            label: Mostrar etiquetas de cargas.
        """
        structure_values = self.values.structure()
        for id_node, load in structure_values[3].items():
            coords = structure_values[0][id_node]
            length_arrow = 0.2 * self.options._length_mean
            
            # Fuerza en dirección X
            if load["fx"] != 0:
                graphic_one_arrow(
                    x=coords[0],
                    y=coords[1],
                    load=load["fx"],
                    length_arrow=length_arrow,
                    angle=0,
                    ax=self.axes[axes_i],
                    color=color,
                    label=label,
                    color_label=color
                )
            
            # Fuerza en dirección Y
            if load["fy"] != 0:
                graphic_one_arrow(
                    x=coords[0],
                    y=coords[1],
                    load=load["fy"],
                    length_arrow=length_arrow,
                    angle=np.pi/2,
                    ax=self.axes[axes_i],
                    color=color,
                    label=label,
                    color_label=color
                )
            
            # Momento en Z
            if load["mz"] != 0:
                moment_fancy_arrow(
                    ax=self.axes[axes_i],
                    x=coords[0],
                    y=coords[1],
                    moment=load["mz"],
                    radio=0.05 * self.options._length_mean,
                    color=color,
                    clockwise=True,
                    label=label,
                    color_label=color
                )

    def _plot_distributed_loads(
        self,
        axes_i: int = 0,
        color: str = "blue",
        label: bool = True
    ) -> None:
        """
        Grafica las cargas distribuidas.
        
        Args:
            axes_i: Índice del eje donde graficar.
            color: Color de las cargas.
            label: Mostrar etiquetas de cargas.
            
        Raises:
            NotImplementedError: Si se intenta graficar momentos distribuidos.
        """
        structure_values = self.values.structure()
        for id_element, load in structure_values[2].items():
            coords = structure_values[1][id_element]
            
            # Calcular longitud y ángulo de rotación del elemento
            x_diff = coords[1][0] - coords[0][0]
            y_diff = coords[1][1] - coords[0][1]
            length = np.sqrt(x_diff**2 + y_diff**2)
            angle_rotation = np.arctan2(y_diff, x_diff)
            
            # Cargas verticales
            if load["q_i"] != 0 or load["q_j"] != 0:
                graphic_n_arrow(
                    x=coords[0][0],
                    y=coords[0][1],
                    load_i=load["q_i"],
                    load_j=load["q_j"],
                    angle=np.pi/2,
                    length=length,
                    ax=self.axes[axes_i],
                    ratio_scale=self.options.ratio_scale_load,
                    nrof_arrows=self.options.nrof_arrows,
                    color=color,
                    angle_rotation=angle_rotation,
                    label=label,
                    color_label=color
                )
            
            # Cargas axiales
            if load["p_i"] != 0 or load["p_j"] != 0:
                graphic_n_arrow(
                    x=coords[0][0],
                    y=coords[0][1],
                    load_i=load["p_i"],
                    load_j=load["p_j"],
                    angle=0,
                    length=length,
                    ax=self.axes[axes_i],
                    ratio_scale=self.options.ratio_scale_axial,
                    nrof_arrows=self.options.nrof_arrows,
                    color=color,
                    angle_rotation=angle_rotation,
                    label=label,
                    color_label=color
                )
            
            # Momentos distribuidos (no implementados)
            if load["m_i"] != 0 or load["m_j"] != 0:
                raise NotImplementedError("Momentos distribuidos no implementados.")

    def show_diagrams(
        self, 
        type: str, 
        axes_i: int = 0, 
        fill: bool = True, 
        npp: int = 40, 
        escala: float = 0.03, 
        show: bool = True
    ) -> None:
        """
        Muestra diagramas de esfuerzos para cada elemento.
        
        Args:
            type: Tipo de diagrama ('axial_force', 'shear_force', 'bending_moment', 'slope', 'deflection').
            axes_i: Índice del eje donde graficar.
            fill: Rellenar el área del diagrama.
            npp: Número de puntos para discretizar el elemento.
            escala: Factor de escala para visualización.
            show: Mostrar el gráfico inmediatamente.
        """
        for element in self.system.element_map.values():
            plotting_element_diagrams(self.axes[axes_i], element, type, fill, npp, escala)
        
        if show:
            plt.show()
    
    def show_deformed(
        self, 
        escala: float = 1, 
        axes_i: int = 0, 
        show: bool = True
    ) -> None:
        """
        Muestra la deformada de la estructura.
        
        Args:
            escala: Factor de escala para visualización.
            axes_i: Índice del eje donde graficar.
            show: Mostrar el gráfico inmediatamente.
        """
        self._plot_element(color_element="#97a3af")
        
        for element in self.system.element_map.values():
            x, y = values_deformed(element, escala)
            self.axes[axes_i].plot(x, y, lw=1, color='#4b35a0')
        
        if show:
            plt.show()
    
    def show_rigid_deformed(
        self, 
        escala: float = 1, 
        axes_i: int = 0, 
        show: bool = True
    ) -> None:
        """
        Muestra la deformada rígida de la estructura.
        
        Args:
            escala: Factor de escala para visualización.
            axes_i: Índice del eje donde graficar.
            show: Mostrar el gráfico inmediatamente.
        """
        for element in self.system.element_map.values():
            x, y = values_rigid_deformed(element, escala)
            self.axes[axes_i].plot(
                x, 
                y, 
                lw=1, 
                color='#54becb',
                linestyle="--"
            )
        
        if show:
            plt.show()


def plotting_element_diagrams(
    ax: "Axes", 
    element: "Element", 
    type: str,
    fill: bool, 
    npp: int, 
    escala: float
) -> None:
    """
    Grafica diagramas para un elemento específico.
    
    Args:
        ax: Eje de matplotlib donde graficar.
        element: Elemento a graficar.
        type: Tipo de diagrama ('axial_force', 'shear_force', 'bending_moment', 'slope', 'deflection').
        fill: Rellenar el área del diagrama.
        npp: Número de puntos para discretizar el elemento.
        escala: Factor de escala para visualización.
        
    Raises:
        ValueError: Si el tipo de diagrama no es válido.
    """
    # Seleccionar la función correspondiente según el tipo de diagrama
    diagram_functions = {
        "axial_force": values_axial_force,
        "shear_force": values_shear_force,
        "bending_moment": values_bending_moment,
        "slope": values_slope,
        "deflection": values_deflection
    }
    
    if type not in diagram_functions:
        raise ValueError(
            "Tipo de diagrama no válido, los tipos válidos son: 'axial_force', 'shear_force', "
            "'bending_moment', 'slope' y 'deflection'."
        )
    
    # Obtener valores del diagrama
    x, n = diagram_functions[type](element, escala, npp)
    
    # Obtener coordenadas del elemento
    coord_elem = np.array([
        np.array([element.node_i.vertex.coordinates]),
        np.array([element.node_j.vertex.coordinates])
    ])

    # Transformar coordenadas
    Nxy = np.column_stack((x, n))
    Nxy = rotate_xy(Nxy, element.angle_x, 0, 0)
    Nxy = traslate_xy(Nxy, *element.node_i.vertex.coordinates)
    Nxy = np.insert(Nxy, 0, coord_elem[0], axis=0)
    Nxy = np.append(Nxy, coord_elem[1], axis=0)

    # Graficar diagrama
    ax.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')
    
    # Rellenar diagrama si se solicita
    if fill:
        NNxy = np.append(Nxy, coord_elem[0], axis=0)
        ax.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)