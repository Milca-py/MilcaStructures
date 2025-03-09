
from utils import rotate_xy, traslate_xy
from typing import TYPE_CHECKING
from core.post_processing import (
    values_axial_force,
    values_shear_force,
    values_bending_moment,
    values_slope,
    values_deflection,
    values_deformed, 
    values_rigid_deformed
)
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, List
import matplotlib.pyplot as plt
from display.options import GraphicOption

from display.suports import (
    support_ttt,
    support_ttf,
    support_tft,
    support_ftt,
    support_tff,
    support_ftf,
    support_fff,
    support_fft
)

from display.load import (
    graphic_n_arrow,
    graphic_one_arrow,
    moment_fancy_arrow,
    moment_n_arrow
)

if TYPE_CHECKING:
    from core.system import SystemMilcaModel

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from core.element import Element
    from core.system import SystemMilcaModel





class PlotterValues:
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system

    def structure(self):
        """Devuelve los valores necesarios para graficar la estructura.

        Returns:
            Tuple[Dict[int, Tuple[float, float]], Dict[int, List[Tuple[float, float]]]]: 
            Valores de los nodos y elementos para graficar.
        """
        # obtener los valores para graficar los nodos {id: (x, y)}
        node_values = {}
        for node in self.system.node_map.values():
            node_values[node.id] = (node.vertex.x, node.vertex.y)

        # obtener los valores para graficar los elementos {id: [(x1, y1), (x2, y2)]}
        element_values = {}
        for element in self.system.element_map.values():
            node_i, node_j = element.node_i, element.node_j
            element_values[element.id] = [
                (node_i.vertex.x, node_i.vertex.y), (node_j.vertex.x, node_j.vertex.y)]

        # obtener los elementos {id: {q_i, q_j, p_i, p_j, m_i, m_j}}
        load_elements = {}
        for load_pattern in self.system.load_pattern_map.values():
            for id_element, load in load_pattern.distributed_loads_map.items():
                load_elements[id_element] = load.to_dict()

        # y nodos cargados {id: {fx, fy, mz}}
        load_nodes = {}
        for load_pattern in self.system.load_pattern_map.values():
            for id_node, load in load_pattern.point_loads_map.items():
                load_nodes[id_node] = load.to_dict()

        # obtener los nodos restrigidos {id: (restricciones)}
        restrained_nodes = {}
        for node in self.system.node_map.values():
            if node.restraints != (False, False, False):
                restrained_nodes[node.id] = node.restraints

        return node_values, element_values, load_elements, load_nodes, restrained_nodes

    def axial_force(self):
        pass

    def shear_force(self):
        pass

    def bending_moment(self):
        pass

    def displacements(self):
        pass


class Plotter:
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system
        self.options = GraphicOption(self.system)
        self.values = PlotterValues(self.system)
        self.axes: List["Axes"] = []
        self.fig: Optional["Figure"] = None
        self.__start_plot(self.options.figsize)

    def __start_plot(
        self, figsize: Optional[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Inicia la gráfica creando una ventana de Matplotlib con el tamaño especificado.

        Args:
            figsize (Optional[Tuple[float, float]]): Tamaño de la figura (ancho, alto).

        Returns:
            Tuple[float, float]: Tamaño de la figura (ancho, alto).
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
        color_point_loads="blue",
        color_distributed_loads="blue",
        show: bool = True
    ) -> None:

        # ploteo de los nodos y elementos
        self._plot_nodes(axes_i=axes_i, labels=labels_nodes,
                            color_node=color_nodes, color_labels=color_labels_node)
        self._plot_element(axes_i=axes_i, labels=labels_elements,
                            color_element=color_elements, color_labels=color_labels_element)
        # ploteo de los apoyos
        self._plot_supports(axes_i=axes_i, color="green")
        # # ploteo de las cargas puntuales
        self._plot_point_loads(axes_i=axes_i, color=color_point_loads, label=labels_point_loads)
        # # ploteo de las cargas distribuidas
        self._plot_distributed_loads(axes_i=axes_i, color=color_distributed_loads, label=labels_distributed_loads)
        if show:
            plt.show()

    def axial_force(self):
        "Grafica las fuerzas axiales"

    def bending_moment(self):
        "Grafica los momentos flectores"

    def shear_force(self):
        "Grafica las fuerzas cortantes"

    def reaction_force(self):
        "Grafica las fuerzas de reacción"

    def displacements(self):
        "Grafica las deformaciones de la estructura"

    def results_plot(self):
        "Grafica los LA ESTRUCTURA, DFA, DFC, DMF, DG, DEFORMADA"

    def _plot_element(
        self,
        labels: bool = False,
        axes_i: int = 0,
        color_element: str = "blue",
        color_labels: str = "red"
    ) -> None:
        # ploteo de los elementos
        for id_element, element in self.values.structure()[1].items():
            self.axes[axes_i].plot([element[0][0], element[1][0]], [element[0][1], element[1][1]],
                                   color=color_element,
                                   lw=1,
                                #    linestyle="--"
                                   )

            # ploteo de las etiquetas de los elementos
            if labels:
                self.axes[axes_i].text(
                    (element[0][0] + element[1][0]) / 2,
                    (element[0][1] + element[1][1]) / 2,
                    f"{id_element}",
                    fontsize=8,
                    color=color_labels,
                    # ha="center",
                    # va="center"
                )

    def _plot_nodes(
        self,
        labels: bool = False,
        axes_i: int = 0,
        color_node: str = "red",
        color_labels: str = "red"
    ) -> None:
        # ploteo de los nodos
        for id_node, node in self.values.structure()[0].items():
            self.axes[axes_i].plot(node[0], node[1],
                                   color=color_node,
                                   )

            # ploteo de las etiquetas de los nodos
            if labels:
                self.axes[axes_i].text(
                    node[0], node[1],
                    f"{id_node}",
                    fontsize=8,
                    color=color_labels,
                    # ha="center",
                    # va="center"
                )

    def _plot_supports(
        self,
        axes_i: int = 0,
        color: str = "green",
    ) -> None:
        # ploteo de los apoyos
        for id_node, restrains in self.values.structure()[4].items():
            if restrains == (True, True, True):
                support_ttt(self.axes[axes_i],
                            self.values.structure()[0][id_node][0],
                            self.values.structure()[0][id_node][1],
                            self.options.support_size,
                            color)
            elif restrains == (False, False, True):
                support_fft(self.axes[axes_i],
                            self.values.structure()[0][id_node][0],
                            self.values.structure()[0][id_node][1],
                            self.options.support_size,
                            color)
            elif restrains == (False, True, False):
                support_ftf(self.axes[axes_i],
                            self.values.structure()[0][id_node][0],
                            self.values.structure()[0][id_node][1],
                            self.options.support_size,
                            color)
            elif restrains == (True, False, False):
                support_tff(self.axes[axes_i],
                            self.values.structure()[0][id_node][0],
                            self.values.structure()[0][id_node][1],
                            self.options.support_size,
                            color)
            elif restrains == (False, True, True):
                support_ftt(self.axes[axes_i],
                            self.values.structure()[0][id_node][0],
                            self.values.structure()[0][id_node][1],
                            self.options.support_size,
                            color)
            elif restrains == (True, False, True):
                support_tft(self.axes[axes_i],
                            self.values.structure()[0][id_node][0],
                            self.values.structure()[0][id_node][1],
                            self.options.support_size,
                            color)
            elif restrains == (True, True, False):
                support_ttf(self.axes[axes_i],
                            self.values.structure()[0][id_node][0],
                            self.values.structure()[0][id_node][1],
                            self.options.support_size,
                            color)
            elif restrains == (False, False, False):
                pass
            else:
                raise ValueError(
                    "Restricciones no válidas, no se puede plotear el apoyo.")

    def _plot_point_loads(
        self,
        axes_i: int = 0,
        color: str = "blue",
        label: bool = False
    ) -> None:
        # ploteo de las cargas puntuales
        for id_node, load in self.values.structure()[3].items():
            coords = self.values.structure()[0][id_node]
            if load["fx"] != 0:
                l = graphic_one_arrow(
                    x=coords[0],
                    y=coords[1],
                    load=load["fx"],
                    length_arrow=(0.2) * self.options._length_mean,
                    # load_mean=self.options._fx_mean*np.sign(load["fx"]),
                    angle=0,
                    ax=self.axes[axes_i],
                    # ratio_scale=self.options.ratio_scale_load,
                    color=color,
                    label=label,
                    color_label=color
                )

            if load["fy"] != 0:
                graphic_one_arrow(
                    x=coords[0],
                    y=coords[1],
                    load=load["fy"],
                    length_arrow=(0.2) * self.options._length_mean,
                    # load_mean=self.options._fx_mean*np.sign(load["fy"]),
                    angle=np.pi/2,
                    ax=self.axes[axes_i],
                    # ratio_scale=self.options.ratio_scale_force,
                    color=color,
                    label=label,
                    color_label=color
                )

            if load["mz"] != 0:
                moment_fancy_arrow(
                    ax=self.axes[axes_i],
                    x=coords[0],
                    y=coords[1],
                    moment=load["mz"],
                    radio=0.05 * self.options._length_mean,
                    # ratio_scale=self.options.ratio_scale_force,
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
        # ploteo de las cargas distribuidas
        for id_element, load in self.values.structure()[2].items():
            coords = self.values.structure()[1][id_element]
            length = np.sqrt((coords[1][0] - coords[0][0])
                             ** 2 + (coords[1][1] - coords[0][1])**2)
            angle_rotation = np.arctan2(
                coords[1][1] - coords[0][1], coords[1][0] - coords[0][0])
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
            if load["m_i"] != 0 or load["m_j"] != 0:
                # moment_n_arrow(
                #     ax=self.axes[axes_i],
                #     x=coords[0][0],
                #     y=coords[0][1],
                #     load_i=load["m_i"],
                #     load_j=load["m_j"],
                #     length=length,
                #     radio=0.04 * self.options._length_mean,
                #     # ratio_scale=self.options.ratio_scale_load,
                #     nrof_arrows=self.options.nrof_arrows,
                #     color=color,
                #     angle_rotation=angle_rotation,
                #     clockwise=True
                # )
                raise NotImplementedError(
                    "Momentos distribuidos no implementados.")


    def show_diagrams(self, type: str, axes_i: int = 0, fill: bool = True, npp: int = 40, escala: int = 0.03, show: bool = True) -> None:
        
        for element in self.system.element_map.values():
            plotting_element_diagrams(self.axes[axes_i], element, type, fill, npp, escala)
        if show:
            plt.show()
    
    def show_deformed(self, escala: float = 1,  axes_i: int = 0, show: bool = True) -> None:
        self._plot_element(color_element="#97a3af")
        for element in self.system.element_map.values():
            x, y = values_deformed(element, escala)
            self.axes[axes_i].plot(x, y, lw=1, color='#4b35a0')
        if show:
            plt.show()
    
    def show_rigid_deformed(self, escala: float = 1,  axes_i: int = 0, show: bool = True) -> None:
        for element in self.system.element_map.values():
            x, y = values_rigid_deformed(element, escala)
            self.axes[axes_i].plot(x, y, lw=1, color='#54becb',
                                linestyle="--"
                                )
        if show:
            plt.show()


def plotting_element_diagrams(ax: "Axes",  element: "Element", type: str,
                              fill: bool, npp: int, escala: int) -> None:

    if type == "axial_force":
        x, n = values_axial_force(element, escala, npp)
        # n = -n    
    elif type == "shear_force":
        x, n = values_shear_force(element, escala, npp)
        n = n
    elif type == "bending_moment":
        x, n = values_bending_moment(element, escala, npp)
        # n = n
    elif type == "slope":
        x, n = values_slope(element, escala, npp)
    elif type == "deflection":
        x, n = values_deflection(element, escala, npp)
    else:
        raise ValueError("Tipo de diagrama no válido, los tipos válidos son: 'axial_force', 'shear_force', 'bending_moment', 'spin' y 'deflection'.")

    coord_elem = np.array([np.array([element.node_i.vertex.coordinates]),
                           np.array([element.node_j.vertex.coordinates])])

    Nxy = np.column_stack((x, n))
    Nxy = rotate_xy(Nxy, element.angle_x, 0, 0)
    Nxy = traslate_xy(Nxy, *element.node_i.vertex.coordinates)
    Nxy = np.insert(Nxy, 0, coord_elem[0], axis=0)
    Nxy = np.append(Nxy, coord_elem[1], axis=0)

    ax.plot(Nxy[:, 0], Nxy[:, 1], label="Axial Force", lw=0.5, color='orange')
    if fill:
        NNxy = np.append(Nxy, coord_elem[0], axis=0)
        plt.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue',
                 alpha=0.5)  # Sombrea la región
