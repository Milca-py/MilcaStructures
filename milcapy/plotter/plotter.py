from ast import Dict
from typing import TYPE_CHECKING, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from milcapy.utils import InternalForceType

from milcapy.utils import rotate_xy, traslate_xy
from milcapy.plotter.suports import (
    support_ttt, support_ttf, support_tft,
    support_ftt, support_tff, support_ftf, support_fft
)
from milcapy.plotter.load import (
    graphic_one_arrow, moment_fancy_arrow, graphic_n_arrow
)
from milcapy.plotter.plotter_values import PlotterValues
from milcapy.plotter.options import PlotterOptions

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class Plotter:
    def __init__(
        self,
        model: 'SystemMilcaModel',
    ) -> None:
        self.model = model
        self.plotter_values: Dict[str, 'PlotterValues'] = {}
        self.figure: Figure = None
        self.axes: Axes = None
        self.current_values: 'PlotterValues' = PlotterValues(self.model,
                                                             list(self.model.results.keys())[0])
        self.initialize_figure()  # Inicializar figura

        # todos los que tengan 🐍 se calculan para cada load pattern
        # todos los que tengan ✅ ya estan implementados
        # ! SOLO SE PLOTEAN LOS LOAD_PATTERN QUE ESTAN ANALIZADOS
        # ? (SOLO LOS QUE TIENEN RESULTADOS EN MODEL.RESULTS)
        # nodos
        self.nodes = {}              # ✅visibilidad, color
        # nodos deformados
        self.deformed_nodes = {}    # 🐍 interactividad al pasar el ratón

        # miembros
        self.members = {}           # ✅visibilidad, color
        # forma deformada
        self.deformed_shape = {}    # ✅🐍 visibilidad, setdata, setType: colorbar
        # forma regida de la deformada
        self.rigid_deformed_shape = {}  # ✅🐍 visibilidad, setdata, setType: colorbar
        # cargas puntuales
        self.point_loads = {}       # ✅🐍 visibilidad
        # cargas distribuidas
        self.distributed_loads = {}  # ✅🐍 visibilidad
        # fuerzas internas (line2D: borde)
        self.internal_forces = {}   # 🐍 visibilidad, setdata, setType: colorbar
        # fillings for internal forces
        self.fillings = {}          # 🐍 visibilidad, setdata, setType: colorbar
        # apooyos
        self.supports = {}          # ✅visibilidad, color, setdata
        # apoyos dezplados
        self.displaced_supports = {}  # 🐍 visibilidad, color, setdata
        # etiquetas
        self.node_labels = {}            # ✅visibilidad, setdata
        self.member_labels = {}          # ✅visibilidad, setdata
        self.point_load_labels = {}      # ✅🐍 visibilidad, setdata
        self.distributed_load_labels = {}  # ✅🐍 visibilidad, setdata
        self.internal_forces_labels = {}  # 🐍 visibilidad, setdata
        self.reactions_labels = {}       # 🐍 visibilidad, setdata
        self.displacement_labels = {}    # 🐍 visibilidad, setdata
        # reacciones
        self.reactions = {}         # 🐍 visibilidad, setdata

        # fuerzas internas (line2D: borde, polygon: relleno)
        self.axial_force = {}
        self.shear_force = {}
        self.bending_moment = {}

    @property
    def plotter_options(self) -> 'PlotterOptions':
        return self.model.plotter_options

    @plotter_options.setter
    def plotter_options(self, value: 'PlotterOptions') -> None:
        self.model.plotter_options = value

    @property
    def current_load_pattern(self) -> Optional[str]:
        return self.model.current_load_pattern

    @current_load_pattern.setter
    def current_load_pattern(self, value: Optional[str]):
        self.model.current_load_pattern = value

    def initialize_plot(self):
        """Plotea por primera y unica vez (crea los objetos artist)"""
        self.plotter_options.load_max(list(self.model.results.keys())[0])
        self.plot_nodes()
        self.plot_members()
        self.plot_supports()
        self.plot_node_labels()
        self.plot_member_labels()
        for load_pattern_name in self.model.results.keys():
            self.plotter_options.load_max(load_pattern_name)
            self.current_load_pattern = load_pattern_name
            self.current_values = self.get_plotter_values(load_pattern_name)
            self.plot_point_loads()
            self.plot_distributed_loads()
            self.plot_rigid_deformed()
            self.plot_deformed()
            self.plot_axial_force()
            self.plot_shear_force()
            self.plot_bending_moment()
            self.plot_reactions()
            self.plot_displaced_nodes()

        # actualizar pattern actual al primero
        self.current_load_pattern = list(self.model.results.keys())[0]
        self.update_change()

    def update_change(self):
        """Oculta atists para todos los load patterns excepto el actual"""
        pt_cache = self.current_load_pattern
        for load_pattern_name in self.model.results.keys():
            self.current_load_pattern = load_pattern_name
            if load_pattern_name != pt_cache:
                self.update_point_load(visibility=False)
                self.update_point_load_labels(visibility=False)
                self.update_distributed_loads(visibility=False)
                self.update_distributed_load_labels(visibility=False)
                self.update_rigid_deformed(visibility=False)
                self.update_deformed(visibility=False)
                self.update_axial_force(visibility=False)
                self.update_shear_force(visibility=False)
                self.update_bending_moment(visibility=False)
                self.update_reactions(visibility=False)
                self.update_displaced_nodes(visibility=False)
            elif load_pattern_name == pt_cache:
                if self.plotter_options.UI_load:
                    self.update_point_load(visibility=True)
                    self.update_point_load_labels(visibility=True)
                    self.update_distributed_loads(visibility=True)
                    self.update_distributed_load_labels(visibility=True)
                if self.plotter_options.UI_rigid_deformed:
                    self.update_rigid_deformed(visibility=True)
                if self.plotter_options.UI_deformed:
                    self.update_deformed(visibility=True)
                if self.plotter_options.UI_deformed or self.plotter_options.UI_rigid_deformed:
                    self.update_displaced_nodes(visibility=True)
                if self.plotter_options.UI_axial:
                    self.update_axial_force(visibility=True)
                if self.plotter_options.UI_shear:
                    self.update_shear_force(visibility=True)
                if self.plotter_options.UI_moment:
                    self.update_bending_moment(visibility=True)
                if self.plotter_options.UI_reactions:
                    self.update_reactions(visibility=True)
                if self.plotter_options.UI_deformed or self.plotter_options.UI_rigid_deformed:
                    self.update_displaced_nodes(visibility=True)
        self.figure.canvas.draw_idle()
        self.current_load_pattern = pt_cache

    def get_plotter_values(self, load_pattern_name: str) -> 'PlotterValues':
        # Comprobar si ya existe en caché
        if load_pattern_name in self.plotter_values:
            return self.plotter_values[load_pattern_name]

        # Verificar que el load pattern existe
        if load_pattern_name not in self.model.load_patterns:
            raise ValueError(
                f"El load pattern '{load_pattern_name}' no se encontró")

        # Verificar que existen resultados para este load pattern
        if load_pattern_name not in self.model.results:
            raise ValueError(
                f"Los resultados para el load pattern '{load_pattern_name}' no se encontraron")

        # Crear nueva instancia de PlotterValues
        plotter_values = PlotterValues(self.model, load_pattern_name)

        # Guardar en caché
        self.plotter_values[load_pattern_name] = plotter_values

        # Actualizar valores
        self.current_values = plotter_values

        return plotter_values

    def initialize_figure(self):
        # Cerrar figuras previas
        plt.close("all")

        # Configurar estilo global
        if self.plotter_options.plot_style in plt.style.available:
            plt.style.use(self.plotter_options.plot_style)

        # Crear figura y ejes
        self.figure = plt.figure(figsize=self.plotter_options.figure_size,
                                 dpi=self.plotter_options.dpi, facecolor=self.plotter_options.UI_background_color)
        self.axes = self.figure.add_subplot(111)

        # Configurar cuadrícula
        if self.plotter_options.grid:
            self.axes.grid(True, linestyle="--", alpha=0.5)

        # Ajustar layout
        if self.plotter_options.tight_layout:
            self.figure.tight_layout()

        # Mantener proporciones iguales
        plt.axis("equal")

        # Activar los ticks secundarios en ambos ejes
        # 5 subdivisiones entre cada tick principal
        self.axes.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.axes.yaxis.set_minor_locator(AutoMinorLocator(5))

        # Activar ticks en los 4 lados (mayores y menores)
        self.axes.tick_params(
            which="both", direction="in", length=6, width=1,
            top=True, bottom=True, left=True, right=True
        )
        # Ticks menores más pequeños y rojos
        self.axes.tick_params(which="minor", length=2,
                              width=0.5, color="black")

        # Mostrar etiquetas en los 4 lados
        self.axes.tick_params(labeltop=True, labelbottom=True,
                              labelleft=True, labelright=True)

        # Asegurar que los ticks se muestran en ambos lados
        self.axes.xaxis.set_ticks_position("both")
        self.axes.yaxis.set_ticks_position("both")

        # Personalizar el color de los ejes
        for spine in ["top", "bottom", "left", "right"]:
            self.axes.spines[spine].set_color("#9bc1bc")  # Color personalizado
            self.axes.spines[spine].set_linewidth(0.5)  # Grosor del borde

        # Personalizar las etiquetas de los ejes
        plt.xticks(fontsize=8, fontfamily="serif",
                   fontstyle="italic", color="#103b58")
        plt.yticks(fontsize=8, fontfamily="serif",
                   fontstyle="italic", color="#103b58")

        # Personalizar los ticks del eje X e Y
        self.axes.tick_params(axis="x", direction="in",
                              length=3.5, width=0.7, color="#21273a")
        self.axes.tick_params(axis="y", direction="in",
                              length=3.5, width=0.7, color="#21273a")

        # Cambiar el color de fondo del área de los ejes
        # self.axes.set_facecolor("#222222")  # Fondo oscuro dentro del Axes

        # Cambiar color del fondo exterior (Canvas)
        self.figure.patch.set_facecolor("#f5f5f5")  # Color gris oscuro

    def change_background_color(self):
        # self.figure.patch.set_facecolor(self.plotter_options.UI_background_color)
        self.axes.set_facecolor(self.plotter_options.UI_background_color)
        self.figure.canvas.draw()

    def plot_nodes(self):
        """
        Dibuja los nodos de la estructura.
        """
        # if not self.plotter_options.UI_show_nodes:
        #     return
        for node_id, coord in self.current_values.nodes.items():
            x = [coord[0]]
            y = [coord[1]]

            node = self.axes.scatter(
                x, y, c=self.plotter_options.node_color, s=self.plotter_options.node_size, marker='o')
            self.nodes[node_id] = node
            node.set_visible(self.plotter_options.UI_show_nodes)
        self.figure.canvas.draw_idle()

    def update_nodes(self):
        for node in self.nodes.values():
            node.set_visible(self.plotter_options.UI_show_nodes)
        self.figure.canvas.draw_idle()

    def plot_members(self):
        """
        Dibuja los elementos de la estructura.
        """
        # if not self.plotter_options.UI_show_members:
        #     return
        for id, coord in self.current_values.members.items():
            x_coords = [coord[0][0], coord[1][0]]
            y_coords = [coord[0][1], coord[1][1]]
            line, = self.axes.plot(x_coords, y_coords, color=self.plotter_options.element_color,
                                   linewidth=self.plotter_options.element_line_width)
            self.members[id] = line
            line.set_visible(self.plotter_options.UI_show_members)
        self.figure.canvas.draw_idle()

    def update_members(self):
        for member in self.members.values():
            member.set_visible(self.plotter_options.UI_show_members)
        self.figure.canvas.draw_idle()

    def plot_supports(self):
        """
        Dibuja los apoyos de la estructura.
        """
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

        for id, restrains in self.current_values.restraints.items():
            node_coords = self.current_values.nodes[id]
            support_func = support_functions.get(restrains)

            if support_func:
                sup_lines = support_func(
                    self.axes,
                    node_coords[0],
                    node_coords[1],
                    self.plotter_options.support_size,
                    self.plotter_options.support_color
                )
                self.supports[id] = sup_lines
        self.figure.canvas.draw_idle()

    def update_supports(self):
        for support in self.supports.values():
            for line in support:
                line.set_visible(self.plotter_options.show_supports)
        self.figure.canvas.draw_idle()

    def plot_node_labels(self):
        for id, coord in self.current_values.nodes.items():
            bbox = {
                "boxstyle": "circle",
                "facecolor": "lightblue",     # Color de fondo
                "edgecolor": "black",         # Color del borde
                "linewidth": 0.5,               # Grosor del borde
                "linestyle": "-",            # Estilo del borde
                "alpha": 0.8                  # Transparencia
            }
            text = self.axes.text(coord[0], coord[1], str(id),
                                  fontsize=self.plotter_options.label_font_size,
                                  ha='left', va='bottom', color="blue", bbox=bbox,
                                  clip_on=True)
            self.node_labels[id] = text
            text.set_visible(self.plotter_options.UI_node_labels)
            self.figure.canvas.draw_idle()

    def update_node_labels(self):
        for text in self.node_labels.values():
            text.set_visible(self.plotter_options.UI_node_labels)
        self.figure.canvas.draw_idle()

    def plot_member_labels(self):
        for element_id, coords in self.current_values.members.items():
            x_val = (coords[0][0] + coords[1][0]) / 2
            y_val = (coords[0][1] + coords[1][1]) / 2

            bbox = {
                "boxstyle": "round,pad=0.2",  # Estilo y padding del cuadro
                "facecolor": "lightblue",     # Color de fondo
                "edgecolor": "black",         # Color del borde
                "linewidth": 0.5,               # Grosor del borde
                "linestyle": "-",            # Estilo del borde
                "alpha": 0.8                  # Transparencia
            }
            text = self.axes.text(x_val, y_val, str(element_id),
                                  fontsize=self.plotter_options.label_font_size,
                                  ha='center', va='center', color="blue", bbox=bbox,
                                  clip_on=True)
            self.member_labels[element_id] = text
            text.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def update_member_labels(self):
        for text in self.member_labels.values():
            text.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def plot_point_loads(self) -> None:
        """
        Grafica las cargas puntuales.
        """
        # if not self.plotter_options.UI_point_load:
        #     return
        self.point_loads[self.current_load_pattern] = {}
        self.point_load_labels[self.current_load_pattern] = {}
        for id_node, load in self.current_values.point_loads.items():
            coords = self.current_values.nodes[id_node]

            arrows = []
            texts = []

            # Fuerza en dirección X
            if load["fx"] != 0:
                arrow, text = graphic_one_arrow(
                    x=coords[0],
                    y=coords[1],
                    load=load["fx"],
                    length_arrow=self.plotter_options.point_load_length_arrow,
                    angle=0 if load["fx"] < 0 else np.pi,
                    ax=self.axes,
                    color=self.plotter_options.point_load_color,
                    label=self.plotter_options.point_load_label,
                    color_label=self.plotter_options.point_load_label_color,
                    label_font_size=self.plotter_options.point_load_label_font_size
                )
                arrows.append(arrow)
                texts.append(text)

            # Fuerza en dirección Y
            if load["fy"] != 0:
                arrow, text = graphic_one_arrow(
                    x=coords[0],
                    y=coords[1],
                    load=load["fy"],
                    length_arrow=self.plotter_options.point_load_length_arrow,
                    angle=np.pi/2 if load["fy"] < 0 else 3*np.pi/2,
                    ax=self.axes,
                    color=self.plotter_options.point_load_color,
                    label=self.plotter_options.point_load_label,
                    color_label=self.plotter_options.point_load_label_color,
                    label_font_size=self.plotter_options.point_load_label_font_size
                )
                arrows.append(arrow)
                texts.append(text)

            # Momento en Z
            if load["mz"] != 0:
                arrow, text = moment_fancy_arrow(
                    ax=self.axes,
                    x=coords[0],
                    y=coords[1],
                    moment=load["mz"],
                    radio=0.70 * self.plotter_options.point_moment_length_arrow,
                    color=self.plotter_options.point_load_color,
                    clockwise=True,
                    label=self.plotter_options.point_load_label,
                    color_label=self.plotter_options.point_load_label_color,
                    label_font_size=self.plotter_options.point_load_label_font_size
                )
                arrows.append(arrow)
                texts.append(text)

            self.point_loads[self.current_load_pattern][id_node] = arrows
            self.point_load_labels[self.current_load_pattern][id_node] = texts
        # for arrow, text in zip(arrows, texts):
        #     arrow.set_visible(self.plotter_options.point_load)
        #     text.set_visible(self.plotter_options.point_load_label)
        self.figure.canvas.draw_idle()

    def update_point_load(self, visibility: bool | None = None):
        visibility = self.plotter_options.UI_load if visibility is None else visibility
        for arrows in self.point_loads[self.current_load_pattern].values():
            for arrow in arrows:
                arrow.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def update_point_load_labels(self, visibility: bool | None = None):
        visibility = self.plotter_options.UI_load if visibility is None else visibility
        for texts in self.point_load_labels[self.current_load_pattern].values():
            for text in texts:
                text.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def plot_distributed_loads(self) -> None:

        self.distributed_loads[self.current_load_pattern] = {}
        self.distributed_load_labels[self.current_load_pattern] = {}
        for id_element, load in self.current_values.distributed_loads.items():

            arrowslist = []
            textslist = []

            coords = self.current_values.members[id_element]

            # Calcular longitud y ángulo de rotación del elemento
            element = self.model.members[id_element]
            length = element.length()
            angle_rotation = element.angle_x()

            # Cargas verticales
            if load["q_i"] != 0 or load["q_j"] != 0:
                arrows, texts = graphic_n_arrow(
                    x=coords[0][0],
                    y=coords[0][1],
                    load_i=-load["q_i"],
                    load_j=-load["q_j"],
                    angle=np.pi/2,
                    length=length,
                    ax=self.axes,
                    ratio_scale=self.plotter_options.scale_dist_load[self.current_load_pattern],
                    nrof_arrows=self.plotter_options.nro_arrows(id_element),
                    color=self.plotter_options.distributed_load_color,
                    angle_rotation=angle_rotation,
                    label=self.plotter_options.distributed_load_label,
                    color_label=self.plotter_options.distributed_load_label_color,
                    label_font_size=self.plotter_options.distributed_load_label_font_size
                )
                arrowslist = arrowslist + arrows
                textslist = textslist + texts

            # Cargas axiales
            if load["p_i"] != 0 or load["p_j"] != 0:
                arrows, texts = graphic_n_arrow(
                    x=coords[0][0],
                    y=coords[0][1],
                    load_i=-load["p_i"],
                    load_j=-load["p_j"],
                    angle=0,
                    length=length,
                    ax=self.axes,
                    ratio_scale=self.plotter_options.scale_dist_load[self.current_load_pattern],
                    nrof_arrows=self.plotter_options.nro_arrows(id_element),
                    color=self.plotter_options.distributed_load_color,
                    angle_rotation=angle_rotation,
                    label=self.plotter_options.distributed_load_label,
                    color_label=self.plotter_options.distributed_load_label_color,
                    label_font_size=self.plotter_options.distributed_load_label_font_size
                )
                arrowslist = arrowslist + arrows
                textslist = textslist + texts
            # Momentos distribuidos (no implementados)
            if load["m_i"] != 0 or load["m_j"] != 0:
                raise NotImplementedError(
                    "Momentos distribuidos no implementados.")

            self.distributed_loads[self.current_load_pattern][id_element] = arrowslist
            self.distributed_load_labels[self.current_load_pattern][id_element] = textslist
        self.figure.canvas.draw_idle()

    def update_distributed_loads(self, visibility: Optional[bool] = None):
        visibility = bool(
            self.plotter_options.UI_load) if visibility is None else visibility
        for arrows in self.distributed_loads[self.current_load_pattern].values():
            for arrow in arrows:
                arrow.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def update_distributed_load_labels(self, visibility: Optional[bool] = None):
        visibility = self.plotter_options.UI_load if visibility is None else visibility
        for texts in self.distributed_load_labels[self.current_load_pattern].values():
            for text in texts:
                text.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def plot_rigid_deformed(self, escala: float | None = None):
        self.rigid_deformed_shape[self.current_load_pattern] = {}
        escala = self.plotter_options.UI_deformation_scale[
            self.current_load_pattern] if escala is None else escala
        for member_id in self.model.members.keys():
            x, y = self.current_values.rigid_deformed(member_id, escala)
            line, = self.axes.plot(
                x, y, color=self.plotter_options.rigid_deformed_color, lw=0.7, ls='--', zorder=1)
            self.rigid_deformed_shape[self.current_load_pattern][member_id] = line
            line.set_visible(self.plotter_options.UI_rigid_deformed)
        self.figure.canvas.draw_idle()

    def update_rigid_deformed(self, visibility: Optional[bool] = None, escala: float | None = None) -> None:
        visibility = self.plotter_options.UI_rigid_deformed if visibility is None else visibility
        for line in self.rigid_deformed_shape[self.current_load_pattern].values():
            line.set_visible(visibility)

        # HACER UN SET_DATA(X, Y) A TODOS LOS MIEMBROS DEL ACTUAL LOAD_PATTERN
        if escala is not None:
            for member in self.model.members.values():
                self.current_values = self.get_plotter_values(self.current_load_pattern)
                x, y = self.current_values.rigid_deformed(member.id, escala)
                self.rigid_deformed_shape[self.current_load_pattern][member.id].set_data(x, y)

        self.figure.canvas.draw_idle()

    def plot_deformed(self, escala: float | None = None) -> None:
        self.deformed_shape[self.current_load_pattern] = {}
        escala = self.plotter_options.UI_deformation_scale[
            self.current_load_pattern] if escala is None else escala
        for element in self.model.members.values():
            x, y = self.current_values.get_deformed_shape(element.id, escala)
            line, = self.axes.plot(x, y, lw=self.plotter_options.deformation_line_width,
                                   color=self.plotter_options.deformation_color)
            self.deformed_shape[self.current_load_pattern][element.id] = line
            line.set_visible(self.plotter_options.UI_deformed)
        self.figure.canvas.draw_idle()

    def update_deformed(self, visibility: Optional[bool] = None, escala: float | None = None):
        visibility = self.plotter_options.UI_deformed if visibility is None else visibility
        for line in self.deformed_shape[self.current_load_pattern].values():
            line.set_visible(visibility)

        # HACER UN SET_DATA(X, Y) A TODOS LOS MIEMBROS DEL ACTUAL LOAD_PATTERN
        if escala is not None:
            for member in self.model.members.values():
                self.current_values = self.get_plotter_values(self.current_load_pattern)
                x, y = self.current_values.get_deformed_shape(member.id, escala)
                self.deformed_shape[self.current_load_pattern][member.id].set_data(x, y)

        self.figure.canvas.draw_idle()

    def plot_internal_forces(self, type: InternalForceType, escala: float | None = None) -> None:

        def calculate_x_intersection(x1, y1, x2, y2):
            """Calcula la intersección con el eje X entre dos puntos"""
            if y1 == y2:
                return x1
            return x1 - y1 * (x2 - x1) / (y2 - y1)

        def separate_areas(array, L):
            """Separa las áreas positivas y negativas con puntos de intersección"""
            x_val = np.linspace(0, L, len(array))
            positive_areas = []
            negative_areas = []
            current_pos = []
            current_neg = []
            prev_sign = None

            for i, (x, y) in enumerate(zip(x_val, array)):
                current_sign = 'pos' if y >= 0 else 'neg'

                if i == 0:
                    if current_sign == 'pos':
                        current_pos.append([x, y])
                    else:
                        current_neg.append([x, y])
                    prev_sign = current_sign
                    continue

                if current_sign != prev_sign:
                    # Calcular punto de intersección
                    x_prev = x_val[i-1]
                    y_prev = array[i-1]
                    x_intersect = calculate_x_intersection(
                        x_prev, y_prev, x, y)

                    # Cerrar el área anterior
                    if prev_sign == 'pos':
                        current_pos.append([x_intersect, 0])
                        positive_areas.append(current_pos)
                        current_pos = []
                    else:
                        current_neg.append([x_intersect, 0])
                        negative_areas.append(current_neg)
                        current_neg = []

                    # Iniciar nueva área
                    if current_sign == 'pos':
                        current_pos = [[x_intersect, 0], [x, y]]
                    else:
                        current_neg = [[x_intersect, 0], [x, y]]
                    prev_sign = current_sign
                else:
                    if current_sign == 'pos':
                        current_pos.append([x, y])
                    else:
                        current_neg.append([x, y])
                    prev_sign = current_sign

            # Añadir áreas restantes
            if current_pos:
                positive_areas.append(current_pos)
            if current_neg:
                negative_areas.append(current_neg)

            return positive_areas, negative_areas

        def process_segments(segments, L):
            """Añade puntos en el eje X para segmentos en los bordes del dominio"""
            processed = []
            for seg in segments:
                new_seg = []
                # Verificar inicio
                if seg and seg[0][0] == 0 and seg[0][1] != 0:
                    new_seg.append([0.0, 0.0])

                new_seg.extend(seg)

                # Verificar final
                if seg and seg[-1][0] == L and seg[-1][1] != 0:
                    new_seg.append([L, 0.0])

                processed.append(new_seg)
            return processed

        # ESCALAS:
        if type == InternalForceType.AXIAL_FORCE:
            escala = self.plotter_options.axial_scale[self.current_load_pattern] if escala is None else escala
            self.axial_force[self.current_load_pattern] = {}
        elif type == InternalForceType.SHEAR_FORCE:
            escala = self.plotter_options.shear_scale[self.current_load_pattern] if escala is None else escala
            self.shear_force[self.current_load_pattern] = {}
        elif type == InternalForceType.BENDING_MOMENT:
            escala = self.plotter_options.moment_scale[self.current_load_pattern] if escala is None else escala
            self.bending_moment[self.current_load_pattern] = {}

        artist = []
        for member_id, member in self.model.members.items():

            # Obtener valores del diagrama
            if type == InternalForceType.AXIAL_FORCE:
                y_val = self.model.results[self.current_load_pattern].members[member_id]["axial_forces"] * escala
            elif type == InternalForceType.SHEAR_FORCE:
                y_val = self.model.results[self.current_load_pattern].members[member_id]["shear_forces"] * escala
            elif type == InternalForceType.BENDING_MOMENT:
                y_val = self.model.results[self.current_load_pattern].members[member_id]["bending_moments"] * escala
            x_val = np.linspace(0, member.length(), len(y_val))


            # Configuración inicial
            L = member.length()
            x = x_val
            y = y_val

            # Separar y procesar áreas
            positive, negative = separate_areas(y, L)
            positive_processed = process_segments(positive, L)
            negative_processed = process_segments(negative, L)

            # areglar el borde
            val = np.stack((x, y), axis=-1)
            if val[0][0] == 0 and val[0][1] != 0:
                val = np.insert(val, 0, [[0.0, 0.0]], axis=0)
            if val[-1][0] == L and val[-1][1] != 0:
                val = np.append(val, [[L, 0.0]], axis=0)

            # Transformar coordenadas y PLOTEAR
            for area in positive_processed:
                area = rotate_xy(area, member.angle_x(), 0, 0)
                area = traslate_xy(area, *member.node_i.vertex.coordinates)
                if len(area) > 2:
                    polygon, = self.axes.fill(*zip(*area), color='#807fff', alpha=0.7)
                    artist.append(polygon)
            for area in negative_processed:
                area = rotate_xy(area, member.angle_x(), 0, 0)
                area = traslate_xy(area, *member.node_i.vertex.coordinates)
                if len(area) > 2:
                    polygon, = self.axes.fill(*zip(*area), color='#ff897b', alpha=0.7)
                    artist.append(polygon)

            val = rotate_xy(val, member.angle_x(), 0, 0)
            val = traslate_xy(val, *member.node_i.vertex.coordinates)

            line, = self.axes.plot(*zip(*val), color='#424242', lw=0.5)  # Línea de la curva
            artist.append(line)

            if type == InternalForceType.AXIAL_FORCE:
                self.axial_force[self.current_load_pattern][member_id] = artist
            elif type == InternalForceType.SHEAR_FORCE:
                self.shear_force[self.current_load_pattern][member_id] = artist
            elif type == InternalForceType.BENDING_MOMENT:
                self.bending_moment[self.current_load_pattern][member_id] = artist
        # visibility
        if type == InternalForceType.AXIAL_FORCE:
            visibility = self.plotter_options.UI_axial
        elif type == InternalForceType.SHEAR_FORCE:
            visibility = self.plotter_options.UI_shear
        elif type == InternalForceType.BENDING_MOMENT:
            visibility = self.plotter_options.UI_moment
        for artist in artist:
            artist.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def update_internal_forces(self, type: InternalForceType, visibility: Optional[bool] = None) -> None:
        if type == InternalForceType.AXIAL_FORCE:
            visibility = self.plotter_options.UI_axial if visibility is None else visibility
            for listArtist in self.axial_force[self.current_load_pattern].values():
                for artist in listArtist:
                    artist.set_visible(visibility)
        elif type == InternalForceType.SHEAR_FORCE:
            visibility = self.plotter_options.UI_shear if visibility is None else visibility
            for listArtist in self.shear_force[self.current_load_pattern].values():
                for artist in listArtist:
                    artist.set_visible(visibility)
        elif type == InternalForceType.BENDING_MOMENT:
            visibility = self.plotter_options.UI_moment if visibility is None else visibility
            for listArtist in self.bending_moment[self.current_load_pattern].values():
                for artist in listArtist:
                    artist.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def plot_axial_force(self, escala: float | None = None) -> None:
        self.plot_internal_forces(InternalForceType.AXIAL_FORCE, escala)

    def update_axial_force(self, visibility: Optional[bool] = None) -> None:
        self.update_internal_forces(InternalForceType.AXIAL_FORCE, visibility)

    def plot_shear_force(self, escala: float | None = None) -> None:
        self.plot_internal_forces(InternalForceType.SHEAR_FORCE, escala)

    def update_shear_force(self, visibility: Optional[bool] = None) -> None:
        self.update_internal_forces(InternalForceType.SHEAR_FORCE, visibility)

    def plot_bending_moment(self, escala: float | None = None) -> None:
        self.plot_internal_forces(InternalForceType.BENDING_MOMENT, escala)

    def update_bending_moment(self, visibility: Optional[bool] = None) -> None:
        self.update_internal_forces(
            InternalForceType.BENDING_MOMENT, visibility)

    def plot_reactions(self) -> None:
        self.reactions[self.current_load_pattern] = {}
        for node in self.model.nodes.values():
            reactions = self.model.results[self.current_load_pattern].get_node_reactions(node.id)
            length_arrow = self.plotter_options.point_load_length_arrow
            moment_length_arrow = 0.70 * self.plotter_options.point_moment_length_arrow
            if reactions[0] != 0:
                arrowRX, textRX = graphic_one_arrow(
                    node.vertex.x, node.vertex.y, round(reactions[0], 2), length_arrow,
                    0 if reactions[0] < 0 else np.pi, self.axes,
                    self.plotter_options.reactions_color, True, "blue", 8)
            if reactions[1] != 0:
                arrowRY, textRY = graphic_one_arrow(
                    node.vertex.x, node.vertex.y, round(reactions[1], 2), length_arrow,
                    np.pi/2 if reactions[1] < 0 else 3*np.pi/2, self.axes,
                    self.plotter_options.reactions_color, True, "blue", 8)
            if reactions[2] != 0:
                arrowMZ, textMZ = moment_fancy_arrow(
                    self.axes, node.vertex.x, node.vertex.y, round(reactions[2], 2), moment_length_arrow,
                    self.plotter_options.reactions_color, True, True, "blue", 8)
                artists = [arrowRX, arrowRY, arrowMZ, textRX, textRY, textMZ]
                for artist in artists:
                    artist.set_visible(self.plotter_options.UI_reactions)
            self.reactions[self.current_load_pattern][node.id] = artists

        self.figure.canvas.draw_idle()

    def update_reactions(self, visibility: Optional[bool] = None) -> None:
        visibility = self.plotter_options.UI_reactions if visibility is None else visibility
        for listArtist in self.reactions[self.current_load_pattern].values():
            for artist in listArtist:
                artist.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def plot_displaced_nodes(self) -> None:
        self.deformed_nodes[self.current_load_pattern] = {}
        for node_id, coord in self.current_values.nodes.items():
            ux = self.model.results[self.current_load_pattern].get_node_displacements(node_id)[0]*self.plotter_options.UI_deformation_scale[self.current_load_pattern]
            vy = self.model.results[self.current_load_pattern].get_node_displacements(node_id)[1]*self.plotter_options.UI_deformation_scale[self.current_load_pattern]
            x = [coord[0] + ux]
            y = [coord[1] + vy]

            node = self.axes.scatter(
                x, y, c=self.plotter_options.node_color, s=self.plotter_options.node_size, marker='o')
            self.deformed_nodes[self.current_load_pattern][node_id] = node
            visibility = self.plotter_options.UI_deformed or self.plotter_options.UI_rigid_deformed
            node.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def update_displaced_nodes(self, visibility: Optional[bool] = None, scale: Optional[float] = None) -> None:
        vis = self.plotter_options.UI_deformed or self.plotter_options.UI_rigid_deformed
        visibility = vis if visibility is None else visibility
        for node in self.deformed_nodes[self.current_load_pattern].values():
            node.set_visible(visibility)
        self.figure.canvas.draw_idle()


        # HACER UN SET_DATA(X, Y) A TODOS LOS NODOS DEL ACTUAL LOAD_PATTERN
        if scale is not None:
            for node_id, node in self.deformed_nodes[self.current_load_pattern].items():
                ux = self.model.results[self.current_load_pattern].get_node_displacements(node_id)[0]*scale
                vy = self.model.results[self.current_load_pattern].get_node_displacements(node_id)[1]*scale
                x = self.model.nodes[node_id].vertex.x + ux
                y = self.model.nodes[node_id].vertex.y + vy
                node.set_offsets([x, y])
            self.figure.canvas.draw_idle()
