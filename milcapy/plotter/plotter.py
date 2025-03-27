from typing import TYPE_CHECKING, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


from milcapy.utils import rotate_xy, traslate_xy
from milcapy.plotter.suports import (
    support_ttt, support_ttf, support_tft,
    support_ftt, support_tff, support_ftf, support_fft
)
from milcapy.plotter.load import (
    graphic_one_arrow, moment_fancy_arrow
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
        options: Optional['PlotterOptions'] = None
    ) -> None:
        self.model = model
        self.plotter_values: Dict[str, 'PlotterValues'] = {}    # {load_pattern_name: PlotterValues}
        self.plotter_options = options if options else model.plotter_options
        self.plotter_options.load_mean("CARGA1")
        self.figure: Figure = None
        self.axes: Axes = None
        self.current_values: 'PlotterValues' | None = None
        self.initialize_figure() # Inicializar figura
        # contenedores de artists de matplotlib

        # todos los que tengan 🐍 se calculan para cada load pattern
        # todos los que tengan ✅ ya estan implementados

        # nodos
        self.nodes= {}              # ✅visibilidad, color
        # nodos deformados
        self.deformed_nodes = {}    # 🐍 interactividad al pasar el ratón

        # miembros
        self.members = {}           # ✅visibilidad, color
        # forma deformada
        self.deformed_shape = {}    # 🐍 visibilidad, setdata, setType: colorbar
        # cargas puntuales
        self.point_loads = {}       # 🐍 visibilidad
        # cargas distribuidas
        self.distributed_loads = {} # 🐍 visibilidad
        # fuerzas internas (line2D: borde)
        self.internal_forces = {}   # 🐍 visibilidad, setdata, setType: colorbar
        # fillings for internal forces
        self.fillings = {}          # 🐍 visibilidad, setdata, setType: colorbar
        # apooyos
        self.supports = {}          # ✅visibilidad, color, setdata
        # apoyos dezplados
        self.displaced_supports = {}# 🐍 visibilidad, color, setdata
        # etiquetas
        self.node_labels = {}            # ✅visibilidad, setdata
        self.member_labels = {}          # ✅visibilidad, setdata
        self.point_load_labels = {}      # 🐍 visibilidad, setdata
        self.distributed_load_labels = {}# 🐍 visibilidad, setdata
        self.internal_forces_labels = {} # 🐍 visibilidad, setdata
        self.reactions_labels = {}       # 🐍 visibilidad, setdata
        self.displacement_labels = {}    # 🐍 visibilidad, setdata
        # reacciones
        self.reactions = {}         # 🐍 visibilidad, setdata

    @property
    def current_load_pattern(self) -> Optional[str]:
        return self.model.current_load_pattern

    @current_load_pattern.setter
    def current_load_pattern(self, value: Optional[str]):
        self.model.current_load_pattern = value

    # ! EJECUTAR AL INICIO DE LA APLICACIÓN Y CADA VEZ QUE SE CAMBIA EL LOAD_PATTERN
    def set_load_pattern_name(self, load_pattern_name: str):
        """Sistemas de NOTIFICACION para establecer el load pattern actual"""
        self.current_load_pattern = load_pattern_name
        self.current_values = self.get_plotter_values(load_pattern_name)
        if self.nodes is {}:
            self.plot_nodes()
        if self.members is {}:
            self.plot_members()
        # if self.point_loads.get(self.current_load_pattern, None) is None or self.point_loads is {}:
            # self.plot_point_loads()
        if self.supports is {}:
            self.plot_supports()
        if self.node_labels is {}:
            self.plot_node_labels()
        if self.member_labels is {}:
            self.plot_member_labels()




    def get_plotter_values(self, load_pattern_name: str) -> 'PlotterValues':
        # Comprobar si ya existe en caché
        if load_pattern_name in self.plotter_values:
            return self.plotter_values[load_pattern_name]

        # Verificar que el load pattern existe
        if load_pattern_name not in self.model.load_patterns:
            raise ValueError(f"El load pattern '{load_pattern_name}' no se encontró")

        # Verificar que existen resultados para este load pattern
        if load_pattern_name not in self.model.results:
            raise ValueError(f"Los resultados para el load pattern '{load_pattern_name}' no se encontraron")

        # Obtener los resultados para este load pattern
        results = self.model.results[load_pattern_name]

        # Crear nueva instancia de PlotterValues
        plotter_values = PlotterValues(self.model, load_pattern_name, results)

        # Guardar en caché
        self.plotter_values[load_pattern_name] = plotter_values

        return plotter_values

    def initialize_figure(self):
        # Cerrar figuras previas
        plt.close("all")

        # Configurar estilo global
        if self.plotter_options.plot_style in plt.style.available:
            plt.style.use(self.plotter_options.plot_style)

        # Crear figura y ejes
        self.figure = plt.figure(figsize=self.plotter_options.figure_size, dpi=self.plotter_options.dpi, facecolor=self.plotter_options.UI_background_color)
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
        self.axes.xaxis.set_minor_locator(AutoMinorLocator(5))  # 5 subdivisiones entre cada tick principal
        self.axes.yaxis.set_minor_locator(AutoMinorLocator(5))

        # Activar ticks en los 4 lados (mayores y menores)
        self.axes.tick_params(
            which="both", direction="in", length=6, width=1,
            top=True, bottom=True, left=True, right=True
        )
        self.axes.tick_params(which="minor", length=2, width=0.5, color="black")  # Ticks menores más pequeños y rojos

        # Mostrar etiquetas en los 4 lados
        self.axes.tick_params(labeltop=True, labelbottom=True, labelleft=True, labelright=True)

        # Asegurar que los ticks se muestran en ambos lados
        self.axes.xaxis.set_ticks_position("both")
        self.axes.yaxis.set_ticks_position("both")

        # Personalizar el color de los ejes
        for spine in ["top", "bottom", "left", "right"]:
            self.axes.spines[spine].set_color("#9bc1bc")  # Color personalizado
            self.axes.spines[spine].set_linewidth(0.5)  # Grosor del borde

        # Personalizar las etiquetas de los ejes
        plt.xticks(fontsize=8, fontfamily="serif", fontstyle="italic", color="#103b58")
        plt.yticks(fontsize=8, fontfamily="serif", fontstyle="italic", color="#103b58")

        # Personalizar los ticks del eje X e Y
        self.axes.tick_params(axis="x", direction="in", length=3.5, width=0.7, color="#21273a")
        self.axes.tick_params(axis="y", direction="in", length=3.5, width=0.7, color="#21273a")

        # Cambiar el color de fondo del área de los ejes
        # self.axes.set_facecolor("#222222")  # Fondo oscuro dentro del Axes

        # Cambiar color del fondo exterior (Canvas)
        self.figure.patch.set_facecolor("#f5f5f5")  # Color gris oscuro

    def plot_nodes(self):
        """
        Dibuja los nodos de la estructura.
        """
        if not self.plotter_options.UI_show_nodes:
            return
        for id, coord in self.current_values.nodes.items():
            x = [coord[0]]
            y = [coord[1]]

            node = self.axes.scatter(x, y, c=self.plotter_options.node_color, s=self.plotter_options.node_size, marker='o')
            self.nodes[id] = node
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
        if not self.plotter_options.UI_show_members:
            return
        for id, coord in self.current_values.members.items():
            x_coords = [coord[0][0], coord[1][0]]
            y_coords = [coord[0][1], coord[1][1]]
            line, = self.axes.plot(x_coords, y_coords, color=self.plotter_options.element_color, linewidth=self.plotter_options.element_line_width)
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
        if not self.plotter_options.show_supports:
            return
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
                for line in sup_lines:
                    line.set_visible(self.plotter_options.show_supports)
        self.figure.canvas.draw_idle()

    def update_supports(self):
        for support in self.supports.values():
            for line in support:
                line.set_visible(self.plotter_options.show_supports)
        self.figure.canvas.draw_idle()

    def plot_node_labels(self):
        for id, coord in self.current_values.nodes.items():
            bbox={
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

            bbox={
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




    # def plot_point_loads(self) -> None:
    #     """
    #     Grafica las cargas puntuales.
    #     """
    #     if not self.plotter_options.UI_point_load:
    #         return
    #     self.point_loads[self.current_load_pattern] = {}
    #     self.point_load_labels[self.current_load_pattern] = {}
    #     for id_node, load in self.current_values.point_loads.items():
    #         coords = self.current_values.nodes[id_node]

    #         arrows = []
    #         texts = []

    #         # Fuerza en dirección X
    #         if load["fx"] != 0:
    #             arrow, text = graphic_one_arrow(
    #                 x=coords[0],
    #                 y=coords[1],
    #                 load=load["fx"],
    #                 length_arrow=self.plotter_options.point_load_length_arrow,
    #                 angle=0 if load["fx"] < 0 else np.pi,
    #                 ax=self.axes,
    #                 color=self.plotter_options.point_load_color,
    #                 label=self.plotter_options.UI_point_load_label,
    #                 color_label=self.plotter_options.point_load_label_color,
    #                 label_font_size=self.plotter_options.point_load_label_font_size
    #             )
    #             arrows.append(arrow)
    #             texts.append(text)

    #         # Fuerza en dirección Y
    #         if load["fy"] != 0:
    #             arrow, text = graphic_one_arrow(
    #                 x=coords[0],
    #                 y=coords[1],
    #                 load=load["fy"],
    #                 length_arrow=self.plotter_options.point_load_length_arrow,
    #                 angle=np.pi/2 if load["fy"] < 0 else 3*np.pi/2,
    #                 ax=self.axes,
    #                 color=self.plotter_options.point_load_color,
    #                 label=self.plotter_options.UI_point_load_label,
    #                 color_label=self.plotter_options.point_load_label_color,
    #                 label_font_size=self.plotter_options.point_load_label_font_size
    #             )
    #             arrows.append(arrow)
    #             texts.append(text)


    #         # Momento en Z
    #         if load["mz"] != 0:
    #             arrow, text = moment_fancy_arrow(
    #                 ax=self.axes,
    #                 x=coords[0],
    #                 y=coords[1],
    #                 moment=load["mz"],
    #                 radio= 0.70 * self.plotter_options.point_moment_length_arrow,
    #                 color=self.plotter_options.point_load_color,
    #                 clockwise=True,
    #                 label=self.plotter_options.UI_point_load_label,
    #                 color_label=self.plotter_options.point_load_label_color,
    #                 label_font_size=self.plotter_options.point_load_label_font_size
    #             )
    #             arrows.append(arrow)
    #             texts.append(text)


    #         self.point_loads[self.current_load_pattern][id_node] = arrows
    #         self.point_load_labels[self.current_load_pattern][id_node] = texts
    #     for arrow, text in zip(arrows, texts):
    #         arrow.set_visible(self.plotter_options.UI_point_load)
    #         text.set_visible(self.plotter_options.UI_point_load)
    #     self.figure.canvas.draw_idle()

    # def update_point_load(self):
    #     for arrows in self.point_loads[self.current_load_pattern].values():
    #         for arrow in arrows:
    #             arrow.set_visible(self.plotter_options.UI_point_load)
    #     self.figure.canvas.draw_idle()

    # def update_point_load_labels(self):
    #     for texts in self.point_load_labels[self.current_load_pattern].values():
    #         for text in texts:
    #             text.set_visible(self.plotter_options.UI_point_load)
    #     self.figure.canvas.draw_idle()





    # def plot_distributed_loads(self) -> None:
    #     for id_element, load in self.plotter_values.distributed_loads.items():
    #         coords = self.plotter_values.elements[id_element]

    #         # Calcular longitud y ángulo de rotación del elemento
    #         x_diff = coords[1][0] - coords[0][0]
    #         y_diff = coords[1][1] - coords[0][1]
    #         length = np.sqrt(x_diff**2 + y_diff**2)
    #         angle_rotation = np.arctan2(y_diff, x_diff)

    #         # Cargas verticales
    #         if load["q_i"] != 0 or load["q_j"] != 0:
    #             graphic_n_arrow(
    #                 x=coords[0][0],
    #                 y=coords[0][1],
    #                 load_i=load["q_i"],
    #                 load_j=load["q_j"],
    #                 angle=np.pi/2,
    #                 length=length,
    #                 ax=self.axes,
    #                 ratio_scale=self.options.ratio_scale_load,
    #                 nrof_arrows=self.options.nrof_arrows,
    #                 color=self.options.distributed_load_color,
    #                 angle_rotation=angle_rotation,
    #                 label=self.options.distributed_load_label,
    #                 color_label=self.options.distributed_load_label_color,
    #                 label_font_size=self.options.distributed_load_label_font_size
    #             )

    #         # Cargas axiales
    #         if load["p_i"] != 0 or load["p_j"] != 0:
    #             graphic_n_arrow(
    #                 x=coords[0][0],
    #                 y=coords[0][1],
    #                 load_i=load["p_i"],
    #                 load_j=load["p_j"],
    #                 angle=0,
    #                 length=length,
    #                 ax=self.axes,
    #                 ratio_scale=self.options.ratio_scale_load,
    #                 nrof_arrows=self.options.nrof_arrows,
    #                 color=self.options.distributed_load_color,
    #                 angle_rotation=angle_rotation,
    #                 label=self.options.distributed_load_label,
    #                 color_label=self.options.distributed_load_label_color,
    #                 label_font_size=self.options.distributed_load_label_font_size
    #             )

    #         # Momentos distribuidos (no implementados)
    #         if load["m_i"] != 0 or load["m_j"] != 0:
    #             raise NotImplementedError("Momentos distribuidos no implementados.")

    # def plot_axial_force(self):

    #     for element_id, element in self.model.element_map.items():

    #         # Obtener valores del diagrama
    #         x, n = self.plotter_values.axial_forces[element_id]

    #         n = n*self.options.internal_forces_scale
    #         # Obtener coordenadas del elemento
    #         coord_elem = np.array([
    #             np.array([element.node_i.vertex.coordinates]),
    #             np.array([element.node_j.vertex.coordinates])
    #         ])

    #         # Transformar coordenadas
    #         Nxy = np.column_stack((x, n))
    #         Nxy = rotate_xy(Nxy, element.angle_x, 0, 0)
    #         Nxy = traslate_xy(Nxy, *element.node_i.vertex.coordinates)
    #         Nxy = np.insert(Nxy, 0, coord_elem[0], axis=0)
    #         Nxy = np.append(Nxy, coord_elem[1], axis=0)

    #         # Graficar diagrama
    #         self.axes.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')

    #         # Rellenar diagrama si se solicita
    #         NNxy = np.append(Nxy, coord_elem[0], axis=0)
    #         self.axes.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)

    # def plot_shear_force(self):

    #     for element_id, element in self.model.element_map.items():

    #         # Obtener valores del diagrama
    #         x, n = self.plotter_values.shear_forces[element_id]

    #         n = n*self.options.internal_forces_scale
    #         # Obtener coordenadas del elemento
    #         coord_elem = np.array([
    #             np.array([element.node_i.vertex.coordinates]),
    #             np.array([element.node_j.vertex.coordinates])
    #         ])

    #         # Transformar coordenadas
    #         Nxy = np.column_stack((x, n))
    #         Nxy = rotate_xy(Nxy, element.angle_x, 0, 0)
    #         Nxy = traslate_xy(Nxy, *element.node_i.vertex.coordinates)
    #         Nxy = np.insert(Nxy, 0, coord_elem[0], axis=0)
    #         Nxy = np.append(Nxy, coord_elem[1], axis=0)

    #         # Graficar diagrama
    #         self.axes.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')

    #         # Rellenar diagrama si se solicita
    #         NNxy = np.append(Nxy, coord_elem[0], axis=0)
    #         self.axes.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)

    # def plot_bending_moment(self):

    #     for element_id, element in self.model.element_map.items():

    #         # Obtener valores del diagrama
    #         x, n = self.plotter_values.bending_moments[element_id]

    #         n = n*self.options.internal_forces_scale
    #         # Obtener coordenadas del elemento
    #         coord_elem = np.array([
    #             np.array([element.node_i.vertex.coordinates]),
    #             np.array([element.node_j.vertex.coordinates])
    #         ])

    #         # Transformar coordenadas
    #         Nxy = np.column_stack((x, n))
    #         Nxy = rotate_xy(Nxy, element.angle_x, 0, 0)
    #         Nxy = traslate_xy(Nxy, *element.node_i.vertex.coordinates)
    #         Nxy = np.insert(Nxy, 0, coord_elem[0], axis=0)
    #         Nxy = np.append(Nxy, coord_elem[1], axis=0)

    #         # Graficar diagrama
    #         self.axes.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')

    #         # Rellenar diagrama si se solicita
    #         NNxy = np.append(Nxy, coord_elem[0], axis=0)
    #         self.axes.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)

    # def plot_slope(self):

    #     for element_id, element in self.model.element_map.items():

    #         # Obtener valores del diagrama
    #         x, n = self.plotter_values.slopes[element_id]

    #         n = n*self.options.internal_forces_scale
    #         # Obtener coordenadas del elemento
    #         coord_elem = np.array([
    #             np.array([element.node_i.vertex.coordinates]),
    #             np.array([element.node_j.vertex.coordinates])
    #         ])

    #         # Transformar coordenadas
    #         Nxy = np.column_stack((x, n))
    #         Nxy = rotate_xy(Nxy, element.angle_x, 0, 0)
    #         Nxy = traslate_xy(Nxy, *element.node_i.vertex.coordinates)
    #         Nxy = np.insert(Nxy, 0, coord_elem[0], axis=0)
    #         Nxy = np.append(Nxy, coord_elem[1], axis=0)

    #         # Graficar diagrama
    #         self.axes.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')

    #         # Rellenar diagrama si se solicita
    #         NNxy = np.append(Nxy, coord_elem[0], axis=0)
    #         self.axes.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)

    # def plot_deflection(self):

    #     for element_id, element in self.model.element_map.items():

    #         # Obtener valores del diagrama
    #         x, n = self.plotter_values.deflections[element_id]

    #         n = n*self.options.internal_forces_scale
    #         # Obtener coordenadas del elemento
    #         coord_elem = np.array([
    #             np.array([element.node_i.vertex.coordinates]),
    #             np.array([element.node_j.vertex.coordinates])
    #         ])

    #         # Transformar coordenadas
    #         Nxy = np.column_stack((x, n))
    #         Nxy = rotate_xy(Nxy, element.angle_x, 0, 0)
    #         Nxy = traslate_xy(Nxy, *element.node_i.vertex.coordinates)
    #         Nxy = np.insert(Nxy, 0, coord_elem[0], axis=0)
    #         Nxy = np.append(Nxy, coord_elem[1], axis=0)

    #         # Graficar diagrama
    #         self.axes.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')

    #         # Rellenar diagrama si se solicita
    #         NNxy = np.append(Nxy, coord_elem[0], axis=0)
    #         self.axes.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)

    # def plot_deformed(self) -> None:
    #     if self.options.show_undeformed:
    #         self.plot_elements()
    #         self.plot_supports()

    #     escala = self.options.deformation_scale
    #     for element in self.model.element_map.values():
    #         x, y = deformed(element, escala)
    #         self.axes.plot(x, y, lw=self.options.deformation_line_width,
    #                     color=self.options.deformation_color)

    # def plot_rigid_deformed(self) -> None:
    #     escala = self.options.deformation_scale
    #     for element in self.model.element_map.values():
    #         x, y = rigid_deformed(element, escala)
    #         self.axes.plot(
    #             x,
    #             y,
    #             lw=1,
    #             color='#54becb',
    #             linestyle="--"
    #         )

    # def plot_structure(self) -> None:
    #     self.plot_elements()
    #     self.plot_supports()
    #     self.plot_point_loads()
    #     self.plot_distributed_loads()

    # def show(self):
    #     """
    #     Muestra la gráfica.
    #     """
    #     root = create_plot_window(self.figure)
    #     root.mainloop()
