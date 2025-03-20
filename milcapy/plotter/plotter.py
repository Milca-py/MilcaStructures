from typing import TYPE_CHECKING, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


from milcapy.utils import rotate_xy, traslate_xy
from milcapy.postprocess.post_processing import deformed, rigid_deformed
from milcapy.plotter.suports import (
    support_ttt, support_ttf, support_tft, 
    support_ftt, support_tff, support_ftf, support_fft
)
from milcapy.plotter.load import (
    graphic_n_arrow, graphic_one_arrow, moment_fancy_arrow
)
from milcapy.plotter.options import PlotterOptions
from milcapy.frontend.widgets.UIdisplay import create_plot_window

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
        self.factory_values = model.plotter_values_factory
        self.options = options if options else model.plotter_options

        # # Atributos para figuras y ejes
        # self.figs: List['Figure'] = []  # Figure
        # self.axes: Dict[int, List['Axes']]    # {fig_number: List[Axes]}

        self.figure: Figure = None
        self.axes: Axes = None
        
        # Inicializar figura
        self.initialize_figure()
        
        # ESTE VALOR SE CAMBIA DINAMICAMENTE PARA EL PLOTEO DE CADA LOAD_PATTERN
        self.load_pattern_name = None
        self.plotter_values = None
        

    def set_load_pattern_name(self, load_pattern_name: str):# hacer esto antes de cada ploteo
        self.load_pattern_name = load_pattern_name
        self.plotter_values = self.factory_values.get_plotter_values(self.load_pattern_name)



    def initialize_figure(self):
        # Cerrar figuras previas
        plt.close("all")

        # Configurar estilo global
        if self.options.plot_style in plt.style.available:
            plt.style.use(self.options.plot_style)

        # Crear figura y ejes
        self.figure = plt.figure(figsize=self.options.figure_size, dpi=self.options.dpi, facecolor=self.options.background_color)
        self.axes = self.figure.add_subplot(111)

        # Configurar cuadrícula
        if self.options.grid:
            self.axes.grid(True, linestyle="--", alpha=0.5)

        # Ajustar layout
        if self.options.tight_layout:
            self.figure.tight_layout()

        # Mantener proporciones iguales
        plt.axis("equal")

        # 📌 Activar los ticks secundarios en ambos ejes
        self.axes.xaxis.set_minor_locator(AutoMinorLocator(5))  # 5 subdivisiones entre cada tick principal
        self.axes.yaxis.set_minor_locator(AutoMinorLocator(5))

        # 📌 Activar ticks en los 4 lados (mayores y menores)
        self.axes.tick_params(
            which="both", direction="in", length=6, width=1,
            top=True, bottom=True, left=True, right=True
        )
        self.axes.tick_params(which="minor", length=2, width=0.5, color="black")  # Ticks menores más pequeños y rojos

        # 📌 Mostrar etiquetas en los 4 lados
        self.axes.tick_params(labeltop=True, labelbottom=True, labelleft=True, labelright=True)
        
        # 📌 Asegurar que los ticks se muestran en ambos lados
        self.axes.xaxis.set_ticks_position("both")
        self.axes.yaxis.set_ticks_position("both")

        # 📌 Personalizar el color de los ejes
        for spine in ["top", "bottom", "left", "right"]:
            self.axes.spines[spine].set_color("#9bc1bc")  # Color personalizado
            self.axes.spines[spine].set_linewidth(0.5)  # Grosor del borde

        # 📌 Personalizar las etiquetas de los ejes
        plt.xticks(fontsize=8, fontfamily="serif", fontstyle="italic", color="#103b58")  
        plt.yticks(fontsize=8, fontfamily="serif", fontstyle="italic", color="#103b58")

        # 📌 Personalizar los ticks del eje X e Y
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
        for coord in self.plotter_values.nodes.values():
            x = [coord[0]]
            y = [coord[1]]
            
            self.axes.scatter(x, y, c=self.options.node_color, s=self.options.node_size, marker='o')

    def plot_elements(self):
        """
        Dibuja los elementos de la estructura.
        """
        for coord in self.plotter_values.elements.values():
            x_coords = [coord[0][0], coord[1][0]]
            y_coords = [coord[0][1], coord[1][1]]
            self.axes.plot(x_coords, y_coords, color=self.options.element_color, 
                        linewidth=self.options.element_line_width)

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
        
        for node_id, restrains in self.plotter_values.restraints.items():
            node_coords = self.plotter_values.nodes[node_id]
            support_func = support_functions.get(restrains)
            
            if support_func:
                support_func(
                    self.axes,
                    node_coords[0],
                    node_coords[1],
                    self.options.support_size,
                    self.options.support_color
                )
            elif restrains != (False, False, False):
                raise ValueError("Restricciones no válidas, no se puede plotear el apoyo.")

    def plot_node_labels(self):
        for node_id, coords in self.plotter_values.nodes.items():
            bbox={
                "boxstyle": "circle", 
                "facecolor": "lightblue",     # Color de fondo
                "edgecolor": "black",         # Color del borde
                "linewidth": 0.5,               # Grosor del borde
                "linestyle": "-",            # Estilo del borde
                "alpha": 0.8                  # Transparencia
            }
            self.axes.text(coords[0], coords[1], str(node_id), 
                        fontsize=self.options.label_font_size,
                        ha='left', va='bottom', color="blue", bbox=bbox,
                        clip_on=True)

    def plot_element_labels(self):
        for element_id, coords in self.plotter_values.elements.items():
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
            self.axes.text(x_val, y_val, str(element_id), 
                        fontsize=self.options.label_font_size,
                        ha='center', va='center', color="blue", bbox=bbox,
                        clip_on=True)

    def plot_point_loads(self) -> None:
        """
        Grafica las cargas puntuales.
        """
        for id_node, load in self.plotter_values.point_loads.items():
            coords = self.plotter_values.nodes[id_node]
            
            # Fuerza en dirección X
            if load["fx"] != 0:
                graphic_one_arrow(
                    x=coords[0],
                    y=coords[1],
                    load=load["fx"],
                    length_arrow=self.options.point_load_length_arrow,
                    angle=0,
                    ax=self.axes,
                    color=self.options.point_load_color,
                    label=self.options.point_load_label,
                    color_label=self.options.point_load_label_color,
                    label_font_size=self.options.point_load_label_font_size
                )
            
            # Fuerza en dirección Y
            if load["fy"] != 0:
                graphic_one_arrow(
                    x=coords[0],
                    y=coords[1],
                    load=load["fy"],
                    length_arrow=self.options.point_load_length_arrow,
                    angle=np.pi/2,
                    ax=self.axes,
                    color=self.options.point_load_color,
                    label=self.options.point_load_label,
                    color_label=self.options.point_load_label_color,
                    label_font_size=self.options.point_load_label_font_size
                )
            
            # Momento en Z
            if load["mz"] != 0:
                moment_fancy_arrow(
                    ax=self.axes,
                    x=coords[0],
                    y=coords[1],
                    moment=load["mz"],
                    radio= self.options.point_moment_length_arrow,
                    color=self.options.point_load_color,
                    clockwise=True,
                    label=self.options.point_load_label,
                    color_label=self.options.point_load_label_color,
                    label_font_size=self.options.point_load_label_font_size
                )

    def plot_distributed_loads(self) -> None:
        for id_element, load in self.plotter_values.distributed_loads.items():
            coords = self.plotter_values.elements[id_element]
            
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
                    ax=self.axes,
                    ratio_scale=self.options.ratio_scale_load,
                    nrof_arrows=self.options.nrof_arrows,
                    color=self.options.distributed_load_color,
                    angle_rotation=angle_rotation,
                    label=self.options.distributed_load_label,
                    color_label=self.options.distributed_load_label_color,
                    label_font_size=self.options.distributed_load_label_font_size
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
                    ax=self.axes,
                    ratio_scale=self.options.ratio_scale_load,
                    nrof_arrows=self.options.nrof_arrows,
                    color=self.options.distributed_load_color,
                    angle_rotation=angle_rotation,
                    label=self.options.distributed_load_label,
                    color_label=self.options.distributed_load_label_color,
                    label_font_size=self.options.distributed_load_label_font_size
                )
            
            # Momentos distribuidos (no implementados)
            if load["m_i"] != 0 or load["m_j"] != 0:
                raise NotImplementedError("Momentos distribuidos no implementados.")

    def plot_axial_force(self):
        
        for element_id, element in self.model.element_map.items():
            
            # Obtener valores del diagrama
            x, n = self.plotter_values.axial_forces[element_id]

            n = n*self.options.internal_forces_scale
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
            self.axes.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')
            
            # Rellenar diagrama si se solicita
            NNxy = np.append(Nxy, coord_elem[0], axis=0)
            self.axes.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)
    
    def plot_shear_force(self):
        
        for element_id, element in self.model.element_map.items():
            
            # Obtener valores del diagrama
            x, n = self.plotter_values.shear_forces[element_id]

            n = n*self.options.internal_forces_scale
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
            self.axes.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')
            
            # Rellenar diagrama si se solicita
            NNxy = np.append(Nxy, coord_elem[0], axis=0)
            self.axes.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)
            
    def plot_bending_moment(self):
        
        for element_id, element in self.model.element_map.items():
            
            # Obtener valores del diagrama
            x, n = self.plotter_values.bending_moments[element_id]

            n = n*self.options.internal_forces_scale
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
            self.axes.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')
            
            # Rellenar diagrama si se solicita
            NNxy = np.append(Nxy, coord_elem[0], axis=0)
            self.axes.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)

    def plot_slope(self):
        
        for element_id, element in self.model.element_map.items():
            
            # Obtener valores del diagrama
            x, n = self.plotter_values.slopes[element_id]

            n = n*self.options.internal_forces_scale
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
            self.axes.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')
            
            # Rellenar diagrama si se solicita
            NNxy = np.append(Nxy, coord_elem[0], axis=0)
            self.axes.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)
    
    def plot_deflection(self):
        
        for element_id, element in self.model.element_map.items():
            
            # Obtener valores del diagrama
            x, n = self.plotter_values.deflections[element_id]

            n = n*self.options.internal_forces_scale
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
            self.axes.plot(Nxy[:, 0], Nxy[:, 1], lw=0.5, color='orange')
            
            # Rellenar diagrama si se solicita
            NNxy = np.append(Nxy, coord_elem[0], axis=0)
            self.axes.fill(NNxy[:, 0], NNxy[:, 1], color='skyblue', alpha=0.5)
    
    
    
    
    
############################################################################################
    def plot_deformed(self) -> None:
        if self.options.show_undeformed:
            self.plot_elements()
            self.plot_supports()
        
        escala = self.options.deformation_scale
        for element in self.model.element_map.values():
            x, y = deformed(element, escala)
            self.axes.plot(x, y, lw=self.options.deformation_line_width,
                        color=self.options.deformation_color)

    def plot_rigid_deformed(self) -> None:
        escala = self.options.deformation_scale
        for element in self.model.element_map.values():
            x, y = rigid_deformed(element, escala)
            self.axes.plot(
                x, 
                y, 
                lw=1, 
                color='#54becb',
                linestyle="--"
            )


    def plot_structure(self) -> None:
        self.plot_elements()
        self.plot_supports()
        self.plot_point_loads()
        self.plot_distributed_loads()
        


    def show(self):
        """
        Muestra la gráfica.
        """
        root = create_plot_window(self.figure)
        root.mainloop()
