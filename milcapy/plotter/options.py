from typing import TYPE_CHECKING

from numpy import True_, mean
if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
class PlotterOptions:        # ✅✅✅
    """Clase que define las opciones de visualización"""

    def __init__(self, model: 'SystemMilcaModel'):
        """
        Inicializa las opciones de visualización con valores predeterminados.
        """
        self.model = model
        # self.values_calculator: GraphicOptionCalculator = graphic_option_calculator
        # Opciones
        # GENERALES
        self.figure_size = (10, 8)      ##
        self.dpi = 100                  ### ✅✅✅
        self.save_fig_dpi = 400         ### ✅✅✅
        self.UI_background_color = 'white' ### ✅✅✅
        self.grid = False                ##
        # 'default', 'dark_background', 'ggplot', etc.
        self.plot_style = 'default'     ##

        # Opciones para visualización de estructura
        # NODOS:
        self.UI_show_nodes = False     # ✅✅✅      #####
        self.node_size = 4              #####
        self.node_color = 'blue'            #####
        # ELEMENTOS:
        self.UI_show_members = True        #####
        self.element_line_width = 2.0   #####
        self.element_color = 'blue'    #####
        # APOYOS:
        self.UI_support = True               ###########
        self.elastic_support_label = False
        self.support_size = 0.5        #####
        self.support_color = '#00ff00'    #####
        # OPCIONES DE SELECCIÓN:
        self.highlight_selected = True
        self.selected_color = 'red'
        # ETIQUETAS DE NODOS Y ELEMENTOS:
        self.UI_node_labels = False              ###########
        self.UI_member_labels = False           ###########
        self.label_font_size = 8                ##########
        self.node_label_color = '#ed0808'        ###########
        self.member_label_color = '#26a699'     ###########


        self.UI_load = True               ###########

        # CARGAS PUNTUALES
        self.point_load = self.UI_load #True               ###########
        self.point_load_color = 'black'           ###########
        self.point_load_length_arrow = 0.2 ############
        self.point_moment_length_arrow = 0.5 ########
        # ETIQUETAS DE CARGAS PUNTUALES
        self.point_load_label = self.UI_load #True                    ##########
        self.point_load_label_color = '#ff0892'         ###########
        self.point_load_label_font_size = 8             ###########

        # CARGAS DISTRIBUIDAS
        self.distributed_load = self.UI_load #True              #######
        self.scale_dist_qload = {}                     ####
        self.scale_dist_pload = {}                     ####
        self.distributed_load_color = '#831f7a'         ########
        # ETIQUETAS DE CARGAS DISTRIBUIDAS
        self.distributed_load_label = self.UI_load #True              #######
        self.distributed_load_label_color = '#511f74'   #######
        self.distributed_load_label_font_size = 8       #######

        self.UI_reactions = False              #######
        self.reactions_color = '#b62ded'           #######

        # DEFORMADA
        self.UI_deformation_scale = {} ########### # Factor de escala para deformaciones
        self.UI_deformed = False ########### # mostrar la deformada
        self.UI_rigid_deformed = False ########### # Color para deformaciones
        self.rigid_deformed_color = '#e81f64' ########### # Color para deformaciones
        self.deformation_line_width = 2.0 ######### # Ancho de línea para deformaciones
        self.deformation_color = 'blue' # #007acc' ########### # Color para deformacione
        # CON ESTOS DATOS DE ACTUALIZA DE FORMA SIN DEFORMAR automatixcamente, y se REVIERTE CON EL BOTON DE DEFORMADA
        self.show_undeformed = False    ########## # Mostrar estructura sin deformar
        self.undeformed_color = '#959a9d'   ######### Color para estructura sin deformar

        # ANOTACIONES DE LOS DEZPLAZAMIENTOS EN NODOS
        self.disp_nodes = True    ########## # Mostrar desplazamientos en nodos
        self.disp_nodes_color = 'black'  ########## # Color para desplazamientos en nodos
        self.disp_nodes_font_size = 8    ########## # Tamaño de fuente para desplazamientos en nodos

        # FUERZAS INTERNAS
        self.moment_on_tension_side = True     # (C| ---|Ɔ)
        self.fi_label = True
        self.fi_line_width = 1.0           # Ancho de línea de contorno para diagramas de esfuerzos
        self.UI_axial = False
        self.axial_scale = {}
        self.UI_shear = False
        self.shear_scale = {}
        self.UI_moment = False
        self.moment_scale = {}
        self.UI_slope = False
        self.slope_scale = {}
        self.UI_deflection = False
        self.deflection_scale = {}

        # RELLENOS Y CONTORNOS
        self.UI_filling_type = 'Barcolor'      # 'solid', 'barcolor'
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



        # TOPICOS:
        # self.end_length_offset = True    # Mostrar brazos en los elementos
        self.end_length_offset_color = 'black'  # Color para brazos en los elementos
        self.end_length_offset_line_width = 3.0  # Ancho de línea para brazos en los elementos

        # DESPLAMIENTOS PRESCRITOS:
        self.disp_pre_length_arrow = 0.3
        self.disp_pre_color = 'red'
        self.disp_pre_label_color = 'black'
        self.disp_pre_label_font_size = 8
        self.disp_pre_length_width = 2


        #! ELEMENTO CST:
        self.cst_alpha = 0.7
        self.cst_element_line_width = 0.7
        self.cst_element_label_color = 'black'
        self.cst_element_label_font_size = 8

        self.cst_edge_color = '#fa644f'
        self.cst_face_color = "#ffb44c"
        self.cst_deformed_color_edge = '#fa644f'
        self.cst_deformed_color_face = '#e37f7f'
        self.cst_undeformed_color_edge = '#b6f9ff'
        self.cst_undeformed_color_face = '#e3eaf3'


        #! ELEMENTO MEMBRANA Q6:
        self.membrane_q6_alpha = 0.7
        self.membrane_q6_element_line_width = 0.7
        self.membrane_q6_element_label_color = 'black'
        self.membrane_q6_element_label_font_size = 8

        self.membrane_q6_edge_color = '#fa644f'
        self.membrane_q6_face_color = "#ffb44c"
        self.membrane_q6_deformed_color_edge = '#fa644f'
        self.membrane_q6_deformed_color_face = '#e37f7f'
        self.membrane_q6_undeformed_color_edge = '#b6f9ff'
        self.membrane_q6_undeformed_color_face = '#e3eaf3'

        #! ELEMENTO MEMBRANA Q6I:
        self.membrane_q6i_alpha = 0.7
        self.membrane_q6i_element_line_width = 0.7
        self.membrane_q6i_element_label_color = 'black'
        self.membrane_q6i_element_label_font_size = 8

        self.membrane_q6i_edge_color = '#11272d'
        self.membrane_q6i_face_color = "#ffb44c"
        self.membrane_q6i_deformed_color_edge = '#fa644f'
        self.membrane_q6i_deformed_color_face = '#e37f7f'
        self.membrane_q6i_undeformed_color_edge = '#b6f9ff'
        self.membrane_q6i_undeformed_color_face = '#e3eaf3'


        #! FRAME RELEASE:
        self.frame_release_color = '#00ff00'
        self.frame_release_point_size = 20
        self.frame_release_length_offset = 0.07 # 5% de la longitud del miembro



    def reset(self, pattern_name: str):
        """
        Reinicia todas las opciones a sus valores predeterminados.
        """
        self.__init__(self.model)
        self.UI_label_font_size = self.label_size
        self.point_load_label_font_size = self.label_size
        self.distributed_load_label_font_size = self.label_size
        self.disp_nodes_font_size = self.relsult_label_size
        self.reactions_font_size = self.relsult_label_size
        self.load_max(pattern_name)

    def load_mean(self, pattern_name: str):
        """
        Calcula y asigna la media de los resultados para un patrón.
        """
        from numpy import nan
        val = self._calculate_mean(pattern_name)
        self.support_size = 0.10 * val["length_mean"] if round(val["length_mean"], 2) not in [0, None, nan] else 0.4
        self.scale_dist_qload[pattern_name] = 0.15 * val["length_mean"] / val["q_mean"] if val["q_mean"] not in [0, None, nan] else 0
        self.scale_dist_pload[pattern_name] = 0.05 * val["length_mean"] / val["p_mean"] if val["p_mean"] not in [0, None, nan] else 0
        self.point_load_length_arrow = 0.15 * val["length_mean"]
        self.point_moment_length_arrow = 0.075 * val["length_mean"]
        self.axial_scale[pattern_name] = 0.3 * val["length_mean"] / val["axial_mean"] if val["axial_mean"] not in [0, None, nan] else 0
        self.shear_scale[pattern_name] = 0.3 * val["length_mean"] / val["shear_mean"] if val["shear_mean"] not in [0, None, nan] else 0
        self.moment_scale[pattern_name] = 0.3 * val["length_mean"] / val["bending_mean"] if val["bending_mean"] not in [0, None, nan] else 0
        self.slope_scale[pattern_name] = 0.3 * val["length_mean"] / val["slope_mean"] if val["slope_mean"] not in [0, None, nan] else 100
        # self.deflection_scale[pattern_name] = 0.15 * val["length_mean"] / val["deflection_mean"]
        self.UI_deformation_scale[pattern_name] = 0.15 * val["length_mean"] / val["deflection_mean"] if val["deflection_mean"] not in [0, None, nan] else 100

    def load_max(self, pattern_name: str):
        """
        Calcula y asigna el máximo de los resultados para un patrón.
        """
        from numpy import nan
        val = self._calculate_max(pattern_name)
        self.support_size = 0.10 * val["length_max"] if round(val["length_max"], 2) not in [0, None, nan] else 0.4
        self.scale_dist_qload[pattern_name] = 0.15 * val["length_max"] / val["q_max"] if val["q_max"] not in [0, None, nan] else 0
        self.scale_dist_pload[pattern_name] = 0.05 * val["length_max"] / val["p_max"] if val["p_max"] not in [0, None, nan] else 0
        self.point_load_length_arrow = 0.15 * val["length_max"]
        self.point_moment_length_arrow = 0.075 * val["length_max"]
        self.axial_scale[pattern_name] = 0.3 * val["length_max"] / val["axial_max"] if val["axial_max"] not in [0, None, nan] else 0
        self.shear_scale[pattern_name] = 0.3 * val["length_max"] / val["shear_max"] if val["shear_max"] not in [0, None, nan] else 0
        self.moment_scale[pattern_name] = 0.3 * val["length_max"] / val["bending_max"] if val["bending_max"] not in [0, None, nan] else 0
        self.slope_scale[pattern_name] = 0.3 * val["length_max"] / val["slope_max"] if val["slope_max"] not in [0, None, nan] else 100
        # self.deflection_scale[pattern_name] = 0.15 * val["length_max"] / val["deflection_max"] if val["deflection_max"] not in [0, None, np.nan] else 0
        self.UI_deformation_scale[pattern_name] = 0.15 * val["length_max"] / val["deflection_max"] if val["deflection_max"] not in [0, None, nan] else 100

    def _calculate_max(self, pattern_name: str):
        """
        Calcula el máximo de los resultados para un patrón.
        """
        length_max = 0
        length_max_mq6 = 0
        length_max_mq6i = 0
        length_max_cst = 0
        q_max = 0
        p_max = 0
        deflection_max = 0
        slope_max = 0
        bending_max = 0
        shear_max = 0
        axial_max = 0
        for member, results in zip(self.model.members.values(), self.model.results[pattern_name].members.values()):
            dist_load = member.get_distributed_load(pattern_name)
            q_m = (abs(dist_load.q_i), abs(dist_load.q_j))
            p_m = (abs(dist_load.p_i), abs(dist_load.p_j))
            axial_m = abs(results["axial_forces"]).max()
            shear_m = abs(results["shear_forces"]).max()
            bending_m = abs(results["bending_moments"]).max()
            slope_m = abs(results["slopes"]).max()
            deflection_m = abs(results["deflections"]).max()

            length_max = member.length() if member.length() > length_max else length_max
            q_max = max(q_m) if max(q_m) > q_max else q_max
            p_max = max(p_m) if max(p_m) > p_max else p_max
            deflection_max = deflection_m if deflection_m > deflection_max else deflection_max
            slope_max = slope_m if slope_m > slope_max else slope_max
            bending_max = bending_m if bending_m > bending_max else bending_max
            shear_max = shear_m if shear_m > shear_max else shear_max
            axial_max = axial_m if axial_m > axial_max else axial_max

        for membrane_q6 in self.model.membrane_q6.values():
            maxxx = max(membrane_q6.node1.vertex.distance_to(membrane_q6.node3.vertex), membrane_q6.node2.vertex.distance_to(membrane_q6.node4.vertex))
            length_max_mq6 = maxxx if maxxx > length_max_mq6 else length_max_mq6

        for membrane_q6i in self.model.membrane_q6i.values():
            maxxx = max(membrane_q6i.node1.vertex.distance_to(membrane_q6i.node3.vertex), membrane_q6i.node2.vertex.distance_to(membrane_q6i.node4.vertex))
            length_max_mq6i = maxxx if maxxx > length_max_mq6i else length_max_mq6i

        for cst in self.model.csts.values():
            maxxx = max(cst.node1.vertex.distance_to(cst.node2.vertex), cst.node2.vertex.distance_to(cst.node3.vertex), cst.node3.vertex.distance_to(cst.node1.vertex))
            length_max_cst = maxxx if maxxx > length_max_cst else length_max_cst
        return {
            "length_max": max(length_max, length_max_mq6, length_max_mq6i, length_max_cst),
            "q_max": q_max,
            "p_max": p_max,
            "deflection_max": deflection_max,
            "slope_max": slope_max,
            "bending_max": bending_max,
            "shear_max": shear_max,
            "axial_max": axial_max
        }

    def _calculate_mean(self, pattern_name: str):
        """
        Calcula la media de los resultados para un patrón.
        """
        length_mean = 0
        length_mean_mq6 = 0
        length_mean_mq6i = 0
        length_mean_cst = 0
        q_mean = 0
        p_mean = 0
        deflection_mean = 0
        slope_mean = 0
        bending_mean = 0
        shear_mean = 0
        axial_mean = 0
        n = len(self.model.members) + len(self.model.membrane_q6) + len(self.model.membrane_q6i) + len(self.model.csts)
        for member, results in zip(self.model.members.values(), self.model.results[pattern_name].members.values()):
            dist_load = member.get_distributed_load(pattern_name)
            length_mean += member.length()
            q_mean += (abs(dist_load.q_i) + abs(dist_load.q_j))/2
            p_mean += (abs(dist_load.p_i) + abs(dist_load.p_j))/2
            deflection_mean += abs(results["deflections"]).max()
            slope_mean += abs(results["slopes"]).max()
            bending_mean += abs(results["bending_moments"]).max()
            shear_mean += abs(results["shear_forces"]).max()
            axial_mean += abs(results["axial_forces"]).max()

        for membrane_q6 in self.model.membrane_q6.values():
            length_mean_mq6 += mean([membrane_q6.node1.vertex.distance_to(membrane_q6.node3.vertex), membrane_q6.node2.vertex.distance_to(membrane_q6.node4.vertex)])

        for membrane_q6i in self.model.membrane_q6i.values():
            length_mean_mq6i += mean([membrane_q6i.node1.vertex.distance_to(membrane_q6i.node3.vertex), membrane_q6i.node2.vertex.distance_to(membrane_q6i.node4.vertex)])

        for cst in self.model.csts.values():
            length_mean_cst += mean([cst.node1.vertex.distance_to(cst.node2.vertex), cst.node2.vertex.distance_to(cst.node3.vertex), cst.node3.vertex.distance_to(cst.node1.vertex)])
        return {
            "length_mean": (length_mean + length_mean_mq6 + length_mean_mq6i + length_mean_cst)/n,
            "q_mean": q_mean/n,
            "p_mean": p_mean/n,
            "deflection_mean": deflection_mean/n,
            "slope_mean": slope_mean/n,
            "bending_mean": bending_mean/n,
            "shear_mean": shear_mean/n,
            "axial_mean": axial_mean/n
        }

    def nro_arrows(self, member_id: int):
        """
        Calcula el número de flechas para un miembro.
        """
        dx = self.support_size/0.10 * (0.10)
        dxx = int(self.model.members[member_id].length()/dx)
        return dxx if dxx > 0 else 1
