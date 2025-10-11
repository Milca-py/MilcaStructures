'''
Modulo para el visualizado dl modelo al mismtiempo tiempo que se modela
'''

import itertools


import matplotlib.pyplot as plt
import numpy as np
import milcapy as mp
from typing import Optional


from milcapy.plotter.options import PlotterOptions
from matplotlib.ticker import AutoMinorLocator


from milcapy.plotter.suports import (
    support_ttt, support_ttf, support_tft,
    support_ftt, support_tff, support_ftf, support_fft,
    support_kt, support_kr
)

from milcapy.plotter.load import (
    graphic_one_arrow, graphic_one_arrow_dof, moment_fancy_arrow, graphic_n_arrow, redondear_si_mas_de_3_decimales
)

from milcapy.utils import rotate_xy, traslate_xy


def initialize_figure(options: PlotterOptions) -> tuple[plt.Figure, plt.Axes]:
    # Cerrar figuras previas
    plt.close("all")

    # Configurar estilo global
    if options.plot_style in plt.style.available:
        plt.style.use(options.plot_style)

    # Crear figura y ejes
    fig = plt.figure(figsize=options.figure_size,
                                dpi=options.dpi, facecolor=options.UI_background_color)
    ax = fig.add_subplot(111)

    # Configurar cuadrícula
    if options.grid:
        ax.grid(True, linestyle="--", alpha=0.5)

    # Ajustar layout
    if options.tight_layout:
        fig.tight_layout()

    # Mantener proporciones iguales
    plt.axis("equal")

    # Activar los ticks secundarios en ambos ejes
    # 5 subdivisiones entre cada tick principal
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    # Activar ticks en los 4 lados (mayores y menores)
    ax.tick_params(
        which="both", direction="in", length=6, width=1,
        top=True, bottom=True, left=True, right=True
    )
    # Ticks menores más pequeños y rojos
    ax.tick_params(which="minor", length=2,
                            width=0.5, color="black")

    # Mostrar etiquetas en los 4 lados
    ax.tick_params(labeltop=True, labelbottom=True,
                            labelleft=True, labelright=True)

    # Asegurar que los ticks se muestran en ambos lados
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    # Personalizar el color de los ejes
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_color("#9bc1bc")  # Color personalizado
        ax.spines[spine].set_linewidth(0.5)  # Grosor del borde

    # Personalizar las etiquetas de los ejes
    plt.xticks(fontsize=8, fontfamily="serif",
                fontstyle="italic", color="#103b58")
    plt.yticks(fontsize=8, fontfamily="serif",
                fontstyle="italic", color="#103b58")

    # Personalizar los ticks del eje X e Y
    ax.tick_params(axis="x", direction="in",
                            length=3.5, width=0.7, color="#21273a")
    ax.tick_params(axis="y", direction="in",
                            length=3.5, width=0.7, color="#21273a")

    # Cambiar el color de fondo del área de los ejes
    # self.axes.set_facecolor("#222222")  # Fondo oscuro dentro del Axes

    # Cambiar color del fondo exterior (Canvas)
    fig.patch.set_facecolor("#f5f5f5")  # Color gris oscuro
    return fig, ax



def plot_model(
    model: mp.SystemMilcaModel,
    load_pattern: Optional[str] = None,
    node_labels: bool = True,
    member_labels: bool = True,
    ):
    '''
    Funcion para visualizar el modelo al mismo tiempo que se modela
    '''
    fig, ax = initialize_figure(model.plotter_options)
    plt_opt = model.plotter_options
    length_mean, q_mean, p_mean = plt_opt._calculate_params_mean(load_pattern)

    #% Plotear los nodos:
    for node in model.nodes.values():
        x = [node.vertex.x]
        y = [node.vertex.y]

        ax.scatter(x, y, c=plt_opt.node_color, s=plt_opt.node_size, marker='o')
        if node_labels:
            ax.text(x[0], y[0], str(node.id),
                                fontsize=plt_opt.label_font_size,
                                ha='left', va='bottom', color="blue",
                                clip_on=True)

    #% plotear los elementos frame:
    for member in model.members.values():
        x_coords = [member.node_i.vertex.x, member.node_j.vertex.x]
        y_coords = [member.node_i.vertex.y, member.node_j.vertex.y]
        ax.plot(x_coords, y_coords, color=plt_opt.element_color,
                                linewidth=plt_opt.element_line_width)
        if member_labels:
            x_val = (x_coords[0] + x_coords[1]) / 2
            y_val = (y_coords[0] + y_coords[1]) / 2
            ax.text(x_val, y_val, str(member.id),
                                fontsize=plt_opt.label_font_size,
                                ha='left', va='bottom', color="black",
                                clip_on=True)

    #% plotear los elementos quad:
    for quad in list(model.membrane_q3dof.values()) + list(model.membrane_q2dof.values()):
        x, y = quad.get_coordinates()

        ax.fill(x, y,
                color=plt_opt.membrane_q3dof_face_color,
                alpha=plt_opt.membrane_q3dof_alpha,
                edgecolor='red',
                linewidth=1)

        if member_labels:
            ax.text(np.mean(x), np.mean(y), f"M: {quad.id}",
                fontsize=plt_opt.label_font_size,
                ha='left', va='bottom', color="blue",
                clip_on=True)

    #% plotear los elementos tri:
    for cst in model.csts.values():
        x, y = cst.get_coordinates()

        ax.plot(x, y,
                color=plt_opt.cst_edge_color,
                linewidth=plt_opt.cst_element_line_width)

        ax.fill(x, y,
                color=plt_opt.cst_face_color,
                alpha=plt_opt.cst_alpha,
                edgecolor='red',
                linewidth=1)

        if member_labels:
            ax.text(np.mean(x), np.mean(y), f"M: {cst.id}",
                                    fontsize=plt_opt.label_font_size,
                                    ha='left', va='bottom', color="blue",
                                clip_on=True)

    #% plotear los elementos truss:
    for truss in model.trusses.values():
        x_coords = [truss.node_i.vertex.x, truss.node_j.vertex.x]
        y_coords = [truss.node_i.vertex.y, truss.node_j.vertex.y]
        ax.plot(x_coords, y_coords, color=plt_opt.truss_color,
                                linewidth=plt_opt.element_line_width)
        if member_labels:
            x_val = (x_coords[0] + x_coords[1]) / 2
            y_val = (y_coords[0] + y_coords[1]) / 2
            ax.text(x_val, y_val, str(truss.id),
                                fontsize=plt_opt.label_font_size,
                                ha='left', va='bottom', color="black",
                                clip_on=True)

    #% plotear los brazos rigidos:
    for member in model.members.values():
        la = member.la or 0
        lb = member.lb or 0
        length = member.length()
        length = length - la - lb
        xi, yi = member.node_i.vertex.x, member.node_i.vertex.y
        xj, yj = member.node_j.vertex.x, member.node_j.vertex.y
        angle_rotation = member.angle_x()
        coords = ((xi + la*np.cos(angle_rotation), yi + la*np.sin(angle_rotation)),
                    (xj - lb*np.cos(angle_rotation), yj - lb*np.sin(angle_rotation)))
        ((xa, ya), (xb, yb)) = coords
        coords_a = [[xi, xa], [yi, ya]]
        coords_b = [[xb, xj], [yb, yj]]

        if member.la:
            line, = ax.plot(coords_a[0], coords_a[1], color=plt_opt.end_length_offset_color,
                                    linewidth=plt_opt.end_length_offset_line_width)
        if member.lb:
            line, = ax.plot(coords_b[0], coords_b[1], color=plt_opt.end_length_offset_color,
                                    linewidth=plt_opt.end_length_offset_line_width)

    #% plotear los soportes y resortes:
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

    for node in model.nodes.values():
        if node.restraints != (False, False, False):
            node_coords = node.vertex.x, node.vertex.y
            support_func = support_functions.get(node.restraints)

            if node.local_axis:
                theta = node.local_axis.angle
            else:
                theta = plt_opt.mod_rotation_angle_conventional_supports.get(node.id, 0)*3.141592653589793/180.
            if support_func:
                line = support_func(
                    ax=ax,
                    x=node_coords[0],
                    y=node_coords[1],
                    size=length_mean * 0.1,
                    color=plt_opt.support_color,
                    zorder=3,
                    theta=theta
                )


    theta_krz = getattr(plt_opt, "mod_krz_rotation_angle", 135)
    if isinstance(theta_krz, (list, tuple)):
        theta_iter = iter(theta_krz)
    elif isinstance(theta_krz, (int, float)):
        theta_iter = itertools.repeat(theta_krz)
    elif isinstance(theta_krz, dict):
        pass  # not implemented yet
    else:
        theta = 135  # Default value if the input is invalid

    for node in model.nodes.values():
        if node.elastic_supports:
            node_coords = node.vertex.x, node.vertex.y
            kx, ky, krz = node.elastic_supports.get_elastic_supports()
            LA = node.local_axis
            if LA is not None:
                T = LA.get_transformation_matrix()
            else:
                T = np.eye(3)
            kx, ky, krz = T @ np.array([kx or 0, ky or 0, krz or 0])
            artist = []
            data_artist = {}
            if redondear_si_mas_de_3_decimales(kx) != 0:
                theta = 0  # if kx < 0 else 180
                line = support_kt(
                    ax,
                    node_coords[0],
                    node_coords[1],
                    length_mean * 0.1,
                    plt_opt.support_color,
                    theta=theta
                )
                x_data, y_data = line.get_data()
                coord_label = [
                    node_coords[0]-plt_opt.support_size/2, node_coords[1]]
                coord_label = rotate_xy(
                    coord_label, theta, node_coords[0], node_coords[1])
                if plt_opt.elastic_support_label:
                    text = ax.text(coord_label[0], coord_label[1], f"kx = {redondear_si_mas_de_3_decimales(kx)}",
                                            fontsize=plt_opt.label_font_size, color=plt_opt.node_label_color)
                    data_artist["kx"] = (x_data, y_data, theta, line, text)
                else:
                    data_artist["kx"] = (x_data, y_data, theta, line)

            if redondear_si_mas_de_3_decimales(ky) != 0:
                theta = 90  # if ky < 0 else 270
                line = support_kt(
                    ax,
                    node_coords[0],
                    node_coords[1],
                    length_mean * 0.1,
                    plt_opt.support_color,
                    theta=theta
                )
                x_data, y_data = line.get_data()
                coord_label = [
                    node_coords[0]-plt_opt.support_size/2, node_coords[1]]
                coord_label = rotate_xy(
                    coord_label, theta, node_coords[0], node_coords[1])
                if plt_opt.elastic_support_label:
                    text = ax.text(coord_label[0], coord_label[1], f"ky = {redondear_si_mas_de_3_decimales(ky)}",
                                            fontsize=plt_opt.label_font_size, color=plt_opt.node_label_color)
                    data_artist["ky"] = (x_data, y_data, theta, line, text)
                else:
                    data_artist["ky"] = (x_data, y_data, theta, line)

            if redondear_si_mas_de_3_decimales(krz) != 0:

                if isinstance(theta_krz, dict):
                    theta = theta_krz.get(node_id, 135)
                else:
                    try:
                        theta = next(theta_iter)
                    except StopIteration:
                        theta = 135

                line = support_kr(
                    ax,
                    node_coords[0],
                    node_coords[1],
                    length_mean * 0.1,
                    plt_opt.support_color,
                    theta=theta
                )

                x_data, y_data = line.get_data()
                coord_label = [
                    node_coords[0]-plt_opt.support_size/2, node_coords[1]]
                coord_label = rotate_xy(
                    coord_label, theta, node_coords[0], node_coords[1])
                if plt_opt.elastic_support_label:
                    text = ax.text(coord_label[0], coord_label[1], f"krz = {redondear_si_mas_de_3_decimales(krz)}",
                                            fontsize=plt_opt.label_font_size, color=plt_opt.node_label_color)
                    data_artist["krz"] = (x_data, y_data, theta, line, text)
                else:
                    data_artist["krz"] = (x_data, y_data, theta, line)

    #% plotear las cargas nodales y distribuidas:
    if load_pattern and model.load_patterns.get(load_pattern):
        for node_id, load in model.load_patterns[load_pattern].point_loads.items():
            x, y = model.nodes[node_id].vertex.x, model.nodes[node_id].vertex.y

            # Fuerza en dirección X
            if load.fx != 0:
                graphic_one_arrow(
                    x=x,
                    y=y,
                    load=load.fx,
                    length_arrow=length_mean*0.1,
                    angle=0 if load.fx < 0 else np.pi,
                    ax=ax,
                    color=plt_opt.point_load_color,
                    label=plt_opt.point_load_label,
                    color_label=plt_opt.point_load_label_color,
                    label_font_size=plt_opt.point_load_label_font_size
                )

            # Fuerza en dirección Y
            if load.fy != 0:
                graphic_one_arrow(
                    x=x,
                    y=y,
                    load=load.fy,
                    length_arrow=length_mean*0.1,
                    angle=np.pi/2 if load.fy < 0 else 3*np.pi/2,
                    ax=ax,
                    color=plt_opt.point_load_color,
                    label=plt_opt.point_load_label,
                    color_label=plt_opt.point_load_label_color,
                    label_font_size=plt_opt.point_load_label_font_size
                )

            # Momento en Z
            if load.mz != 0:
                moment_fancy_arrow(
                    ax=ax,
                    x=x,
                    y=y,
                    moment=load.mz,
                    radio=length_mean*0.07,
                    color=plt_opt.point_load_color,
                    clockwise=True,
                    label=plt_opt.point_load_label,
                    color_label=plt_opt.point_load_label_color,
                    label_font_size=plt_opt.point_load_label_font_size
                )


        for id_element, load in model.load_patterns[load_pattern].distributed_loads.items():

            # Calcular longitud y ángulo de rotación del elemento
            if id_element in model.trusses:
                element = model.trusses[id_element]
                element.qla, element.qlb = None, None
            else:
                element = model.members[id_element]
            length = element.length()
            angle_rotation = element.angle_x()

            if not (element.qla and element.qlb):  # Si no hay cargas en el brazo rigido
                la = element.la or 0
                lb = element.lb or 0
                length = length - la - lb
                if id_element in model.trusses:
                    e = model.trusses[id_element]
                    xi, yi, xj, yj = e.node_i.vertex.x, e.node_i.vertex.y, e.node_j.vertex.x, e.node_j.vertex.y
                else:
                    e = model.members[id_element]
                    xi, yi, xj, yj = e.node_i.vertex.x, e.node_i.vertex.y, e.node_j.vertex.x, e.node_j.vertex.y
                coords = ((xi + la*np.cos(angle_rotation), yi + la*np.sin(angle_rotation)),
                        (xj - lb*np.cos(angle_rotation), yj - lb*np.sin(angle_rotation)))
            else:
                if id_element in model.trusses:
                    e = model.trusses[id_element]
                    xi, yi, xj, yj = e.node_i.vertex.x, e.node_i.vertex.y, e.node_j.vertex.x, e.node_j.vertex.y
                else:
                    e = model.members[id_element]
                    xi, yi, xj, yj = e.node_i.vertex.x, e.node_i.vertex.y, e.node_j.vertex.x, e.node_j.vertex.y
                coords = ((xi, yi), (xj, yj))

            # Cargas transversales
            if round(load.q_i, 2) != 0.00 or round(load.q_j, 2) != 0.00:
                graphic_n_arrow(
                    x=coords[0][0],
                    y=coords[0][1],
                    load_i=-redondear_si_mas_de_3_decimales(load.q_i),
                    load_j=-redondear_si_mas_de_3_decimales(load.q_j),
                    angle=np.pi/2,
                    length=length,
                    ax=ax,
                    ratio_scale=length_mean/q_mean * 0.04,
                    nrof_arrows=plt_opt.nro_arrows(id_element),
                    color=plt_opt.distributed_load_color,
                    angle_rotation=angle_rotation,
                    label=plt_opt.distributed_load_label,
                    color_label=plt_opt.distributed_load_label_color,
                    label_font_size=plt_opt.distributed_load_label_font_size
                )

            # Cargas axiales
            if round(load.p_i, 2) != 0.00 or round(load.p_j, 2) != 0.00:
                graphic_n_arrow(
                    x=coords[0][0],
                    y=coords[0][1],
                    load_i=-redondear_si_mas_de_3_decimales(load.p_i),
                    load_j=-redondear_si_mas_de_3_decimales(load.p_j),
                    angle=0,
                    length=length,
                    ax=ax,
                    ratio_scale=length_mean/p_mean * 0.04,
                    nrof_arrows=plt_opt.nro_arrows(id_element),
                    color=plt_opt.distributed_load_color,
                    angle_rotation=angle_rotation,
                    label=plt_opt.distributed_load_label,
                    color_label=plt_opt.distributed_load_label_color,
                    label_font_size=plt_opt.distributed_load_label_font_size,
                    length_arrow=length_mean * 0.05
                )


        for id_node, PDOF in model.load_patterns[load_pattern].prescribed_dof.items():
            x, y = model.nodes[id_node].vertex.x, model.nodes[id_node].vertex.y
            arrowstyle = "-|>"
            # dezplamientos en dirección X
            if PDOF.ux != 0 and PDOF.ux is not None:
                if model.nodes[id_node].local_axis is not None:
                    if PDOF.ux > 0:
                        angle = model.nodes[id_node].local_axis.angle
                    else:
                        angle = model.nodes[id_node].local_axis.angle + np.pi
                else:
                    angle = np.pi if PDOF.ux < 0 else 0
                graphic_one_arrow_dof(
                    x=x,
                    y=y,
                    load=PDOF.ux,
                    length_arrow=length_mean * 0.1,
                    angle=angle,
                    ax=ax,
                    color=plt_opt.disp_pre_color,
                    label=plt_opt.point_load_label,
                    color_label=plt_opt.disp_pre_label_color,
                    label_font_size=plt_opt.disp_pre_label_font_size,
                    lw=plt_opt.disp_pre_length_width,
                    arrowstyle=arrowstyle
                )

            # Fuerza en dirección Y
            if PDOF.uy != 0 and PDOF.uy is not None:
                if model.nodes[id_node].local_axis is not None:
                    if PDOF.uy > 0:
                        angle = np.pi/2 + \
                            model.nodes[id_node].local_axis.angle
                    else:
                        angle = 3*np.pi/2 + \
                            model.nodes[id_node].local_axis.angle
                else:
                    angle = np.pi/2 if PDOF.uy > 0 else 3*np.pi/2
                graphic_one_arrow_dof(
                    x=x,
                    y=y,
                    load=PDOF.uy,
                    length_arrow=length_mean * 0.1,
                    angle=angle,  # np.pi/2 if PDOF.uy < 0 else 3*np.pi/2,
                    ax=ax,
                    color=plt_opt.disp_pre_color,
                    label=plt_opt.point_load_label,
                    color_label=plt_opt.disp_pre_label_color,
                    label_font_size=plt_opt.disp_pre_label_font_size,
                    lw=plt_opt.disp_pre_length_width,
                    arrowstyle=arrowstyle
                )

            # Momento en Z
            if PDOF.rz != 0 and PDOF.rz is not None:
                moment_fancy_arrow(
                    ax=ax,
                    x=x,
                    y=y,
                    moment=PDOF.rz,
                    radio=0.50 * length_mean * 0.07,
                    color=plt_opt.disp_pre_color,
                    clockwise=True,
                    label=plt_opt.point_load_label,
                    color_label=plt_opt.disp_pre_label_color,
                    label_font_size=plt_opt.disp_pre_label_font_size,
                    lw=plt_opt.disp_pre_length_width,
                    arrowstyle=arrowstyle
                )


    #% plotear los releases:
    for member in model.members.values():
        if member.release:
            length = member.length()
            offset = plt_opt.frame_release_length_offset * length
            release = member.release.get_dof_release()
            scatter = []

            # Nodo i: liberaciones en 0, 1 o 2
            if any(r in release for r in (0, 1, 2)):
                UnitVector = (member.node_j.vertex.coordinates -
                                member.node_i.vertex.coordinates) / length
                point = UnitVector * offset + member.node_i.vertex.coordinates
                scatter.append(
                    ax.scatter(
                        point[0], point[1],
                        s=plt_opt.frame_release_point_size,
                        color=plt_opt.frame_release_color,
                        zorder=80
                    )
                )

            # Nodo j: liberaciones en 3, 4 o 5
            if any(r in release for r in (3, 4, 5)):
                UnitVector = (member.node_j.vertex.coordinates -
                                member.node_i.vertex.coordinates) / length
                point = -UnitVector * offset + member.node_j.vertex.coordinates
                scatter.append(
                    ax.scatter(
                        point[0], point[1],
                        s=plt_opt.frame_release_point_size,
                        color=plt_opt.frame_release_color,
                        zorder=80
                    )
                )

    #% mostrar grafica:
    plt.show()