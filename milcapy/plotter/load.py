import numpy as np
from typing import TYPE_CHECKING
from matplotlib.patches import FancyArrowPatch
from milcapy.utils import rotate_xy, vertex_range


if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

def _correction_angle(angle: float) -> float:
    """
    Corrige el angulo para que se dibuje correctamente
    """
    if 0 <= angle <= 90 or 270 < angle < 360:
        return angle
    elif 90 < angle <= 270:
        return (angle - 180)


def graphic_one_arrow(
    x: float,
    y: float,
    load: float,
    length_arrow: float,
    angle: float,
    ax: "Axes",
    color: str = "blue",
    label: bool = True,
    color_label: str = "black",
    label_font_size: int = 8,
    lw: float = 1,
    arrowstyle: str = "->"
) -> None:
    """
    Dibuja una flecha en un punto.
    """
    a = np.array([x, y])
    b = np.array([x + length_arrow * np.cos(angle), y + length_arrow * np.sin(angle)])

    # coordenadas al 15% de la punta de la flecha
    coord15p = np.array([x + 0.85 * length_arrow * np.cos(angle), y + 0.85 * length_arrow * np.sin(angle)])
    arrow = FancyArrowPatch(
        b, a,
        transform=ax.transData,
        color=color,
        linewidth=lw,
        arrowstyle=arrowstyle,
        mutation_scale=10,
        zorder=80
    )
    ax.add_patch(arrow)


    if label:
        text =ax.text(
            coord15p[0], coord15p[1],
            f"{redondear_si_mas_de_3_decimales(abs(load))}",
            fontsize=label_font_size,
            ha="right",
            va="bottom",
            color=color_label,
            rotation= _correction_angle(angle*180/np.pi),
            zorder=80
        )
    return arrow, text

def graphic_one_arrow_dof(
    x: float,
    y: float,
    load: float,
    length_arrow: float,
    angle: float,
    ax: "Axes",
    color: str = "blue",
    label: bool = True,
    color_label: str = "black",
    label_font_size: int = 8,
    lw: float = 1,
    arrowstyle: str = "->"
) -> None:
    """
    Dibuja una flecha en un punto.
    """
    a = np.array([x, y])
    b = np.array([x - length_arrow * np.cos(angle), y - length_arrow * np.sin(angle)])

    # coordenadas al 15% de la punta de la flecha
    coord15p = np.array([x - 0.85 * length_arrow * np.cos(angle), y - 0.85 * length_arrow * np.sin(angle)])
    arrow = FancyArrowPatch(
        b, a,
        transform=ax.transData,
        color=color,
        linewidth=lw,
        arrowstyle=arrowstyle,
        mutation_scale=10,
        zorder=80
    )
    ax.add_patch(arrow)


    if label:
        text =ax.text(
            coord15p[0], coord15p[1],
            f"{redondear_si_mas_de_3_decimales(abs(load))}",
            fontsize=label_font_size,
            ha="right",
            va="bottom",
            color=color_label,
            rotation= _correction_angle(angle*180/np.pi),
            zorder=80
        )
    return arrow, text



# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()

# x = 0.65
# y = 0.5
# load = 400
# length_arrow = 0.5
# angle = np.pi/180 * 180
# graphic_one_arrow(x, y, load, length_arrow, angle, ax)
# plt.show()


# """
# si: X -> (-)=0, (+)=180
# si: Y -> (-)=90, (+)=270
# """


def graphic_n_arrow(
    x: float,
    y: float,
    load_i: float,
    load_j: float,
    angle: float,
    length: float,
    ax: "Axes",
    ratio_scale: float,
    nrof_arrows: int,
    color: str = "blue",
    angle_rotation: float = 0,
    label: bool = True,
    color_label: str = 'blue',
    label_font_size: int = 8
) -> None:
    """
    Dibuja una flecha en un punto.
    """
    # load_i, load_j = -load_i, -load_j
    # coordenadas de los extremos de la barra
    a = rotate_xy(np.array([x, y]), angle_rotation, x, y)
    b = rotate_xy(np.array([x + length, y]), angle_rotation, x, y)

    # coordenadas de los extremos carga
    c = rotate_xy(np.array([x + load_i * ratio_scale * np.cos(angle), y + load_i * ratio_scale * np.sin(angle)]), angle_rotation, x, y)
    d = rotate_xy(np.array([x + load_j * ratio_scale * np.cos(angle) + length, y + load_j * ratio_scale * np.sin(angle)]), angle_rotation, x, y)

    cood_i = vertex_range(c, d, nrof_arrows)
    cood_j = vertex_range(a, b, nrof_arrows)

    # dibujar flechas
    arrows = []
    for start, end in zip(cood_i, cood_j):
        arrow = FancyArrowPatch(
            start, end,
            transform=ax.transData,
            color=color,
            linewidth=0.7,
            arrowstyle="->",
            mutation_scale=7,
            zorder=80
        )
        ax.add_patch(arrow)
        arrows.append(arrow)
    # linea que une las flechas
    line =ax.plot([c[0], d[0]], [c[1], d[1]], linewidth=0.7, color=color, zorder=80)

    # texto de la flecha
    texts = []
    if label:
        if load_i == load_j:
            coord_label = (c + d) / 2
            text = ax.text(
                coord_label[0], coord_label[1],
                f"{abs(load_i)}",
                fontsize=label_font_size,
                # ha="center",
                # va="center",
                color=color_label,
                zorder=80
            )
            texts.append(text)
        else:
            coord_label_i = c
            coord_label_j = d
            if load_i == 0:
                pass
            else:
                text = ax.text(
                    coord_label_i[0], coord_label_i[1],
                    f"{abs(load_i)}",
                    fontsize=label_font_size,
                    # ha="center",
                    # va="center",
                    color=color_label,
                    zorder=80
                )
                texts.append(text)
            if load_j == 0:
                pass
            else:
                text = ax.text(
                    coord_label_j[0], coord_label_j[1],
                    f"{abs(load_j)}",
                    fontsize=label_font_size,
                    # ha="center",
                    # va="center",
                    color=color_label,
                    zorder=80
                )
                texts.append(text)

        arrows.extend(line)  # Desempaqueta la lista de Line2D en arrows

        return arrows, texts


# x = 0
# y = 0
# load_i = 2
# load_j = 1
# length = 10
# angle = np.pi/2
# ratio_scale = 1.0
# nrof_arrows = 10
# color = 'blue'
# angle_rotation = 45
# label = True
# color_label = 'blue'
# label_font_size = 8

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()

# graphic_n_arrow(
#     x, y, load_i, load_j, angle, length, ax, ratio_scale, nrof_arrows, color, angle_rotation, label, color_label, label_font_size)
# plt.axis("equal")
# plt.show()

"""
si: X -> (-)=0, (+)=180
si: Y -> (-)=90, (+)=270
"""


def redondear_si_mas_de_3_decimales(num):
    """
    Redondea un número a 3 decimales solo si:
      - Es float, y
      - Tiene más de 3 decimales.
    Si es entero, lo devuelve como int.
    Elimina ceros innecesarios en los decimales.
    """
    # Si es equivalente a entero (int o float sin decimales)
    if float(num).is_integer():
        return int(num)

    # Si es float, contamos decimales reales
    num_str = f"{num:.16f}".rstrip("0").rstrip(".")
    if '.' in num_str:
        decimales = len(num_str.split('.')[1])
        if decimales > 3:
            num = round(num, 3)

    # Convertimos a str y quitamos ceros innecesarios
    return float(f"{num:.16f}".rstrip("0").rstrip("."))




def moment_fancy_arrow(
    ax: "Axes",
    x: float,
    y: float,
    moment: float,
    radio: float,
    color: str = 'blue',
    clockwise: bool = True,
    label: bool = True,
    color_label: str = 'blue',
    label_font_size: int = 8,
    lw: float = 1,
    arrowstyle: str = "->",
    ) :
    """
    Dibuja una flecha arco en un punto.
    """
    r = radio
    if moment < 0:
        curvature: float = -1.0 if clockwise else 1.0
    else:
        curvature: float = 1.0 if clockwise else -1.0

    arrow = FancyArrowPatch(
        (x - r, y - r), (x + r, y + r),
        connectionstyle=f"arc3,rad={curvature}",
        arrowstyle=arrowstyle,
        color=color,
        lw=lw,
        mutation_scale=10,
        zorder=80
    )
    ax.add_patch(arrow)

    if moment < 0:
        pos_label = (x - np.cos(np.pi/4) *(1.2)* r, y + np.cos(np.pi/4) * (1.8)*r)
    else:
        pos_label = (x + np.cos(np.pi/4) *(1.8)* r, y - np.cos(np.pi/4) * (2.1)*r)
    if label:
        text = ax.text(
            pos_label[0], pos_label[1],
            f"{redondear_si_mas_de_3_decimales(abs(moment))}",
            fontsize=label_font_size,
            ha="right",
            va="bottom",
            color=color_label,
            zorder=80
        )
    return arrow, text



# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()

# x = 0.65
# y = 0.5
# moment = -400
# radio = 0.1
# arrow, text = moment_fancy_arrow(ax, x, y, moment, radio)
# plt.show()







def analyze_areas(points):
    """
    Analiza las áreas de un polígono.
    """
    if not points:
        return []

    def get_sign(y):
        return 1 if y >= 0 else -1

    areas = []
    current_area = [points[0]]
    current_sign = get_sign(points[0][1])

    for i in range(1, len(points)):
        current_point = points[i-1]
        next_point = points[i]
        next_sign = get_sign(next_point[1])

        if current_sign == next_sign:
            current_area.append(next_point)
        else:
            # Calcular la intersección
            x1, y1 = current_point
            x2, y2 = next_point
            delta_y = y2 - y1
            t = (-y1) / delta_y
            x_intercept = x1 + t * (x2 - x1)
            intersection = [x_intercept, 0.0]

            current_area.append(intersection)
            areas.append(current_area)

            # Iniciar nueva área
            current_area = [intersection, next_point]
            current_sign = next_sign

    # Añadir el área actual restante
    areas.append(current_area)

    return areas

# import numpy as np
# import matplotlib.pyplot as plt


# # Función para rotar puntos 45°
# def rotate_45(points):
#     theta = np.radians(45)
#     rotation_matrix = np.array([
#         [np.cos(theta), -np.sin(theta)],
#         [np.sin(theta), np.cos(theta)]
#     ])
#     rotated = []
#     for point in points:
#         rotated_point = rotation_matrix @ np.array(point)
#         rotated.append(rotated_point)
#     return np.array(rotated)


# # Generar datos de ejemplo
# x = np.linspace(0, 10, 100)
# y = np.sin(x) * 2  # Función con cambios de signo
# points = np.column_stack((x, y)).tolist()

# # Analizar áreas
# areas = analyze_areas(points)

# # Configurar el plot
# plt.figure(figsize=(12, 6))

# # Plot original
# plt.subplot(1, 2, 1)
# plt.title("Original")
# for i, area in enumerate(areas):
#     area = np.array(area)
#     color = 'blue' if area[0,1] >= 0 else 'red'
#     plt.plot(area[:,0], area[:,1], color=color, marker='o', markersize=3)
#     plt.fill_between(area[:,0], area[:,1], 0, color=color, alpha=0.2)
# plt.axhline(0, color='black', linestyle='--')
# plt.grid(True)

# # Plot rotado 45°
# plt.subplot(1, 2, 2)
# plt.title("Rotado 45°")
# for i, area in enumerate(areas):
#     rotated_area = rotate_45(area)
#     color = 'blue' if area[0][1] >= 0 else 'red'
#     plt.plot(rotated_area[:,0], rotated_area[:,1], color=color, marker='o', markersize=3)
#     plt.fill(rotated_area[:,0], rotated_area[:,1], color=color, alpha=0.2)
# plt.axhline(0, color='black', linestyle='--')
# plt.grid(True)

# plt.tight_layout()
# plt.show()