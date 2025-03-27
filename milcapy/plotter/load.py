import numpy as np
from typing import TYPE_CHECKING
from matplotlib.patches import FancyArrowPatch
from milcapy.utils import rotate_xy, vertex_range


if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

def _correction_angle(angle: float) -> float:
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
    label_font_size: int = 8
) -> None:
    """Dibuja una flecha en un punto."""
    a = np.array([x, y])
    b = np.array([x + length_arrow * np.cos(angle), y + length_arrow * np.sin(angle)])

    # coordenadas al 15% de la punta de la flecha
    coord15p = np.array([x + 0.85 * length_arrow * np.cos(angle), y + 0.85 * length_arrow * np.sin(angle)])
    arrow = FancyArrowPatch(
        b, a,
        transform=ax.transData,
        color=color,
        linewidth=0.7,
        arrowstyle="->",
        mutation_scale=10
    )
    ax.add_patch(arrow)


    if label:
        text =ax.text(
            coord15p[0], coord15p[1],
            f"{abs(load)}",
            fontsize=label_font_size,
            ha="right",
            va="bottom",
            color=color_label,
            rotation= _correction_angle(angle*180/np.pi)
        )
    return arrow, text



# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()

# x = 0.65
# y = 0.5
# load = 400
# length_arrow = 0.5
# angle = np.pi/180 * 90
# graphic_one_arrow(x, y, load, length_arrow, angle, ax)
# plt.show()


"""
si: X -> (-)=0, (+)=180
si: Y -> (-)=90, (+)=270
"""
















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
    """Dibuja una flecha en un punto."""
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
            mutation_scale=7
        )
        ax.add_patch(arrow)
        arrows.append(arrow)
    # linea que une las flechas
    line =ax.plot([c[0], d[0]], [c[1], d[1]], linewidth=0.7, color=color)

    # texto de la flecha
    texts = []
    if label:
        if load_i == load_j:
            coord_label = (c + d) / 2
            text = ax.text(
                coord_label[0], coord_label[1],
                f"{abs(load_i):.2f}",
                fontsize=label_font_size,
                # ha="center",
                # va="center",
                color=color_label
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
                    f"{abs(load_i):.2f}",
                    fontsize=label_font_size,
                    # ha="center",
                    # va="center",
                    color=color_label
                )
                texts.append(text)
            if load_j == 0:
                pass
            else:
                text = ax.text(
                    coord_label_j[0], coord_label_j[1],
                    f"{abs(load_j):.2f}",
                    fontsize=label_font_size,
                    # ha="center",
                    # va="center",
                    color=color_label,
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
    label_font_size: int = 8
    ) :

    r = radio
    if moment < 0:
        curvature: float = 1.0 if clockwise else -1.0
    else:
        curvature: float = -1.0 if clockwise else 1.0

    arrow = FancyArrowPatch(
        (x - r, y - r), (x + r, y + r),
        connectionstyle=f"arc3,rad={curvature}",
        arrowstyle="->",
        color=color,
        lw=1,
        mutation_scale=10
    )
    ax.add_patch(arrow)

    if moment > 0:
        pos_label = (x - np.cos(np.pi/4) *(1.2)* r, y + np.cos(np.pi/4) * (1.8)*r)
    else:
        pos_label = (x + np.cos(np.pi/4) *(1.8)* r, y - np.cos(np.pi/4) * (2.1)*r)
    if label:
        text = ax.text(
            pos_label[0], pos_label[1],
            f"{abs(moment)}",
            fontsize=label_font_size,
            ha="right",
            va="bottom",
            color=color_label
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
