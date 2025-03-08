import numpy as np
from typing import TYPE_CHECKING
from utils import vertex_range, rotate_xy
from matplotlib.patches import FancyArrowPatch


if TYPE_CHECKING:
    from matplotlib.pyplot import Axes


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
    color_label: str = 'blue'
) -> None:
    """Dibuja una flecha en un punto."""
    load_i, load_j = -load_i, -load_j
    # coordenadas de los extremos de la barra
    a = rotate_xy(np.array([x, y]), angle_rotation, x, y)
    b = rotate_xy(np.array([x + length, y]), angle_rotation, x, y)
    
    # coordenadas de los extremos carga
    c = rotate_xy(np.array([x + load_i * ratio_scale * np.cos(angle), y + load_i * ratio_scale * np.sin(angle)]), angle_rotation, x, y)
    d = rotate_xy(np.array([x + load_j * ratio_scale * np.cos(angle) + length, y + load_j * ratio_scale * np.sin(angle)]), angle_rotation, x, y)
    
    cood_i = vertex_range(c, d, nrof_arrows)
    cood_j = vertex_range(a, b, nrof_arrows) 
    
    for start, end in zip(cood_i, cood_j):
        arrow = FancyArrowPatch(
            start, end,
            transform=ax.transData,
            color=color,
            linewidth=0.7,
            arrowstyle="->",
            mutation_scale=10
        )
        ax.add_patch(arrow)
        
    ax.plot([c[0], d[0]], [c[1], d[1]], linewidth=0.7, color=color)
    
    if label:
        if load_i == load_j:
            coord_label = (c + d) / 2
            ax.text(
                coord_label[0], coord_label[1],
                f"{abs(load_i):.2f}",
                fontsize=8,
                # ha="center",
                # va="center",
                color=color_label
            )   
        else:
            coord_label_i = c
            coord_label_j = d
            if load_i == 0:
                pass
            else:
                ax.text(
                    coord_label_i[0], coord_label_i[1],
                    f"{abs(load_i):.2f}",
                    fontsize=8,
                    # ha="center",
                    # va="center",
                    color=color_label
                )
            if load_j == 0:
                pass
            else:
                ax.text(
                    coord_label_j[0], coord_label_j[1],
                    f"{abs(load_j):.2f}",
                    fontsize=8,
                    # ha="center",
                    # va="center",
                    color=color_label
                )

def graphic_one_arrow(
    x: float,
    y: float,
    load: float,
    length_arrow: float,
    # load_mean: float,
    angle: float, # angulo que forma la (cola - cabeza) de la carga con el eje x
    ax: "Axes",
    # ratio_scale: float,
    color: str = "blue",
    label: bool = True,
    color_label: str = "black"
) -> None:
    """Dibuja una flecha en un punto."""

    if load < 0:
        angle = angle + np.pi
    load_cal = load
    # load_graf = load_mean
    a = np.array([x, y]) # cabeza del vector
    # b = np.array([x + -load_graf * ratio_scale * np.cos(-angle), y + load_graf * ratio_scale * np.sin(-angle)])
    b = np.array([x - length_arrow * np.cos(angle), y - length_arrow * np.sin(angle)])

    # coordenadas al 15% de la punta de la flecha
    coord15p = np.array([x - 0.15 * length_arrow * np.cos(angle), y - 0.15 * length_arrow * np.sin(angle)])
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
        ax.text(
            coord15p[0], coord15p[1], 
            f"{abs(load_cal):.2f}",
            fontsize=8,
            ha="right", 
            va="bottom",
            color=color_label
        )

def moment_fancy_arrow(
    ax: "Axes",
    x: float,
    y: float,
    moment: float,
    # ratio_scale: float,
    radio: float,
    color: str = 'blue',
    clockwise: bool = True,
    label: bool = True,
    color_label: str = 'blue',
    ) -> None:
    """
    Dibuja un arco representando un momento puntual usando FancyArrowPatch.
    
    Parámetros:
        ax (Axes): Eje de Matplotlib donde se dibujará el momento.
        x (float): Coordenada x del centro del momento.
        y (float): Coordenada y del centro del momento.
        moment (float): Valor del momento aplicado.
        ratio_scale (float): Factor de escala para ajustar el tamaño del arco.
        color (str, opcional): Color de la flecha. Predeterminado es 'blue'.
        clockwise (bool, opcional): Dirección de la flecha (True para horario). Predeterminado es True.
    
    Retorna:
        None: La función modifica el objeto `ax` directamente.
    """
    # r: float = abs(moment) * ratio_scale * 0.25
    r = radio
    if moment < 0:
        curvature: float = 1.0 if clockwise else -1.0
    else:
        curvature: float = -1.0 if clockwise else 1.0

    # Crear la flecha curva
    arrow = FancyArrowPatch(
        (x - r, y - r), (x + r, y + r),
        connectionstyle=f"arc3,rad={curvature}",
        arrowstyle="->",
        color=color,
        lw=1,
        mutation_scale=10
    )
    ax.add_patch(arrow)
    
    pos_label = (x - np.cos(np.pi/4) *(1.4)* r, y + np.cos(np.pi/4) * (1.4)*r)
    if label:
        ax.text(
            pos_label[0], pos_label[1],
            f"{abs(moment):.2f}",
            fontsize=8,
            ha="right",
            va="bottom",
            color=color_label
        )

def moment_n_arrow(
    ax: "Axes",
    x: float,
    y: float,
    load_i: float,
    load_j: float,
    length: float,
    radio: float,
    # ratio_scale: float,
    nrof_arrows: int,
    color: str = 'blue',
    angle_rotation: float = 0,
    clockwise: bool = True,
    label: bool = True,
    color_label: str = 'blue'
    ) -> None:
    radio = radio
    """
    Dibuja una carga distribuida de momento a lo largo de un elemento.
    """
    
    # r: float = abs(load_i) * ratio_scale * 0.25
    # R: float = abs(load_j) * ratio_scale * 0.25
    r = radio
    R = radio
    
    a = rotate_xy(np.array([x + r + (2*r), y + 0.4 * r]), angle_rotation, x, y)           # coords inicio superior
    b = rotate_xy(np.array([x + length + R + (2*R), y +  0.4 * R]), angle_rotation, x, y)  # coords final superior
    c = rotate_xy(np.array([x - r + (2*r), y -  0.4 * r]), angle_rotation, x, y)           # coords inicio inferior
    d = rotate_xy(np.array([x + length - R + (2*R), y -  0.4 * R]), angle_rotation, x, y)  # coords final inferior
    
    
    coord_label = rotate_xy(np.array([x + length/2, y + 2.4*r]), angle_rotation, x, y)  
    coord_label_i = rotate_xy(np.array([x + r, y + r]), angle_rotation, x, y)
    coord_label_j = rotate_xy(np.array([x + length + r, y + R]), angle_rotation, x, y)

    cood_i = vertex_range(c, d, nrof_arrows)
    cood_j = vertex_range(a, b, nrof_arrows)
    moment_i = np.linspace(load_i, load_j, nrof_arrows)
    
    for start, end, moment in zip(cood_i, cood_j, moment_i):
        if moment < 0:
            curvature: float = 1.0 if clockwise else -1.0
        else:
            curvature: float = -1.0 if clockwise else 1.0
        arrow = FancyArrowPatch(
            start, end,
            connectionstyle=f"arc3,rad={curvature}",
            transform=ax.transData,
            color=color,
            linewidth=0.7,
            arrowstyle="->",
            mutation_scale=10
        )
        ax.add_patch(arrow)
    
    if label:
        if load_i == load_j:
            ax.text(
                coord_label[0], coord_label[1],
                f"{abs(load_i):.2f}",
                fontsize=8,
                # ha="right",
                # va="bottom",
                color=color_label
            )
        else:
            if load_i == 0:
                pass
            else:
                ax.text(
                    coord_label_i[0], coord_label_i[1],
                    f"{abs(load_i):.2f}",
                    fontsize=8,
                    # ha="right",
                    # va="bottom",
                    color=color_label
                )
            if load_j == 0:
                pass
            else:
                ax.text(
                    coord_label_j[0], coord_label_j[1],
                    f"{abs(moment):.2f}",
                    fontsize=8,
                    # ha="right",
                    # va="bottom",
                    color=color_label
                )