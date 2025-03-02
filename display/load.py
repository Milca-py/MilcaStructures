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
    angle_rotation: float = 0
) -> None:
    """Dibuja una flecha en un punto."""
    load_i, load_j = -load_i, -load_j
    a = rotate_xy(np.array([x, y]), angle_rotation, x, y)
    b = rotate_xy(np.array([x + length, y]), angle_rotation, x, y)
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

def graphic_one_arrow(
    x: float,
    y: float,
    load: float,
    angle: float, # angulo que forma la (cola - cabeza) de la carga con el eje x
    ax: "Axes",
    ratio_scale: float,
    color: str = "blue",
) -> None:
    """Dibuja una flecha en un punto."""

    a = np.array([x, y]) # cabeza del vector
    b = np.array([x + -load * ratio_scale * np.cos(-angle), y + load * ratio_scale * np.sin(-angle)])
    
    arrow = FancyArrowPatch(
        b, a,
        transform=ax.transData,
        color=color,
        linewidth=0.7,
        arrowstyle="->",
        mutation_scale=10
    )
    ax.add_patch(arrow)


def moment_fancy_arrow(
    ax: "Axes",
    x: float,
    y: float,
    moment: float,
    ratio_scale: float,
    color: str = 'blue',
    clockwise: bool = True
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
    r: float = abs(moment) * ratio_scale * 0.25
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

def moment_n_arrow(
    ax: "Axes",
    x: float,
    y: float,
    load_i: float,
    load_j: float,
    length: float,
    ratio_scale: float,
    nrof_arrows: int,
    color: str = 'blue',
    angle_rotation: float = 0,
    clockwise: bool = True,
    ) -> None:
    """
    Dibuja una carga distribuida de momento a lo largo de un elemento.
    """
    
    r: float = abs(load_i) * ratio_scale * 0.25
    R: float = abs(load_j) * ratio_scale * 0.25

    
    a = rotate_xy(np.array([x + r, y + r]), angle_rotation, x, y)
    b = rotate_xy(np.array([x + length + R, y + R]), angle_rotation, x, y)
    c = rotate_xy(np.array([x - r, y - r]), angle_rotation, x, y)
    d = rotate_xy(np.array([x + length - R, y - R]), angle_rotation, x, y)

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
