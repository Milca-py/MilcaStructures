from matplotlib.patches import PathPatch
from matplotlib.path import Path
from math import sin, cos, pi

def _create_path_patch(vertices, color, lw, closed=False):
    """ Crea y devuelve un PathPatch a partir de una lista de vértices. """
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
    if closed:
        codes[-1] = Path.CLOSEPOLY  # Cierra el polígono si es necesario
    path = Path(vertices, codes)
    return PathPatch(path, edgecolor=color, facecolor='none', lw=lw)

def support_ttt(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    rect_vertices = [(x - a / 2, y - a), (x + a / 2, y - a),
                     (x + a / 2, y - a / 3), (x - a / 2, y - a / 3), (x - a / 2, y - a)]
    barra_vertices = [(x, y - a/3), (x, y)]
    ax.add_patch(_create_path_patch(rect_vertices, color, lw, closed=True))
    ax.add_patch(_create_path_patch(barra_vertices, color, lw))

def support_ttf(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    triangle_vertices = [(x, y-a/4), (x - a / 2, y - a), (x + a / 2, y - a), (x, y-a/4)]
    barra_vertices = [(x, y - a/4), (x, y)]
    ax.add_patch(_create_path_patch(triangle_vertices, color, lw))
    ax.add_patch(_create_path_patch(barra_vertices, color, lw))

def support_tft(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    ax.add_patch(_create_path_patch([(x, y), (x - a / 2, y)], color, lw))
    ax.add_patch(_create_path_patch([(x - a / 2, y - a/2), (x - a / 2, y + a/2)], color, lw))
    ax.add_patch(_create_path_patch([(x - a, y - a/2), (x - a, y + a/2)], color, lw))

def support_ftt(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    ax.add_patch(_create_path_patch([(x, y), (x, y - a / 2)], color, lw))
    ax.add_patch(_create_path_patch([(x - a / 2, y - a/2), (x + a / 2, y - a/2)], color, lw))
    ax.add_patch(_create_path_patch([(x - a/2, y - a), (x + a/2, y - a)], color, lw))

def support_tff(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    triangle_vertices = [(x, y), (x - 3*a / 4, y - a/2), (x - 3*a / 4, y + a/2), (x, y)]
    barra_vertices = [(x-a, y - a/2), (x-a, y + a/2)]
    ax.add_patch(_create_path_patch(triangle_vertices, color, lw))
    ax.add_patch(_create_path_patch(barra_vertices, color, lw))

def support_ftf(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    triangle_vertices = [(x, y), (x - a/2, y - 3*a/4), (x + a/2, y - 3*a/4), (x, y)]
    barra_vertices = [(x-a/2, y - a), (x+a/2, y - a)]
    ax.add_patch(_create_path_patch(triangle_vertices, color, lw))
    ax.add_patch(_create_path_patch(barra_vertices, color, lw))

def support_fff(ax, x, y, size=0.1, color='#20dd75', lw=1):
    """ Punto sin restricciones, puede representarse vacío. """
    pass

def support_fft(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    lines = [
        [(x-a/2, y), (x+a/2, y)],
        [(x, y-a/2), (x, y+a/2)],
        [(x+cos(pi/4)*a/2, y + sin(pi/4)*a/2), (x+cos(5*pi/4)*a/2, y+sin(5*pi/4)*a/2)],
        [(x + cos(3*pi/4)*a/2, y+sin(3*pi/4)*a/2), (x+cos(7*pi/4)*a/2, y +sin(7*pi/4)*a/2)]
    ]
    for line in lines:
        ax.add_patch(_create_path_patch(line, color, lw))