import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import sin, cos, pi

def _create_line2d(vertices, color, lw):
    """ Crea y devuelve un Line2D a partir de una lista de vértices. """
    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    return Line2D(x, y, color=color, linewidth=lw)

def support_ttt(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    rect_vertices = [(x - a / 2, y - a), (x + a / 2, y - a),
                     (x + a / 2, y - a / 3), (x - a / 2, y - a / 3), (x - a / 2, y - a)]
    barra_vertices = [(x, y - a/3), (x, y)]
    
    rect_line = _create_line2d(rect_vertices, color, lw)
    barra_line = _create_line2d(barra_vertices, color, lw)
    
    ax.add_line(rect_line)
    ax.add_line(barra_line)
    
    return [rect_line, barra_line]

def support_ttf(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    triangle_vertices = [(x, y-a/4), (x - a / 2, y - a), (x + a / 2, y - a), (x, y-a/4)]
    barra_vertices = [(x, y - a/4), (x, y)]
    
    triangle_line = _create_line2d(triangle_vertices, color, lw)
    barra_line = _create_line2d(barra_vertices, color, lw)
    
    ax.add_line(triangle_line)
    ax.add_line(barra_line)
    
    return [triangle_line, barra_line]

def support_tft(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    horizontal_line = _create_line2d([(x, y), (x - a / 2, y)], color, lw)
    vertical_line1 = _create_line2d([(x - a / 2, y - a/2), (x - a / 2, y + a/2)], color, lw)
    vertical_line2 = _create_line2d([(x - a, y - a/2), (x - a, y + a/2)], color, lw)
    
    ax.add_line(horizontal_line)
    ax.add_line(vertical_line1)
    ax.add_line(vertical_line2)
    
    return [horizontal_line, vertical_line1, vertical_line2]

def support_ftt(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    vertical_line1 = _create_line2d([(x, y), (x, y - a / 2)], color, lw)
    horizontal_line1 = _create_line2d([(x - a / 2, y - a/2), (x + a / 2, y - a/2)], color, lw)
    horizontal_line2 = _create_line2d([(x - a/2, y - a), (x + a/2, y - a)], color, lw)
    
    ax.add_line(vertical_line1)
    ax.add_line(horizontal_line1)
    ax.add_line(horizontal_line2)
    
    return [vertical_line1, horizontal_line1, horizontal_line2]

def support_tff(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    triangle_vertices = [(x, y), (x - 3*a / 4, y - a/2), (x - 3*a / 4, y + a/2), (x, y)]
    barra_vertices = [(x-a, y - a/2), (x-a, y + a/2)]
    
    triangle_line = _create_line2d(triangle_vertices, color, lw)
    barra_line = _create_line2d(barra_vertices, color, lw)
    
    ax.add_line(triangle_line)
    ax.add_line(barra_line)
    
    return [triangle_line, barra_line]

def support_ftf(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    triangle_vertices = [(x, y), (x - a/2, y - 3*a/4), (x + a/2, y - 3*a/4), (x, y)]
    barra_vertices = [(x-a/2, y - a), (x+a/2, y - a)]
    
    triangle_line = _create_line2d(triangle_vertices, color, lw)
    barra_line = _create_line2d(barra_vertices, color, lw)
    
    ax.add_line(triangle_line)
    ax.add_line(barra_line)
    
    return [triangle_line, barra_line]

def support_fff(ax, x, y, size=0.1, color='#20dd75', lw=1):
    """ Punto sin restricciones, puede representarse vacío. """
    return []

def support_fft(ax, x, y, size=0.1, color='#20dd75', lw=1):
    a = size
    lines_vertices = [
        [(x-a/2, y), (x+a/2, y)],
        [(x, y-a/2), (x, y+a/2)],
        [(x+cos(pi/4)*a/2, y + sin(pi/4)*a/2), (x+cos(5*pi/4)*a/2, y+sin(5*pi/4)*a/2)],
        [(x + cos(3*pi/4)*a/2, y+sin(3*pi/4)*a/2), (x+cos(7*pi/4)*a/2, y +sin(7*pi/4)*a/2)]
    ]
    
    lines = []
    for line_vertices in lines_vertices:
        line = _create_line2d(line_vertices, color, lw)
        ax.add_line(line)
        lines.append(line)
    
    return lines

