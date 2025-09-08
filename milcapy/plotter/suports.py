from matplotlib.lines import Line2D
from milcapy.utils import rotate_xy
from math import sin, cos, pi
import numpy as np

lw=0.7

def draw_roller_support(ax, x, y, size=0.1, theta=0, color='#20dd75', lw=lw):
    """
    Dibuja un apoyo movil sin fijacion de rotacion.
    """
    a = size
    theta = np.deg2rad(theta)
    vertex = np.array([
        (x, y), #1
        (x, y - a / 2),#2
        (x - a / 2, y - a/2),#3
        (x + a / 2, y - a/2),#4
        (x, y - a / 2),#2
        (np.nan, np.nan),#5
        (x, y - a),#5
        (x - a/2, y - a),#5
        (x + a/2, y - a),#6
    ])
    rotated = rotate_xy(vertex, theta, x, y)
    line = Line2D(rotated[:, 0], rotated[:, 1], color=color, linewidth=lw)
    ax.add_line(line)
    return line

def draw_pinned_support(ax, x, y, size=0.1, theta=0, color='#20dd75', lw=lw):
    """
    Dibuja un apoyo fijo.
    """
    a = size
    theta = np.deg2rad(theta)
    vertex = np.array([
        (x, y-a/4),
        (x - a / 2, y - a),
        (x + a / 2, y - a),
        (x, y-a/4),
        (x, y - a/4),
        (x, y)
    ])
    rotated = rotate_xy(vertex, theta, x, y)
    line = Line2D(rotated[:, 0], rotated[:, 1], color=color, linewidth=lw)
    ax.add_line(line)
    return line

def draw_fixed_support(ax, x, y, size=0.1, theta=0, color='#20dd75', lw=lw):
    """
    Dibuja un apoyo empotrado.
    """
    a = size
    theta = np.deg2rad(theta)
    vertex = np.array([
        (x, y - a/3),
        (x - a/2, y - a/3),
        (x - a/2, y - a),
        (x + a/2, y - a),
        (x + a/2, y - a/3),
        (x, y - a/3),
        (x, y)
    ])
    rotated = rotate_xy(vertex, theta, x, y)
    line = Line2D(rotated[:, 0], rotated[:, 1], color=color, linewidth=lw)
    ax.add_line(line)
    return line

def draw_roller_support2(ax, x, y, size=0.1, theta=0, color='#20dd75', lw=lw):
    """
    Dibuja un apoyo movil con fijacion de rotacion.
    """
    a = size
    theta = np.deg2rad(theta)
    vertex = np.array([
        (x, y),
        (x - a/2, y - 3*a/4),
        (x + a/2, y - 3*a/4),
        (x, y),
        (np.nan, np.nan),
        (x-a/2, y - a),
        (x+a/2, y - a)
    ])
    rotated = rotate_xy(vertex, theta, x, y)
    line = Line2D(rotated[:, 0], rotated[:, 1], color=color, linewidth=lw)
    ax.add_line(line)
    return line





def support_ttt(ax, x, y, size=0.1, color='#20dd75', lw=lw, zorder=3, theta=0):
    """
    Dibuja un apoyo empotrado.
    """
    a = size
    vertex = np.array([
        (x, y - a/3),
        (x - a/2, y - a/3),
        (x - a/2, y - a),
        (x + a/2, y - a),
        (x + a/2, y - a/3),
        (x, y - a/3),
        (x, y)
    ])
    vertex = rotate_xy(vertex, theta, x, y)
    line = Line2D(vertex[:, 0], vertex[:, 1], color=color, linewidth=lw, zorder=zorder)
    ax.add_line(line)
    return line

def support_ttf(ax, x, y, size=0.1, color='#20dd75', lw=lw, zorder=3, theta=0):
    """
    Dibuja un apoyo fijo sobre el eje X.
    """
    a = size

    vertex = np.array([
        (x, y-a/4),
        (x - a / 2, y - a),
        (x + a / 2, y - a),
        (x, y-a/4),
        (x, y - a/4),
        (x, y)
    ])
    vertex = rotate_xy(vertex, theta, x, y)
    line = Line2D(vertex[:, 0], vertex[:, 1], color=color, linewidth=lw, zorder=zorder)
    ax.add_line(line)
    return line

def support_tft(ax, x, y, size=0.1, color='#20dd75', lw=lw, zorder=3, theta=0):
    """
    Dibuja un apoyo movil sobre el eje Y.
    """
    a = size
    vertex = np.array([
        (x, y),
        (x - a / 2, y),
        (x - a / 2, y - a/2),
        (x - a / 2, y + a/2),
        (x - a / 2, y),
        (np.nan, np.nan),
        (x - a, y),
        (x - a, y - a/2),
        (x - a, y + a/2)
    ])
    vertex = rotate_xy(vertex, theta, x, y)
    line = Line2D(vertex[:, 0], vertex[:, 1], color=color, linewidth=lw, zorder=zorder)
    ax.add_line(line)
    return line

def support_ftt(ax, x, y, size=0.1, color='#20dd75', lw=lw, zorder=3, theta=0):
    """
    Dibuja un apoyo movil sobre el eje X.
    """
    a = size
    vertex = np.array([
        (x, y), #1
        (x, y - a / 2),#2
        (x - a / 2, y - a/2),#3
        (x + a / 2, y - a/2),#4
        (x, y - a / 2),#2
        (np.nan, np.nan),#5
        (x, y - a),#5
        (x - a/2, y - a),#5
        (x + a/2, y - a),#6
    ])
    vertex = rotate_xy(vertex, theta, x, y)
    line = Line2D(vertex[:, 0], vertex[:, 1], color=color, linewidth=lw, zorder=zorder)
    ax.add_line(line)
    return line

def support_tff(ax, x, y, size=0.1, color='#20dd75', lw=lw, zorder=3, theta=0):
    """
    Dibuja un apoyo fijo sobre el eje X y movil sobre el eje Y libre a rotacion.
    """
    a = size
    vertex = np.array([
        (x, y),#1
        (x - 3*a / 4, y - a/2),#2
        (x - 3*a / 4, y + a/2),#3
        (x, y),#1
        (np.nan, np.nan),#4
        (x-a, y - a/2),#5
        (x-a, y + a/2)#6
    ])
    vertex = rotate_xy(vertex, theta, x, y)
    line = Line2D(vertex[:, 0], vertex[:, 1], color=color, linewidth=lw, zorder=zorder)
    ax.add_line(line)
    return line

def support_ftf(ax, x, y, size=0.1, color='#20dd75', lw=lw, zorder=3, theta=0):
    """
    Dibuja un apoyo movil sobre el eje X y fijo sobre el eje Y libre a rotacion.
    """
    a = size
    vertex = np.array([
        (x, y),
        (x - a/2, y - 3*a/4),
        (x + a/2, y - 3*a/4),
        (x, y),
        (np.nan, np.nan),
        (x-a/2, y - a),
        (x+a/2, y - a)
    ])
    vertex = rotate_xy(vertex, theta, x, y)
    line = Line2D(vertex[:, 0], vertex[:, 1], color=color, linewidth=lw, zorder=zorder)
    ax.add_line(line)
    return line

def support_fff(ax, x, y, size=0.1, color='#20dd75', lw=lw, zorder=3, theta=0):
    """
    Punto sin restricciones, puede representarse vacío.
    """
    return

def support_fft(ax, x, y, size=0.1, color='#20dd75', lw=lw, zorder=3, theta=0):
    """
    Dibuja un apoyo movil en el eje X e Y y fijo a rotacion.
    """
    a = size
    vertex = np.array([
        (x-a/2, y),
        (x+a/2, y),
        (x, y),
        (x, y-a/2),
        (x, y+a/2),
        (x, y),
        (x+cos(pi/4)*a/2, y + sin(pi/4)*a/2),
        (x+cos(5*pi/4)*a/2, y+sin(5*pi/4)*a/2),
        (x, y),
        (x + cos(3*pi/4)*a/2, y+sin(3*pi/4)*a/2),
        (x+cos(7*pi/4)*a/2, y +sin(7*pi/4)*a/2)
    ])
    vertex = rotate_xy(vertex, theta, x, y)
    line = Line2D(vertex[:, 0], vertex[:, 1], color=color, linewidth=lw, zorder=zorder)
    ax.add_line(line)
    return line

def support_kt(ax, x, y, size=0.1, color='#20dd75', lw=lw, zorder=3, theta=0):
    """
    Dibuja un apoyo elastico (resorte a desplazamiento).
    """
    a = size
    k1 = 1/4
    k2 = 3/16
    vertex = [
        (-a, -k1*a),
        (-a, k1*a),
        (-a, 0),
        (-3*a/4, 0),
        (-3*a/4, k2*a),
        (-a/2, -k2*a),
        (-a/2, k2*a),
        (-a/2, k2*a),
        (-a/4, -k2*a),
        (-a/4, 0),
        (0, 0)
    ]
    vertex = np.array(vertex)

    # Rotación
    rad = np.deg2rad(theta)
    rot_matrix = np.array([[np.cos(rad), -np.sin(rad)],
                           [np.sin(rad),  np.cos(rad)]])
    vertex = vertex @ rot_matrix.T

    # Traslación
    vertex[:, 0] += x
    vertex[:, 1] += y

    # Dibujar
    line = Line2D(vertex[:, 0], vertex[:, 1], color=color, linewidth=lw, zorder=zorder)
    ax.add_line(line)
    return line

def support_kr(ax, x, y, size=0.1, color='#20dd75', lw=lw, zorder=3, theta=0):
    """
    Dibuja un apoyo elastico (resorte a rotacion).
    """
    a = size
    k1 = 1/4
    n_vueltas = 2
    ri = 0.03 * a
    rf = a/2 + ri - 0.25 * a
    puntos_por_vuelta = 20

    # Espiral
    t = np.linspace(0, 2 * np.pi * n_vueltas, n_vueltas * puntos_por_vuelta)
    r = np.linspace(ri, rf, len(t))
    xs = r * np.cos(t) - a/2 - ri
    ys = r * np.sin(t)

    # Vértices iniciales
    vertex = [
        (-a, -k1*a),
        (-a,  k1*a),
        (-a,  0),
        (-a/2, 0)
    ]

    # Añadir espiral
    vertex.extend(list(zip(xs, ys)))
    vertex.append((0, 0))

    # Convertir a numpy
    vertex = np.array(vertex)

    # Rotar vertices
    rad = np.deg2rad(theta)
    rot_matrix = np.array([[np.cos(rad), -np.sin(rad)],
                           [np.sin(rad),  np.cos(rad)]])
    vertex = vertex @ rot_matrix.T

    # Trasladar al punto (x, y)
    vertex[:, 0] += x
    vertex[:, 1] += y

    # Dibujar
    line = Line2D(vertex[:, 0], vertex[:, 1], color=color, linewidth=lw, zorder=zorder)
    ax.add_line(line)
    return line
