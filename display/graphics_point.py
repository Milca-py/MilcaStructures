import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from math import cos, sin
from typing import Literal

def carga_distribuida(xi, yi, ax: plt.Axes, phi: float, direccion: Literal["X", "Y"], longitud: float = 10, w: float = -100, k: float = 0.01, color: str = "blue"):
    """
    Función para graficar una carga distribuida en el plano bajo la influencia de la gravedad.
    
    :param ax: Ejes de Matplotlib donde se dibujará la carga distribuida.
    :param phi: Ángulo de inclinación de la carga en radianes.
    :param direccion: Dirección de la carga, puede ser "X" o "Y".
    :param longitud: Longitud de la línea sobre la que actúa la carga distribuida.
    :param w: Magnitud de la carga distribuida.
    :param k: Constante de proporcionalidad.
    :param color: Color de las flechas que representan la carga distribuida.
    """
    # Validación de la dirección
    if direccion not in ["X", "Y"]:
        raise ValueError("La dirección debe ser 'X' o 'Y'")

    # Coordenadas de los nodos
    xi, yi = xi, yi
    xj, yj = longitud * cos(phi), longitud * sin(phi)

    # Coordenadas de las cargas distribuidas
    if direccion == "Y":
        xqi, yqi = 0, -w * k
        xqj, yqj = longitud * cos(phi), longitud * sin(phi) - w * k
    elif direccion == "X":
        xqi, yqi = -w * k, 0
        xqj, yqj = longitud * cos(phi) - w * k, longitud * sin(phi)

    # Interpolación de las coordenadas
    x = np.linspace(xi, xj, 10)
    y = np.linspace(yi, yj, 10)
    xq = np.linspace(xqi, xqj, 10)
    yq = np.linspace(yqi, yqj, 10)

    # Graficar la línea que representa la carga distribuida
    ax.plot([xi, xj], [yi, yj], color="#00d068")

    # Graficar la carga distribuida (flechas)
    for i in range(len(x)):
        arrow = FancyArrowPatch(
            (xq[i], yq[i]), (x[i], y[i]),
            transform=ax.transData,  # Las posiciones están en coordenadas de datos
            color=color, linewidth=0.5,
            arrowstyle="->", mutation_scale=15  # mutation_scale fija el tamaño de la flecha en píxeles
        )
        ax.add_patch(arrow)
    # cerrar los extremos de la carga distribuida
    ax.plot([xqi, xqj], [yqi, yqj], color=color, linewidth=0.5)


# Uso de la función
def main():
    fig, ax = plt.subplots(figsize=(6, 6))

    # Llamada a la función para graficar la carga distribuida
    carga_distribuida(ax, phi=3.1416/5, direccion="X", longitud=10, w=-100, k=0.01, color="blue")
    # Configuración de la gráfica
    # ax.set_aspect('equal')
    ax.set_xticks([])  # Eliminar los ticks de los ejes
    ax.set_yticks([])
    plt.tight_layout(pad=1)
    plt.axis("equal")

    # Mostrar la gráfica
    plt.show()




if __name__ == "__main__":
    main()
