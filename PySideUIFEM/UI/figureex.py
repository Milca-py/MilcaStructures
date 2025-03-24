import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

def get_figure():
    # Datos
    x_val = np.linspace(0, 10, 100)

    # Crear figura y ejes
    fig, ax = plt.subplots()

    # Graficar funciones
    ax.plot(x_val, np.cos(x_val), label="Coseno", color="#4C72B0", linestyle="--", linewidth=1.5)
    ax.plot(x_val, np.sin(x_val), label="Seno", color="#DD8452", linestyle="-.", linewidth=1.5)

    # Personalizar ejes con color rojo
    for spine in ax.spines.values():
        spine.set_color("#D62728")  # Rojo
        spine.set_linewidth(0.5)    # Grosor moderado

    # Personalizar ticks principales y secundarios
    ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.6, color="#0057B8")  # Azul
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.6, color="#00AEEF")  # Celeste

    # Agregar ticks secundarios (menores)
    ax.minorticks_on()

    # Ajustar ticks y etiquetas para mostrarse en todos los lados
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=True)  # Etiquetas arriba y abajo
    ax.yaxis.set_tick_params(labelleft=True, labelright=True)  # Etiquetas izquierda y derecha

    # Aplicar estilos a las etiquetas de los ticks sin modificar sus valores
    font = fm.FontProperties(family="isocpeur", size=10, weight="bold")
    plt.setp(ax.get_xticklabels(), fontproperties=font)
    plt.setp(ax.get_yticklabels(), fontproperties=font)

    # Cambiar el color del fondo exterior
    fig.patch.set_facecolor("lightgray")

    # Ajustar el diseño para evitar superposiciones
    plt.tight_layout(pad=0)
    plt.axis("equal")

    return fig

if __name__ == "__main__":
    fig = get_figure()
    plt.show()
