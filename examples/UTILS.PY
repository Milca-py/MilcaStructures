import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import numpy as np

# Crear datos
np.random.seed(0)
xs = np.random.rand(10)
ys = np.random.rand(10)
zs = np.random.rand(10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(xs, ys, zs, c='r', picker=True)

# Crear anotación oculta
annotation = ax.annotate(
    "", xy=(0, 0), xytext=(20, 20),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->")
)
annotation.set_visible(False)

# Función para actualizar anotación
def update_annotation(ind):
    x2, y2, _ = proj3d.proj_transform(xs[ind], ys[ind], zs[ind], ax.get_proj())
    annotation.xy = (x2, y2)
    text = f"x={xs[ind]:.2f}\ny={ys[ind]:.2f}\nz={zs[ind]:.2f}"
    annotation.set_text(text)
    annotation.set_visible(True)
    fig.canvas.draw_idle()

# Función para manejar el clic
def on_click(event):
    if event.inaxes != ax:
        return

    # proyectar puntos 3D a 2D
    x2d, y2d, _ = proj3d.proj_transform(xs, ys, zs, ax.get_proj())
    x2d = np.array(x2d)
    y2d = np.array(y2d)

    # calcular distancia al clic
    d = np.sqrt((x2d - event.xdata)**2 + (y2d - event.ydata)**2)
    i = np.argmin(d)

    # si el clic fue suficientemente cerca, mostrar anotación
    if d[i] < 0.05:  # umbral de sensibilidad
        update_annotation(i)
    else:
        annotation.set_visible(False)
        fig.canvas.draw_idle()

# Conectar evento
fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()
