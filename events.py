# import sys
# import numpy as np
# from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt

# class NodePlotter(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("Seleccionar Nodo")
#         self.setGeometry(100, 100, 600, 500)

#         # Widget principal
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)

#         # Layout
#         layout = QVBoxLayout(self.central_widget)

#         # Matplotlib Figure y Canvas
#         self.fig, self.ax = plt.subplots()
#         self.canvas = FigureCanvas(self.fig)
#         layout.addWidget(self.canvas)

#         # Datos de nodos (coordenadas X, Y)
#         self.nodes = np.array([[1, 2], [3, 4], [5, 1], [6, 3]])
#         self.node_colors = ['blue'] * len(self.nodes)  # Color inicial de los nodos
#         self.selected_node = None  # Nodo actualmente seleccionado

#         # Ploteo de nodos
#         self.scatter = self.ax.scatter(self.nodes[:, 0], self.nodes[:, 1], color=self.node_colors, picker=True)

#         # Anotación para mostrar información
#         self.annotation = self.ax.annotate(
#             "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
#             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="yellow"),
#             arrowprops=dict(arrowstyle="->", color="black")
#         )
#         self.annotation.set_visible(False)  # Ocultar inicialmente

#         # Conectar evento de clic
#         self.canvas.mpl_connect("button_press_event", self.on_click)

#     def on_click(self, event):
#         """ Manejo del evento de clic en el gráfico """
#         if event.inaxes is None:
#             return  # Clic fuera del gráfico

#         # Obtener coordenadas del clic
#         x_click, y_click = event.xdata, event.ydata

#         # Verificar si se hizo clic en un nodo
#         distances = np.linalg.norm(self.nodes - np.array([x_click, y_click]), axis=1)
#         min_index = np.argmin(distances)

#         if distances[min_index] < 0.5:  # Tolerancia para selección
#             # Restaurar color del nodo previamente seleccionado
#             if self.selected_node is not None:
#                 self.node_colors[self.selected_node] = 'blue'

#             # Seleccionar nuevo nodo
#             self.selected_node = min_index
#             self.node_colors[self.selected_node] = 'red'  # Cambiar color

#             # Actualizar anotación
#             self.annotation.set_text(f"Nodo {min_index}\n({self.nodes[min_index, 0]}, {self.nodes[min_index, 1]})")
#             self.annotation.xy = self.nodes[min_index]
#             self.annotation.set_visible(True)

#         else:
#             # Clic fuera de un nodo: ocultar anotación y restaurar color
#             if self.selected_node is not None:
#                 self.node_colors[self.selected_node] = 'blue'
#             self.selected_node = None
#             self.annotation.set_visible(False)

#         # Actualizar ploteo
#         self.scatter.set_color(self.node_colors)
#         self.canvas.draw_idle()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = NodePlotter()
#     window.show()
#     sys.exit(app.exec())























import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

class NodePlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Seleccionar Nodo")
        self.setGeometry(100, 100, 600, 500)

        # Widget principal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        layout = QVBoxLayout(self.central_widget)

        # Matplotlib Figure y Canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Datos de nodos (coordenadas X, Y)
        self.nodes = np.array([[1, 2], [3, 4], [5, 1], [6, 3]])
        self.node_colors = ['blue'] * len(self.nodes)  # Color inicial de los nodos
        self.selected_node = None  # Nodo actualmente seleccionado

        # Ploteo de nodos con `picker=True`
        self.scatter = self.ax.scatter(self.nodes[:, 0], self.nodes[:, 1], color=self.node_colors, picker=True)

        # Anotación para mostrar información del nodo seleccionado
        self.annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="yellow"),
            arrowprops=dict(arrowstyle="->", color="black")
        )
        self.annotation.set_visible(False)  # Ocultar inicialmente

        # Conectar el evento `pick_event`
        self.canvas.mpl_connect("pick_event", self.on_pick)

    def on_pick(self, event):
        """ Manejo del evento de selección de nodo con `pick_event` """
        artist = event.artist

        # Si se selecciona un nodo (scatter point)
        if isinstance(artist, Line2D):  # Solo aplicable si hay líneas
            return
        elif isinstance(artist, Text):  # Evita selección de texto
            return
        elif artist == None:  # Evita selección de texto
            # Clic fuera de un nodo: ocultar anotación y restaurar color
            if self.selected_node is not None:
                self.node_colors[self.selected_node] = 'blue'
            self.selected_node = None
            self.annotation.set_visible(False)

        elif artist == self.scatter:
            ind = event.ind[0]  # Índice del nodo seleccionado

            # Restaurar color del nodo previamente seleccionado
            if self.selected_node is not None:
                self.node_colors[self.selected_node] = 'blue'

            # Seleccionar nuevo nodo
            self.selected_node = ind
            self.node_colors[self.selected_node] = 'red'  # Cambiar color

            # Actualizar anotación
            self.annotation.set_text(f"Nodo {ind}\n({self.nodes[ind, 0]}, {self.nodes[ind, 1]})")
            self.annotation.xy = self.nodes[ind]
            self.annotation.set_visible(True)

        # Actualizar ploteo
        self.scatter.set_color(self.node_colors)
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NodePlotter()
    window.show()
    sys.exit(app.exec())














































































import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

class NodePlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Seleccionar Nodo")
        self.setGeometry(100, 100, 600, 500)

        # Widget principal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        layout = QVBoxLayout(self.central_widget)

        # Matplotlib Figure y Canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Datos de nodos (coordenadas X, Y)
        self.nodes = np.array([[1, 2], [3, 4], [5, 1], [6, 3]])
        self.node_colors = ['blue'] * len(self.nodes)  # Color inicial de los nodos
        self.selected_node = None  # Nodo actualmente seleccionado

        # Ploteo de nodos con `picker=True`
        self.scatter = self.ax.scatter(self.nodes[:, 0], self.nodes[:, 1], color=self.node_colors, picker=True)

        # Anotación para mostrar información del nodo seleccionado
        self.annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="yellow"),
            arrowprops=dict(arrowstyle="->", color="black")
        )
        self.annotation.set_visible(False)  # Ocultar inicialmente

        # Conectar eventos
        self.canvas.mpl_connect("pick_event", self.on_pick)  # Para seleccionar nodos
        self.canvas.mpl_connect("button_press_event", self.on_click_outside)  # Para detectar clics fuera de nodos

    def on_pick(self, event):
        """ Manejo del evento de selección de nodo con `pick_event` """
        artist = event.artist

        if isinstance(artist, Line2D) or isinstance(artist, Text):
            return
        elif artist == self.scatter:
            ind = event.ind[0]  # Índice del nodo seleccionado

            # Restaurar color del nodo previamente seleccionado
            if self.selected_node is not None:
                self.node_colors[self.selected_node] = 'blue'

            # Seleccionar nuevo nodo
            self.selected_node = ind
            self.node_colors[self.selected_node] = 'red'  # Cambiar color

            # Actualizar anotación
            self.annotation.set_text(f"Nodo {ind}\n({self.nodes[ind, 0]}, {self.nodes[ind, 1]})")
            self.annotation.xy = self.nodes[ind]
            self.annotation.set_visible(True)

            # Actualizar ploteo
            self.scatter.set_color(self.node_colors)
            self.canvas.draw_idle()

    def on_click_outside(self, event):
        """ Manejo del evento de clic fuera de los nodos """
        if event.inaxes is None:  # Clic fuera del gráfico, ignorar
            return

        # Si el clic NO fue en un nodo
        if self.selected_node is not None and not event.dblclick:
            self.node_colors[self.selected_node] = 'blue'  # Restaurar color
            self.selected_node = None
            self.annotation.set_visible(False)  # Ocultar anotación

            # Actualizar gráfico
            self.scatter.set_color(self.node_colors)
            self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NodePlotter()
    window.show()
    sys.exit(app.exec())
















































import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class NodePlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Snap en Píxeles")
        self.setGeometry(100, 100, 600, 500)

        # Widget principal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        layout = QVBoxLayout(self.central_widget)

        # Matplotlib Figure y Canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Datos de nodos (coordenadas X, Y)
        self.nodes = np.array([[1, 2], [3, 4], [5, 1], [6, 3]])
        self.node_colors = ['blue'] * len(self.nodes)  # Color inicial de los nodos
        self.selected_node = None  # Nodo actualmente seleccionado

        # Ploteo de nodos
        self.scatter = self.ax.scatter(self.nodes[:, 0], self.nodes[:, 1], color=self.node_colors)

        # Anotación para mostrar información del nodo seleccionado
        self.annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="yellow"),
            arrowprops=dict(arrowstyle="->", color="black")
        )
        self.annotation.set_visible(False)  # Ocultar inicialmente

        # Conectar eventos
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        """ Detecta el nodo más cercano en píxeles y activa el snap """
        if event.inaxes is None:  # Clic fuera del gráfico
            return

        # Convertir coordenadas de nodos a píxeles de pantalla
        node_pixels = self.ax.transData.transform(self.nodes)  # (N, 2)

        # Coordenadas del clic en píxeles
        click_pixel = np.array([event.x, event.y])

        # Calcular distancia en píxeles entre el clic y cada nodo
        distances = np.linalg.norm(node_pixels - click_pixel, axis=1)

        # Definir umbral de snap (10 píxeles)
        snap_threshold = 10
        closest_index = np.argmin(distances)
        closest_distance = distances[closest_index]

        if closest_distance <= snap_threshold:
            # Restaurar color del nodo previamente seleccionado
            if self.selected_node is not None:
                self.node_colors[self.selected_node] = 'blue'

            # Seleccionar nuevo nodo
            self.selected_node = closest_index
            self.node_colors[self.selected_node] = 'red'  # Cambiar color

            # Actualizar anotación
            self.annotation.set_text(f"Nodo {closest_index}\n({self.nodes[closest_index, 0]}, {self.nodes[closest_index, 1]})")
            self.annotation.xy = self.nodes[closest_index]
            self.annotation.set_visible(True)
        else:
            # Si no hay nodos cercanos, ocultar anotación y restaurar color
            if self.selected_node is not None:
                self.node_colors[self.selected_node] = 'blue'
            self.selected_node = None
            self.annotation.set_visible(False)

        # Actualizar ploteo
        self.scatter.set_color(self.node_colors)
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NodePlotter()
    window.show()
    sys.exit(app.exec())
