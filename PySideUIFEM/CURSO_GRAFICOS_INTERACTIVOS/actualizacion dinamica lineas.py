import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class MatplotlibCanvas(QWidget):
    """Widget con gráfico de matplotlib con actualización dinámica."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.points_x = np.random.uniform(0, 10, 10)
        self.points_y = np.random.uniform(0, 10, 10)

        self.lines = []
        self.labels = {}  # Diccionario de etiquetas {índice_punto: text}
        self.selected_points = []
        self.dragging_point = None  # Índice del punto en movimiento

        self.plot_data()
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def plot_data(self):
        """Dibuja los puntos, líneas y etiquetas."""
        self.ax.clear()

        # Dibujar líneas con etiquetas
        for i, j in self.lines:
            self.ax.plot(
                [self.points_x[i], self.points_x[j]],
                [self.points_y[i], self.points_y[j]],
                color="black", linewidth=2
            )

            # Calcular el punto medio de la línea
            mid_x = (self.points_x[i] + self.points_x[j]) / 2
            mid_y = (self.points_y[i] + self.points_y[j]) / 2
            length = np.linalg.norm([self.points_x[i] - self.points_x[j], self.points_y[i] - self.points_y[j]])

            # Agregar etiqueta de longitud
            self.ax.text(mid_x, mid_y, f"{length:.2f}", fontsize=10, color="blue", ha="center", va="center")

        # Dibujar puntos y etiquetas
        self.ax.scatter(self.points_x, self.points_y, color="red", picker=True, s=100, edgecolors="black")

        for idx, (x, y) in enumerate(zip(self.points_x, self.points_y)):
            self.ax.text(x, y, str(idx), fontsize=10, color="black", ha="right", va="bottom")

        self.ax.set_title("Arrastra un punto para moverlo")
        self.canvas.draw_idle()  # Actualiza de manera eficiente

    def on_click(self, event):
        """Maneja la selección de puntos para moverlos o crear líneas."""
        if event.inaxes:
            x, y = event.xdata, event.ydata
            distances = np.sqrt((self.points_x - x) ** 2 + (self.points_y - y) ** 2)
            index = np.argmin(distances)

            if distances[index] < 0.5:  # Si está cerca de un punto, permitir arrastre
                self.dragging_point = index
            else:  # Si no, seleccionar para línea
                self.select_point(index)

    def on_release(self, event):
        """Finaliza el movimiento de un punto."""
        self.dragging_point = None

    def on_motion(self, event):
        """Actualiza la posición del punto y redibuja dinámicamente."""
        if self.dragging_point is not None and event.inaxes:
            self.points_x[self.dragging_point] = event.xdata
            self.points_y[self.dragging_point] = event.ydata
            self.plot_data()  # Redibujar dinámicamente

    def select_point(self, index):
        """Maneja la selección de puntos para crear líneas."""
        if index not in self.selected_points:
            self.selected_points.append(index)

        if len(self.selected_points) == 2:
            i, j = self.selected_points
            if (i, j) not in self.lines and (j, i) not in self.lines:
                self.lines.append((i, j))  # Agregar línea
            self.selected_points = []
            self.plot_data()


class MainWindow(QMainWindow):
    """Ventana principal con gráfico interactivo."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Actualización Dinámica de Gráficos")
        self.setGeometry(100, 100, 800, 600)

        self.plot_widget = MatplotlibCanvas(self)
        self.setCentralWidget(self.plot_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
