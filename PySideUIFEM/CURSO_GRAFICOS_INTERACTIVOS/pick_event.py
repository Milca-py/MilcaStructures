import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class InteractivePlot(QWidget):
    """Widget con gráfico interactivo avanzado."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Datos iniciales
        self.points_x = np.random.uniform(0, 10, 10)
        self.points_y = np.random.uniform(0, 10, 10)
        self.selected_point = None  # Índice del punto seleccionado
        self.annotation = None  # Anotación dinámica

        # Dibujar puntos interactivos
        self.scatter = self.ax.scatter(self.points_x, self.points_y, color="red", picker=True, s=100)

        self.ax.set_title("Haz clic en un punto para ver detalles o moverlo")
        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.canvas.draw()

    def on_pick(self, event):
        """Maneja clics en los elementos del gráfico."""
        if isinstance(event.artist, plt.Line2D):
            return  # Ignorar si es una línea

        ind = event.ind[0]
        self.selected_point = ind

        x, y = self.points_x[ind], self.points_y[ind]

        # Agregar anotación dinámica
        if self.annotation:
            self.annotation.remove()

        self.annotation = self.ax.annotate(
            f"({x:.2f}, {y:.2f})", 
            (x, y),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
            color="blue",
            arrowprops=dict(arrowstyle="->", color="blue")
        )

        self.canvas.draw_idle()

    def on_motion(self, event):
        """Permite mover un punto seleccionado."""
        if self.selected_point is not None and event.inaxes:
            self.points_x[self.selected_point] = event.xdata
            self.points_y[self.selected_point] = event.ydata
            self.update_plot()

    def on_release(self, event):
        """Libera la selección cuando se suelta el clic."""
        self.selected_point = None

    def update_plot(self):
        """Redibuja los puntos y la anotación."""
        self.scatter.set_offsets(np.column_stack([self.points_x, self.points_y]))

        if self.selected_point is not None:
            x, y = self.points_x[self.selected_point], self.points_y[self.selected_point]
            self.annotation.set_position((x, y))
            self.annotation.set_text(f"({x:.2f}, {y:.2f})")

        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    """Ventana principal con gráfico interactivo."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gráficos Interactivos Avanzados")
        self.setGeometry(100, 100, 800, 600)

        self.plot_widget = InteractivePlot(self)
        self.setCentralWidget(self.plot_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
