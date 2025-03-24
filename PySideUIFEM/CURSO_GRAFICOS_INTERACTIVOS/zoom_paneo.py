import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MatplotlibCanvas(QWidget):
    """Widget que integra un gráfico de matplotlib dentro de PySide6 con zoom y paneo."""

    def __init__(self, parent=None, figure=None):
        super().__init__(parent)

        # Crear la figura y el lienzo (canvas)
        self.figure = figure
        self.ax = self.figure.gca()  # Obtener los ejes de la figura
        self.canvas = FigureCanvas(self.figure)

        # Layout para organizar el gráfico y la barra de herramientas
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)  # Lienzo de matplotlib
        self.setLayout(layout)

        # Datos iniciales
        self.plot_data()

        # Conectar eventos de zoom y paneo
        self.canvas.mpl_connect("scroll_event", self.on_scroll)  # Zoom
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)  # Inicio de paneo
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)  # Fin de paneo
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)  # Mover para panear

        # Variables de paneo
        self.pan_start = None
        self.pan_active = False

    def plot_data(self):
        """Genera y dibuja una función seno."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.ax.plot(x, y, label="Seno", color="blue")
        self.ax.set_title("Zoom y Paneo con Mouse")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()
        self.canvas.draw()

    def on_scroll(self, event):
        """Zoom con la rueda del mouse."""
        if event.inaxes:
            scale_factor = 1.2  # Factor de zoom
            if event.step > 0:  # Zoom in
                scale_factor = 1 / scale_factor

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            x_center = event.xdata
            y_center = event.ydata

            x_min = x_center - (x_center - xlim[0]) * scale_factor
            x_max = x_center + (xlim[1] - x_center) * scale_factor
            y_min = y_center - (y_center - ylim[0]) * scale_factor
            y_max = y_center + (ylim[1] - y_center) * scale_factor

            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.canvas.draw_idle()

    def on_mouse_press(self, event):
        """Detecta el inicio del paneo cuando se presiona el botón central del mouse."""
        if event.button == 2 and event.inaxes:  # Botón central (scroll)
            self.pan_active = True
            self.pan_start = (event.xdata, event.ydata)
            self.canvas.setCursor(Qt.ClosedHandCursor)

    def on_mouse_release(self, event):
        """Finaliza el paneo cuando se suelta el botón central del mouse."""
        if event.button == 2:
            self.pan_active = False
            self.pan_start = None
            self.canvas.setCursor(Qt.ArrowCursor)

    def on_mouse_move(self, event):
        """Realiza el paneo al mover el mouse con el botón central presionado."""
        if self.pan_active and event.inaxes:
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            self.canvas.draw_idle()


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación."""

    def __init__(self, fig):
        super().__init__()
        self.setWindowTitle("Zoom y Paneo con PySide6 y Matplotlib")
        self.setGeometry(100, 100, 800, 600)

        # Agregar el widget con el gráfico
        self.plot_widget = MatplotlibCanvas(self, fig)
        self.setCentralWidget(self.plot_widget)





if __name__ == "__main__":
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = x * np.cos(x**3)
    ax.plot(x, y, label="x cos(x³)", color="red")
    # Verificar si ya existe una instancia de QApplication
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    window = MainWindow(fig)
    window.show()
    sys.exit(app.exec())
