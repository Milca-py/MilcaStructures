import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class MatplotlibCanvas(QWidget):
    """Widget que integra un gráfico de matplotlib dentro de PySide6 con eventos del mouse."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Crear la figura y el lienzo (canvas)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layout para organizar el gráfico y la barra de herramientas
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)  # Barra de herramientas
        layout.addWidget(self.canvas)   # Lienzo de matplotlib
        self.setLayout(layout)

        # Variables para almacenar el punto clicado
        self.point = None  # Guardará el punto dibujado

        # Generar y graficar datos iniciales
        self.plot_data()

        # Conectar evento de clic en la gráfica
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def plot_data(self):
        """Genera y dibuja los datos en el gráfico."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.ax.plot(x, y, label="Seno", color="blue")
        self.ax.set_title("Interacción con Clicks")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()
        self.canvas.draw()

    def on_click(self, event):
        """Evento cuando se hace clic en el gráfico."""
        if event.inaxes:  # Solo si el clic fue dentro del gráfico
            x, y = event.xdata, event.ydata

            # Borrar el punto anterior si existe
            if self.point:
                self.point.remove()

            # Dibujar un punto rojo donde se hizo clic
            self.point = self.ax.plot(x, y, "ro", markersize=8)[0]
            self.canvas.draw()

            # Mostrar las coordenadas en la barra de estado
            self.parent().statusBar().showMessage(f"Click en: x={x:.2f}, y={y:.2f}")


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactividad con Clicks en Matplotlib")
        self.setGeometry(100, 100, 800, 600)

        # Crear barra de estado para mostrar coordenadas
        self.status = QLabel("Haz clic en el gráfico para ver coordenadas")
        self.status.setAlignment(Qt.AlignCenter)
        self.statusBar().addWidget(self.status)

        # Agregar el widget con el gráfico
        self.plot_widget = MatplotlibCanvas(self)
        self.setCentralWidget(self.plot_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
