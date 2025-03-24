import sys
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class MatplotlibCanvas(QWidget):
    """Widget que integra un gráfico de matplotlib dentro de PySide6."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Crear la figura y el lienzo (canvas)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        # self.toolbar = NavigationToolbar(self.canvas, self)

        # Generar datos y graficar
        self.plot_data()

        # Layout para organizar el gráfico y la barra de herramientas
        layout = QVBoxLayout()
        # layout.addWidget(self.toolbar)  # Barra de herramientas
        layout.addWidget(self.canvas)   # Lienzo de matplotlib
        self.setLayout(layout)

    def plot_data(self):
        """Genera y dibuja los datos en el gráfico."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.ax.plot(x, y, label="Seno", color="blue")
        self.ax.set_title("Gráfico de Seno")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()
        self.canvas.draw()  # Dibujar la figura


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gráfico en PySide6")
        self.setGeometry(100, 100, 800, 600)

        # Agregar el widget con el gráfico al centro de la ventana
        self.plot_widget = MatplotlibCanvas(self)
        self.setCentralWidget(self.plot_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
