import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QComboBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class MatplotlibCanvas(QWidget):
    """Widget con gráfico de matplotlib con soporte para agregar y eliminar curvas dinámicamente."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Activar blit para mejorar rendimiento
        self.canvas.blit = True

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.lines = {}  # Diccionario {nombre: Line2D}
        self.ax.set_title("Agregar y Eliminar Curvas Dinámicamente")

        self.canvas.draw()

    def add_curve(self):
        """Agrega una nueva curva con datos aleatorios."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.uniform(-0.5, 0.5, size=len(x))

        curve_name = f"Curva {len(self.lines) + 1}"
        line, = self.ax.plot(x, y, label=curve_name)

        self.lines[curve_name] = line
        self.ax.legend()
        self.canvas.draw()

        return curve_name  # Devuelve el nombre de la curva agregada

    def remove_curve(self, curve_name):
        """Elimina una curva específica."""
        if curve_name in self.lines:
            self.lines[curve_name].remove()
            del self.lines[curve_name]

            self.ax.legend()
            self.canvas.draw()


class MainWindow(QMainWindow):
    """Ventana principal con opciones para agregar y eliminar curvas."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agregar y Eliminar Curvas con blit")
        self.setGeometry(100, 100, 800, 600)

        self.plot_widget = MatplotlibCanvas(self)
        self.setCentralWidget(self.plot_widget)

        # Crear barra de herramientas personalizada (usando QWidget)
        self.toolbar = QWidget(self)
        self.toolbar_layout = QHBoxLayout(self.toolbar)

        self.add_button = QPushButton("Agregar Curva")
        self.add_button.clicked.connect(self.add_curve)
        self.toolbar_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("Eliminar Curva")
        self.remove_button.clicked.connect(self.remove_curve)
        self.toolbar_layout.addWidget(self.remove_button)

        self.curve_selector = QComboBox()
        self.toolbar_layout.addWidget(self.curve_selector)

        self.toolbar.setLayout(self.toolbar_layout)

        # Agregar la barra de herramientas en la parte superior
        self.setMenuWidget(self.toolbar)

    def add_curve(self):
        """Maneja la adición de una nueva curva."""
        curve_name = self.plot_widget.add_curve()
        self.curve_selector.addItem(curve_name)

    def remove_curve(self):
        """Maneja la eliminación de una curva seleccionada."""
        curve_name = self.curve_selector.currentText()
        if curve_name:
            self.plot_widget.remove_curve(curve_name)
            self.curve_selector.removeItem(self.curve_selector.currentIndex())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
