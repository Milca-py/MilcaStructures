import sys
import numpy as np
import random
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class MatplotlibCanvas(QWidget):
    """Widget que integra un gráfico de matplotlib dentro de PySide6 con actualización dinámica."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Crear la figura y el lienzo (canvas)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Crear un botón para actualizar el gráfico
        self.button = QPushButton("Generar nueva función")
        self.button.clicked.connect(self.update_plot)

        # Crear un slider para modificar la frecuencia
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(20)
        self.slider.setValue(5)  # Valor inicial
        self.slider.valueChanged.connect(self.update_frequency)

        # Etiqueta para mostrar la frecuencia actual
        self.label = QLabel("Frecuencia: 5 Hz")

        # Layout para organizar el gráfico y los controles
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)   # Barra de herramientas
        layout.addWidget(self.canvas)    # Lienzo de matplotlib
        layout.addWidget(self.label)     # Etiqueta de frecuencia
        layout.addWidget(self.slider)    # Slider de frecuencia
        layout.addWidget(self.button)    # Botón de actualización
        self.setLayout(layout)

        # Datos iniciales
        self.freq = 5  # Frecuencia inicial
        self.plot_data()

    def plot_data(self):
        """Genera y dibuja la función seno."""
        self.ax.clear()  # Limpiar el gráfico
        x = np.linspace(0, 10, 100)
        y = np.sin(self.freq * x)
        self.ax.plot(x, y, label=f"Seno ({self.freq} Hz)", color="blue")
        self.ax.set_title("Gráfico Dinámico")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()
        self.canvas.draw()  # Dibujar la figura

    def update_plot(self):
        """Genera una nueva función aleatoria y actualiza el gráfico."""
        self.freq = random.randint(1, 20)  # Escoger una nueva frecuencia aleatoria
        self.label.setText(f"Frecuencia: {self.freq} Hz")
        self.slider.setValue(self.freq)  # Sincronizar el slider
        self.plot_data()

    def update_frequency(self, value):
        """Actualiza la frecuencia de la función seno en tiempo real."""
        self.freq = value
        self.label.setText(f"Frecuencia: {self.freq} Hz")
        self.plot_data()


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Actualización Dinámica del Gráfico")
        self.setGeometry(100, 100, 800, 600)

        # Agregar el widget con el gráfico y controles
        self.plot_widget = MatplotlibCanvas(self)
        self.setCentralWidget(self.plot_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
