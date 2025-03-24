import sys
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QComboBox, QCheckBox, QLabel
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class MatplotlibCanvas(QWidget):
    """Widget de Matplotlib con controles interactivos."""
    
    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Layout principal
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Configurar el gráfico inicial
        self.x_data = np.linspace(0, 10, 200)
        self.freq = 1.0  # Frecuencia inicial
        self.line_color = 'blue'  # Color inicial
        self.visible = True  # Visibilidad inicial

        # Dibujar línea inicial
        self.line, = self.ax.plot(self.x_data, np.sin(self.freq * self.x_data), color=self.line_color, lw=2)
        self.ax.set_title("Interactividad con PySide6 y Matplotlib")

        self.canvas.draw_idle()

    def update_plot(self):
        """Actualiza la gráfica con los nuevos parámetros."""
        if self.visible:
            self.line.set_ydata(np.sin(self.freq * self.x_data))
            self.line.set_color(self.line_color)
            self.line.set_visible(True)
        else:
            self.line.set_visible(False)

        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    """Ventana principal con controles interactivos."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactividad en Matplotlib con PySide6")
        self.setGeometry(100, 100, 900, 600)

        # Widget de la gráfica
        self.plot_widget = MatplotlibCanvas(self)
        self.setCentralWidget(self.plot_widget)

        # Contenedor de controles
        self.controls = QWidget(self)
        controls_layout = QHBoxLayout(self.controls)

        # **QSlider para modificar la frecuencia**
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setMinimum(1)
        self.freq_slider.setMaximum(10)
        self.freq_slider.setValue(1)
        self.freq_slider.valueChanged.connect(self.update_frequency)

        # **QComboBox para cambiar el color**
        self.color_selector = QComboBox()
        self.color_selector.addItems(["blue", "red", "green", "purple", "black"])
        self.color_selector.currentTextChanged.connect(self.update_color)

        # **QCheckBox para mostrar/ocultar la curva**
        self.visibility_checkbox = QCheckBox("Mostrar Línea")
        self.visibility_checkbox.setChecked(True)
        self.visibility_checkbox.stateChanged.connect(self.update_visibility)

        # Botón de Reset
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_plot)

        # Etiquetas
        self.freq_label = QLabel("Frecuencia:")
        self.color_label = QLabel("Color:")

        # Agregar controles al layout
        controls_layout.addWidget(self.freq_label)
        controls_layout.addWidget(self.freq_slider)
        controls_layout.addWidget(self.color_label)
        controls_layout.addWidget(self.color_selector)
        controls_layout.addWidget(self.visibility_checkbox)
        controls_layout.addWidget(self.reset_button)

        self.controls.setLayout(controls_layout)

        # Agregar barra de herramientas a la UI
        self.setMenuWidget(self.controls)

    def update_frequency(self):
        """Cambia la frecuencia de la función senoidal."""
        self.plot_widget.freq = self.freq_slider.value()
        self.plot_widget.update_plot()

    def update_color(self, color):
        """Cambia el color de la curva."""
        self.plot_widget.line_color = color
        self.plot_widget.update_plot()

    def update_visibility(self, state):
        """Muestra u oculta la curva."""
        self.plot_widget.visible = state == Qt.Checked
        self.plot_widget.update_plot()

    def reset_plot(self):
        """Restablece los valores predeterminados."""
        self.freq_slider.setValue(1)
        self.color_selector.setCurrentText("blue")
        self.visibility_checkbox.setChecked(True)
        self.plot_widget.freq = 1.0
        self.plot_widget.line_color = "blue"
        self.plot_widget.visible = True
        self.plot_widget.update_plot()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
