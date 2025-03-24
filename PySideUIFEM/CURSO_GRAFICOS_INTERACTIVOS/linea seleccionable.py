import sys
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QSlider, QComboBox, QCheckBox, QLabel, QListWidget
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class InteractivePlot(QWidget):
    """Widget de Matplotlib con múltiples curvas seleccionables y edición interactiva."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax.set_title("Gráfico Interactivo con Selección de Líneas y Puntos")

        self.num_curves = 5  # Número inicial de curvas
        self.x_data = np.linspace(0, 10, 100)
        self.curves = {}  # Diccionario de curvas

        # Colores disponibles
        self.colors = ['blue', 'red', 'green', 'purple', 'orange']

        # Crear curvas iniciales
        for i in range(self.num_curves):
            y_data = np.sin(self.x_data + i)
            line, = self.ax.plot(self.x_data, y_data, color=self.colors[i % len(self.colors)], lw=2, picker=5)
            self.curves[line] = {'index': i, 'color': self.colors[i % len(self.colors)], 'visible': True}

        # Conectar eventos
        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.selected_curve = None  # Almacenar la línea seleccionada
        self.selected_point = None  # Almacenar el punto seleccionado
        self.canvas.draw()

    def on_pick(self, event):
        """Maneja la selección de líneas y puntos."""
        artist = event.artist

        if isinstance(artist, plt.Line2D):  # Si es una línea
            self.selected_curve = artist
            self.highlight_curve(artist)
        elif hasattr(artist, 'get_offsets'):  # Si es un scatter (puntos)
            self.selected_point = event.ind[0]
            print(f"Punto seleccionado en índice: {self.selected_point}")

    def highlight_curve(self, curve):
        """Resalta la curva seleccionada."""
        for line in self.curves:
            line.set_linewidth(2)  # Restaurar grosor normal
        curve.set_linewidth(4)  # Resaltar curva seleccionada
        self.canvas.draw()

    def update_curve_properties(self, color=None, visible=None):
        """Cambia las propiedades de la curva seleccionada."""
        if self.selected_curve:
            if color:
                self.selected_curve.set_color(color)
                self.curves[self.selected_curve]['color'] = color
            if visible is not None:
                self.selected_curve.set_visible(visible)
                self.curves[self.selected_curve]['visible'] = visible
            self.canvas.draw()

    def add_new_curve(self):
        """Agrega una nueva curva al gráfico."""
        new_index = len(self.curves)
        y_data = np.sin(self.x_data + new_index)
        color = self.colors[new_index % len(self.colors)]
        line, = self.ax.plot(self.x_data, y_data, color=color, lw=2, picker=5)
        self.curves[line] = {'index': new_index, 'color': color, 'visible': True}
        self.canvas.draw()


class MainWindow(QMainWindow):
    """Ventana principal con controles para editar curvas."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selección de Líneas y Puntos en Matplotlib")
        self.setGeometry(100, 100, 900, 600)

        self.plot_widget = InteractivePlot(self)
        self.setCentralWidget(self.plot_widget)

        # Controles
        self.controls = QWidget(self)
        layout = QVBoxLayout(self.controls)

        # Selector de color
        self.color_selector = QComboBox()
        self.color_selector.addItems(["blue", "red", "green", "purple", "orange"])
        self.color_selector.currentTextChanged.connect(self.change_curve_color)

        # CheckBox para visibilidad
        self.visibility_checkbox = QCheckBox("Mostrar línea")
        self.visibility_checkbox.setChecked(True)
        self.visibility_checkbox.stateChanged.connect(self.toggle_curve_visibility)

        # Botón para agregar nueva curva
        self.add_curve_button = QPushButton("Agregar Curva")
        self.add_curve_button.clicked.connect(self.plot_widget.add_new_curve)

        # Lista de curvas
        self.curve_list = QListWidget()
        self.update_curve_list()

        layout.addWidget(QLabel("Selecciona Color:"))
        layout.addWidget(self.color_selector)
        layout.addWidget(self.visibility_checkbox)
        layout.addWidget(self.add_curve_button)
        layout.addWidget(QLabel("Curvas Disponibles:"))
        layout.addWidget(self.curve_list)

        self.controls.setLayout(layout)
        self.setMenuWidget(self.controls)

    def change_curve_color(self, color):
        """Cambia el color de la curva seleccionada."""
        self.plot_widget.update_curve_properties(color=color)

    def toggle_curve_visibility(self, state):
        """Muestra u oculta la curva seleccionada."""
        self.plot_widget.update_curve_properties(visible=state == Qt.Checked)

    def update_curve_list(self):
        """Actualiza la lista de curvas disponibles."""
        self.curve_list.clear()
        for line in self.plot_widget.curves:
            self.curve_list.addItem(f"Curva {self.plot_widget.curves[line]['index']}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
