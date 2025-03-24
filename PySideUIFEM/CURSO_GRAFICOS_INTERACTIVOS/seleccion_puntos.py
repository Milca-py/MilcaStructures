import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QColorDialog, QDialog, QFormLayout, QDoubleSpinBox
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class PointEditorDialog(QDialog):
    """Diálogo para editar las propiedades de un punto seleccionado."""

    def __init__(self, x, y, color, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editar Punto")

        self.new_x = x
        self.new_y = y
        self.new_color = color

        layout = QFormLayout()

        # Selector de coordenadas
        self.x_input = QDoubleSpinBox()
        self.x_input.setValue(x)
        self.x_input.setRange(-100, 100)
        layout.addRow("X:", self.x_input)

        self.y_input = QDoubleSpinBox()
        self.y_input.setValue(y)
        self.y_input.setRange(-100, 100)
        layout.addRow("Y:", self.y_input)

        # Selector de color
        self.color_button = QPushButton("Seleccionar Color")
        self.color_button.clicked.connect(self.choose_color)
        layout.addRow("Color:", self.color_button)

        # Botón de aceptar
        self.ok_button = QPushButton("Aceptar")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def choose_color(self):
        """Abre un selector de color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.new_color = color.name()

    def accept(self):
        """Guarda los valores editados y cierra el diálogo."""
        self.new_x = self.x_input.value()
        self.new_y = self.y_input.value()
        super().accept()


class MatplotlibCanvas(QWidget):
    """Widget que integra un gráfico de matplotlib dentro de PySide6 con selección y edición de puntos."""

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

        # Datos iniciales (10 puntos aleatorios)
        self.points_x = np.random.uniform(0, 10, 10)
        self.points_y = np.random.uniform(0, 10, 10)
        self.colors = ["blue"] * len(self.points_x)  # Todos los puntos inician en azul

        self.selected_index = None  # Índice del punto seleccionado

        # Dibujar los puntos
        self.plot_data()

        # Conectar eventos de selección de puntos
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def plot_data(self):
        """Dibuja los puntos en el gráfico."""
        self.ax.clear()
        self.ax.scatter(self.points_x, self.points_y, c=self.colors, picker=True, s=100, edgecolors="black")
        self.ax.set_title("Selecciona un Punto para Editar")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.canvas.draw()

    def on_click(self, event):
        """Detecta si se hizo clic en un punto y abre el diálogo de edición."""
        if event.inaxes:
            x, y = event.xdata, event.ydata
            distances = np.sqrt((self.points_x - x) ** 2 + (self.points_y - y) ** 2)
            index = np.argmin(distances)

            # Si el punto está suficientemente cerca del clic
            if distances[index] < 0.5:
                self.selected_index = index
                self.open_edit_dialog()

    def open_edit_dialog(self):
        """Abre un cuadro de diálogo para modificar el punto seleccionado."""
        if self.selected_index is not None:
            x, y = self.points_x[self.selected_index], self.points_y[self.selected_index]
            color = self.colors[self.selected_index]

            dialog = PointEditorDialog(x, y, color, self)
            if dialog.exec():
                # Actualizar valores
                self.points_x[self.selected_index] = dialog.new_x
                self.points_y[self.selected_index] = dialog.new_y
                self.colors[self.selected_index] = dialog.new_color

                self.plot_data()  # Redibujar gráfico


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selección y Edición de Puntos con Matplotlib y PySide6")
        self.setGeometry(100, 100, 800, 600)

        # Agregar el widget con el gráfico interactivo
        self.plot_widget = MatplotlibCanvas(self)
        self.setCentralWidget(self.plot_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
