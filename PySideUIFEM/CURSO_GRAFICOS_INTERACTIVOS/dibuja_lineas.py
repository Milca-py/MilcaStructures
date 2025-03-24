import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QColorDialog, QDialog, QFormLayout, QDoubleSpinBox, QLabel
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class LineEditorDialog(QDialog):
    """Diálogo para editar las propiedades de una línea seleccionada."""

    def __init__(self, color, width, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editar Línea")

        self.new_color = color
        self.new_width = width

        layout = QFormLayout()

        # Selector de grosor de la línea
        self.width_input = QDoubleSpinBox()
        self.width_input.setValue(width)
        self.width_input.setRange(0.1, 10)
        layout.addRow("Grosor:", self.width_input)

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
        self.new_width = self.width_input.value()
        super().accept()


class MatplotlibCanvas(QWidget):
    """Widget que integra un gráfico de matplotlib dentro de PySide6 con selección de puntos y líneas."""

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

        self.lines = []  # Lista de líneas [(i, j, color, width)]
        self.selected_points = []  # Puntos seleccionados para crear una línea

        # Dibujar los puntos
        self.plot_data()

        # Conectar eventos de selección de puntos
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def plot_data(self):
        """Dibuja los puntos y líneas en el gráfico."""
        self.ax.clear()

        # Dibujar líneas existentes
        for i, j, color, width in self.lines:
            self.ax.plot(
                [self.points_x[i], self.points_x[j]],
                [self.points_y[i], self.points_y[j]],
                color=color,
                linewidth=width,
                picker=True
            )

        # Dibujar los puntos
        self.ax.scatter(self.points_x, self.points_y, c=self.colors, picker=True, s=100, edgecolors="black")
        self.ax.set_title("Selecciona dos Puntos para Crear una Línea")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.canvas.draw()

    def on_click(self, event):
        """Detecta si se hizo clic en un punto y maneja la selección de puntos o edición de líneas."""
        if event.inaxes:
            x, y = event.xdata, event.ydata
            distances = np.sqrt((self.points_x - x) ** 2 + (self.points_y - y) ** 2)
            index = np.argmin(distances)

            # Si el punto está suficientemente cerca del clic
            if distances[index] < 0.5:
                self.select_point(index)
            else:
                # Revisar si se hizo clic en una línea
                for k, (i, j, color, width) in enumerate(self.lines):
                    if self.is_near_line(x, y, i, j):
                        self.edit_line(k)
                        return

    def select_point(self, index):
        """Maneja la selección de dos puntos para crear una línea."""
        if index not in self.selected_points:
            self.selected_points.append(index)

        if len(self.selected_points) == 2:
            # Crear una nueva línea entre los puntos seleccionados
            i, j = self.selected_points
            self.lines.append((i, j, "black", 2))  # Línea inicial en negro, grosor 2
            self.selected_points = []  # Resetear selección
            self.plot_data()  # Redibujar gráfico

    def edit_line(self, line_index):
        """Abre un cuadro de diálogo para modificar la línea seleccionada."""
        i, j, color, width = self.lines[line_index]

        dialog = LineEditorDialog(color, width, self)
        if dialog.exec():
            # Actualizar valores de la línea
            self.lines[line_index] = (i, j, dialog.new_color, dialog.new_width)
            self.plot_data()  # Redibujar gráfico

    def is_near_line(self, x, y, i, j):
        """Determina si un punto (x, y) está cerca de la línea entre los puntos i y j."""
        x1, y1 = self.points_x[i], self.points_y[i]
        x2, y2 = self.points_x[j], self.points_y[j]

        # Calcular la distancia del punto a la línea usando geometría
        num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        return (num / den) < 0.2  # Si está cerca de la línea, retorna True


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dibujar y Editar Líneas con Matplotlib y PySide6")
        self.setGeometry(100, 100, 800, 600)

        # Agregar el widget con el gráfico interactivo
        self.plot_widget = MatplotlibCanvas(self)
        self.setCentralWidget(self.plot_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
