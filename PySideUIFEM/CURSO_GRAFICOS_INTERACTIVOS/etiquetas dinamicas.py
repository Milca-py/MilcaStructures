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
    """Diálogo para editar propiedades de una línea."""

    def __init__(self, color, width, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editar Línea")

        self.new_color = color
        self.new_width = width

        layout = QFormLayout()

        self.width_input = QDoubleSpinBox()
        self.width_input.setValue(width)
        self.width_input.setRange(0.1, 10)
        layout.addRow("Grosor:", self.width_input)

        self.color_button = QPushButton("Seleccionar Color")
        self.color_button.clicked.connect(self.choose_color)
        layout.addRow("Color:", self.color_button)

        self.ok_button = QPushButton("Aceptar")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def choose_color(self):
        """Seleccionar color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.new_color = color.name()

    def accept(self):
        """Guardar cambios."""
        self.new_width = self.width_input.value()
        super().accept()


class MatplotlibCanvas(QWidget):
    """Widget con gráfico de matplotlib y etiquetas en puntos y líneas."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.points_x = np.random.uniform(0, 10, 10)
        self.points_y = np.random.uniform(0, 10, 10)

        self.lines = []
        self.labels = {}  # Diccionario de etiquetas {índice_punto: text}
        self.selected_points = []

        self.plot_data()
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def plot_data(self):
        """Dibuja los puntos, líneas y etiquetas."""
        self.ax.clear()

        # Dibujar líneas con etiquetas
        for i, j, color, width in self.lines:
            line = self.ax.plot(
                [self.points_x[i], self.points_x[j]],
                [self.points_y[i], self.points_y[j]],
                color=color, linewidth=width, picker=True
            )

            # Calcular el punto medio de la línea
            mid_x = (self.points_x[i] + self.points_x[j]) / 2
            mid_y = (self.points_y[i] + self.points_y[j]) / 2
            length = np.linalg.norm([self.points_x[i] - self.points_x[j], self.points_y[i] - self.points_y[j]])

            # Agregar etiqueta de longitud
            self.ax.text(mid_x, mid_y, f"{length:.2f}", fontsize=10, color=color, ha="center", va="center")

        # Dibujar puntos y etiquetas
        self.ax.scatter(self.points_x, self.points_y, color="blue", picker=True, s=100, edgecolors="black")

        for idx, (x, y) in enumerate(zip(self.points_x, self.points_y)):
            self.ax.text(x, y, str(idx), fontsize=10, color="red", ha="right", va="bottom")

        self.ax.set_title("Selecciona dos Puntos para Crear una Línea")
        self.canvas.draw()

    def on_click(self, event):
        """Maneja selección de puntos y edición de líneas."""
        if event.inaxes:
            x, y = event.xdata, event.ydata
            distances = np.sqrt((self.points_x - x) ** 2 + (self.points_y - y) ** 2)
            index = np.argmin(distances)

            if distances[index] < 0.5:
                self.select_point(index)
            else:
                for k, (i, j, color, width) in enumerate(self.lines):
                    if self.is_near_line(x, y, i, j):
                        self.edit_line(k)
                        return

    def select_point(self, index):
        """Maneja la selección de puntos para crear líneas."""
        if index not in self.selected_points:
            self.selected_points.append(index)

        if len(self.selected_points) == 2:
            i, j = self.selected_points
            self.lines.append((i, j, "black", 2))  # Línea inicial en negro
            self.selected_points = []
            self.plot_data()

    def edit_line(self, line_index):
        """Abre diálogo para editar una línea."""
        i, j, color, width = self.lines[line_index]

        dialog = LineEditorDialog(color, width, self)
        if dialog.exec():
            self.lines[line_index] = (i, j, dialog.new_color, dialog.new_width)
            self.plot_data()

    def is_near_line(self, x, y, i, j):
        """Determina si un punto (x, y) está cerca de la línea entre los puntos i y j."""
        x1, y1 = self.points_x[i], self.points_y[i]
        x2, y2 = self.points_x[j], self.points_y[j]

        num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        return (num / den) < 0.2  # Si está cerca de la línea


class MainWindow(QMainWindow):
    """Ventana principal con gráfico interactivo."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dibujar y Editar Líneas con Etiquetas")
        self.setGeometry(100, 100, 800, 600)

        self.plot_widget = MatplotlibCanvas(self)
        self.setCentralWidget(self.plot_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
