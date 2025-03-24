import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QDialog,
    QLabel, QComboBox, QSlider, QCheckBox, QHBoxLayout, QColorDialog
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class LinePlotWidget(QWidget):
    def __init__(self, n_lines=5):
        super().__init__()
        self.n_lines = n_lines
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Crear la figura de matplotlib
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Generar líneas aleatorias
        self.lines = []
        x = np.linspace(0, 10, 100)
        for i in range(self.n_lines):
            y = np.sin(x + i)
            line, = self.ax.plot(x, y, label=f"Línea {i+1}", alpha=1.0, lw=2)
            self.lines.append(line)

        self.ax.legend()
        self.canvas.draw()

        # Botón para abrir el configurador
        self.config_button = QPushButton("Configurar Líneas")
        self.config_button.clicked.connect(self.open_config_window)
        layout.addWidget(self.config_button)

    def open_config_window(self):
        config_dialog = LineConfigDialog(self)
        config_dialog.exec()


class LineConfigDialog(QDialog):
    def __init__(self, plot_widget):
        super().__init__()
        self.plot_widget = plot_widget
        self.setWindowTitle("Configurar Líneas")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Selección de línea específica o todas
        self.line_selector = QComboBox()
        self.line_selector.addItem("Todas las líneas")
        self.line_selector.addItems([f"Línea {i+1}" for i in range(len(self.plot_widget.lines))])
        layout.addWidget(QLabel("Seleccionar línea:"))
        layout.addWidget(self.line_selector)

        # Botón para cambiar color
        self.color_button = QPushButton("Cambiar Color")
        self.color_button.clicked.connect(self.change_color)
        layout.addWidget(self.color_button)

        # Slider de transparencia
        layout.addWidget(QLabel("Transparencia (0-100):"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(100)
        self.alpha_slider.valueChanged.connect(self.change_alpha)
        layout.addWidget(self.alpha_slider)

        # Slider de grosor
        layout.addWidget(QLabel("Grosor de línea:"))
        self.linewidth_slider = QSlider(Qt.Horizontal)
        self.linewidth_slider.setRange(1, 10)
        self.linewidth_slider.setValue(2)
        self.linewidth_slider.valueChanged.connect(self.change_linewidth)
        layout.addWidget(self.linewidth_slider)

        # Checkbox para visibilidad
        self.visible_checkbox = QCheckBox("Visible")
        self.visible_checkbox.setChecked(True)
        self.visible_checkbox.stateChanged.connect(self.toggle_visibility)
        layout.addWidget(self.visible_checkbox)

        # Botón para aplicar cambios
        self.apply_button = QPushButton("Aplicar")
        self.apply_button.clicked.connect(self.apply_changes)
        layout.addWidget(self.apply_button)

    def get_selected_lines(self):
        """Devuelve la(s) línea(s) seleccionada(s)"""
        index = self.line_selector.currentIndex()
        if index == 0:  # Todas las líneas
            return self.plot_widget.lines
        return [self.plot_widget.lines[index - 1]]

    def change_color(self):
        """Abre un cuadro de diálogo para seleccionar color"""
        color = QColorDialog.getColor()
        if color.isValid():
            selected_lines = self.get_selected_lines()
            for line in selected_lines:
                line.set_color(color.name())
            self.plot_widget.canvas.draw()

    def change_alpha(self):
        """Cambia la transparencia de la(s) línea(s)"""
        alpha_value = self.alpha_slider.value() / 100.0
        selected_lines = self.get_selected_lines()
        for line in selected_lines:
            line.set_alpha(alpha_value)
        self.plot_widget.canvas.draw()

    def change_linewidth(self):
        """Cambia el grosor de la(s) línea(s)"""
        lw_value = self.linewidth_slider.value()
        selected_lines = self.get_selected_lines()
        for line in selected_lines:
            line.set_linewidth(lw_value)
        self.plot_widget.canvas.draw()

    def toggle_visibility(self):
        """Cambia la visibilidad de la(s) línea(s)"""
        visible = self.visible_checkbox.isChecked()
        selected_lines = self.get_selected_lines()
        for line in selected_lines:
            line.set_visible(visible)
        self.plot_widget.canvas.draw()

    def apply_changes(self):
        """Fuerza la actualización de la gráfica"""
        self.plot_widget.ax.legend()
        self.plot_widget.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gráfica de Líneas con Configuración")
        self.setGeometry(100, 100, 800, 600)
        self.plot_widget = LinePlotWidget(n_lines=5)
        self.setCentralWidget(self.plot_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
