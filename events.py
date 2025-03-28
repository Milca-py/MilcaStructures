import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class NodePlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Seleccionar Nodo")
        self.setGeometry(100, 100, 600, 500)

        # Widget principal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        layout = QVBoxLayout(self.central_widget)

        # Matplotlib Figure y Canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Etiqueta para mostrar información
        self.label = QLabel("Seleccione un nodo", self)
        layout.addWidget(self.label)
        self.label.setVisible(False)  # Inicialmente oculta

        # Datos de nodos (coordenadas X, Y)
        self.nodes = np.array([[1, 2], [3, 4], [5, 1], [6, 3]])
        self.selected_node = None  # Nodo actualmente seleccionado

        # Ploteo de nodos
        self.scatter = self.ax.scatter(self.nodes[:, 0], self.nodes[:, 1], color='blue', picker=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        """ Manejo del evento de clic en el gráfico """
        if event.inaxes is None:
            return  # Clic fuera del gráfico

        # Obtener coordenadas del clic
        x_click, y_click = event.xdata, event.ydata

        # Verificar si se hizo clic en un nodo
        distances = np.linalg.norm(self.nodes - np.array([x_click, y_click]), axis=1)
        min_index = np.argmin(distances)

        if distances[min_index] < 0.5:  # Tolerancia para selección
            self.selected_node = min_index
            self.label.setText(f"Nodo {min_index}: ({self.nodes[min_index, 0]}, {self.nodes[min_index, 1]})")
            self.label.setVisible(True)
        else:
            self.selected_node = None
            self.label.setVisible(False)  # Ocultar si se hizo clic fuera de un nodo

        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NodePlotter()
    window.show()
    sys.exit(app.exec())
