import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class AnimatedPlot(QWidget):
    """Widget de Matplotlib con animación eficiente usando blit."""
    
    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Configuración del gráfico
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-2, 2)
        self.ax.set_title("Animación con blit en Matplotlib")

        # Crear múltiples líneas animadas
        self.num_lines = 5
        self.lines = [self.ax.plot([], [], lw=2)[0] for _ in range(self.num_lines)]

        # Datos base
        self.x_data = np.linspace(0, 10, 200)

        # Iniciar animación
        self.anim = FuncAnimation(self.figure, self.update, frames=200, 
                                  init_func=self.init_animation, interval=50, blit=True)

    def init_animation(self):
        """Inicializa las líneas vacías."""
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def update(self, frame):
        """Actualiza las líneas en cada frame."""
        for i, line in enumerate(self.lines):
            y_data = np.sin(self.x_data + frame * 0.1 + i)  # Variación en cada línea
            line.set_data(self.x_data, y_data)
        return self.lines


class MainWindow(QMainWindow):
    """Ventana principal con control para iniciar y detener la animación."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animación con blit en PySide6")
        self.setGeometry(100, 100, 800, 600)

        self.plot_widget = AnimatedPlot(self)
        self.setCentralWidget(self.plot_widget)

        # Botón para iniciar/detener animación
        self.toolbar = QWidget(self)
        layout = QVBoxLayout(self.toolbar)

        self.start_button = QPushButton("Reiniciar Animación")
        self.start_button.clicked.connect(self.restart_animation)
        layout.addWidget(self.start_button)

        self.toolbar.setLayout(layout)
        self.setMenuWidget(self.toolbar)

    def restart_animation(self):
        """Reinicia la animación."""
        self.plot_widget.anim.event_source.stop()
        self.plot_widget.anim = FuncAnimation(self.plot_widget.figure, 
                                              self.plot_widget.update, frames=200, 
                                              init_func=self.plot_widget.init_animation, 
                                              interval=50, blit=True)
        self.plot_widget.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
