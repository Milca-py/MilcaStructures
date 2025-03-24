from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class MatplotlibCanvas(QWidget):
    def __init__(self, parent, figure):
        super().__init__(parent)
        self.figure = figure
        self.axes = self.figure.axes  # Obtener todos los ejes

        if not self.axes:
            raise ValueError("Figure must contain at least one axis")

        # Almacenar límites originales de cada eje
        self.original_limits = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in self.axes}

        # Crear el lienzo de Matplotlib
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Variables de paneo
        self._pan_start = None
        self.pan_active = False
        self.active_ax = None  # Eje activo

        # Conectar eventos de ratón
        self._connect_events()

    def _connect_events(self):
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def _on_scroll(self, event):
        """Zoom con la rueda del ratón."""
        ax = event.inaxes
        if ax is None:
            return

        base_scale = 1.2
        scale_factor = 1 / base_scale if event.step > 0 else base_scale  # Qt usa 'step'

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        x_center = event.xdata
        y_center = event.ydata

        x_min = x_center - (x_center - x_min) * scale_factor
        x_max = x_center + (x_max - x_center) * scale_factor
        y_min = y_center - (y_center - y_min) * scale_factor
        y_max = y_center + (y_max - y_center) * scale_factor

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        self.canvas.draw_idle()

    def _on_button_press(self, event):
        """Activar el paneo con el botón del medio."""
        ax = event.inaxes
        if ax is not None and event.button == 2:  # Botón medio
            self._pan_start = (event.xdata, event.ydata)
            self.pan_active = True
            self.active_ax = ax
            self.setCursor(Qt.ClosedHandCursor)

    def _on_button_release(self, event):
        """Desactivar el paneo."""
        if event.button == 2:
            self.pan_active = False
            self._pan_start = None
            self.active_ax = None
            self.unsetCursor()

    def _on_mouse_move(self, event):
        """Mover el gráfico al hacer paneo."""
        if self.pan_active and self.active_ax and event.inaxes == self.active_ax and self._pan_start:
            dx = self._pan_start[0] - event.xdata
            dy = self._pan_start[1] - event.ydata

            x_min, x_max = self.active_ax.get_xlim()
            y_min, y_max = self.active_ax.get_ylim()

            self.active_ax.set_xlim(x_min + dx, x_max + dx)
            self.active_ax.set_ylim(y_min + dy, y_max + dy)
            self.canvas.draw_idle()

