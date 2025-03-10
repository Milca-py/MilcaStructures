# import tkinter as tk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# class MatplotlibCanvas(tk.Frame):
#     def __init__(self, parent, figure, *args, **kwargs):
#         tk.Frame.__init__(self, parent, *args, **kwargs)
#         self.figure = figure
        
#         # Get figure axes
#         self.ax = self.figure.axes[0] if len(self.figure.axes) > 0 else None
#         if self.ax is None:
#             raise ValueError("Figure must contain at least one axis")
        
#         # Store original limits
#         self.original_xlim = self.ax.get_xlim()
#         self.original_ylim = self.ax.get_ylim()
        
#         # Create Matplotlib canvas
#         self.canvas = FigureCanvasTkAgg(self.figure, self)
#         self.canvas.draw()
#         self.canvas_widget = self.canvas.get_tk_widget()
#         self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
#         # Pan variables
#         self._pan_start = None
#         self.pan_active = False
        
#         # Configure mouse events
#         self._connect_events()

#     def _connect_events(self):
#         self.canvas.mpl_connect('scroll_event', self._on_scroll)
#         self.canvas.mpl_connect('button_press_event', self._on_button_press)
#         self.canvas.mpl_connect('button_release_event', self._on_button_release)
#         self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
    
#     def _on_scroll(self, event):
#         if event.inaxes != self.ax:
#             return
        
#         base_scale = 1.2
#         scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        
#         x_min, x_max = self.ax.get_xlim()
#         y_min, y_max = self.ax.get_ylim()
        
#         x_center = event.xdata
#         y_center = event.ydata
        
#         x_min = x_center - (x_center - x_min) * scale_factor
#         x_max = x_center + (x_max - x_center) * scale_factor
#         y_min = y_center - (y_center - y_min) * scale_factor
#         y_max = y_center + (y_max - y_center) * scale_factor
        
#         self.ax.set_xlim(x_min, x_max)
#         self.ax.set_ylim(y_min, y_max)
#         self.canvas.draw_idle()
    
#     def _on_button_press(self, event):
#         if event.button == 2 and event.inaxes == self.ax:  # Middle button (scroll)
#             self._pan_start = (event.xdata, event.ydata)
#             self.pan_active = True
#             self.canvas_widget.config(cursor="fleur")
    
#     def _on_button_release(self, event):
#         if event.button == 2:
#             self.pan_active = False
#             self._pan_start = None
#             self.canvas_widget.config(cursor="")
    
#     def _on_mouse_move(self, event):
#         if self.pan_active and event.inaxes == self.ax and self._pan_start:
#             dx = self._pan_start[0] - event.xdata
#             dy = self._pan_start[1] - event.ydata
            
#             x_min, x_max = self.ax.get_xlim()
#             y_min, y_max = self.ax.get_ylim()
            
#             self.ax.set_xlim(x_min + dx, x_max + dx)
#             self.ax.set_ylim(y_min + dy, y_max + dy)
#             self.canvas.draw_idle()



# def create_plot_window(fig):
#     root = tk.Tk()
#     root.title("Milca System Plotter")
#     root.geometry("800x800")           # MODIFICAR EL TAMAÑO DE LA VENTANA
    
#     # Set window icon
#     try:
#         root.iconbitmap("frontend/assets/milca.ico")
#     except tk.TclError:
#         print("Warning: Could not load icon file 'milca.ico'")
    
#     frame = MatplotlibCanvas(root, fig)
#     frame.pack(fill=tk.BOTH, expand=True)
#     return root





import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MatplotlibCanvas(tk.Frame):
    def __init__(self, parent, figure, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.figure = figure
        self.axes = self.figure.axes  # Obtener todos los ejes de la figura
        
        if not self.axes:
            raise ValueError("Figure must contain at least one axis")
        
        # Almacenar los límites originales de cada eje
        self.original_limits = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in self.axes}
        
        # Crear el lienzo de Matplotlib
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Variables para el paneo
        self._pan_start = None
        self.pan_active = False
        self.active_ax = None  # Eje activo donde se mueve el mouse
        
        # Conectar eventos del mouse
        self._connect_events()

    def _connect_events(self):
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return
        
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        
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
        ax = event.inaxes
        if ax is not None and event.button == 2:  # Botón del medio (scroll)
            self._pan_start = (event.xdata, event.ydata)
            self.pan_active = True
            self.active_ax = ax
            self.canvas_widget.config(cursor="fleur")

    def _on_button_release(self, event):
        if event.button == 2:
            self.pan_active = False
            self._pan_start = None
            self.active_ax = None
            self.canvas_widget.config(cursor="")

    def _on_mouse_move(self, event):
        if self.pan_active and self.active_ax and event.inaxes == self.active_ax and self._pan_start:
            dx = self._pan_start[0] - event.xdata
            dy = self._pan_start[1] - event.ydata
            
            x_min, x_max = self.active_ax.get_xlim()
            y_min, y_max = self.active_ax.get_ylim()
            
            self.active_ax.set_xlim(x_min + dx, x_max + dx)
            self.active_ax.set_ylim(y_min + dy, y_max + dy)
            self.canvas.draw_idle()


def create_plot_window(fig):
    root = tk.Tk()
    root.title("Milca System Plotter")
    root.geometry("800x800")  # Modificar el tamaño de la ventana
    root.iconbitmap("frontend/assets/milca.ico")
    
    frame = MatplotlibCanvas(root, fig)
    frame.pack(fill=tk.BOTH, expand=True)
    return root