import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MatplotlibCanvas(tk.Frame):
    def __init__(self, parent, figure, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.figure = figure
        
        # Get figure axes
        self.ax = self.figure.axes[0] if len(self.figure.axes) > 0 else None
        if self.ax is None:
            raise ValueError("Figure must contain at least one axis")
        
        # Store original limits
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        
        # Create Matplotlib canvas
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Pan variables
        self._pan_start = None
        self.pan_active = False
        
        # Configure mouse events
        self._connect_events()

    def _connect_events(self):
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
    
    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        x_center = event.xdata
        y_center = event.ydata
        
        x_min = x_center - (x_center - x_min) * scale_factor
        x_max = x_center + (x_max - x_center) * scale_factor
        y_min = y_center - (y_center - y_min) * scale_factor
        y_max = y_center + (y_max - y_center) * scale_factor
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.canvas.draw_idle()
    
    def _on_button_press(self, event):
        if event.button == 2 and event.inaxes == self.ax:  # Middle button (scroll)
            self._pan_start = (event.xdata, event.ydata)
            self.pan_active = True
            self.canvas_widget.config(cursor="fleur")
    
    def _on_button_release(self, event):
        if event.button == 2:
            self.pan_active = False
            self._pan_start = None
            self.canvas_widget.config(cursor="")
    
    def _on_mouse_move(self, event):
        if self.pan_active and event.inaxes == self.ax and self._pan_start:
            dx = self._pan_start[0] - event.xdata
            dy = self._pan_start[1] - event.ydata
            
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            
            self.ax.set_xlim(x_min + dx, x_max + dx)
            self.ax.set_ylim(y_min + dy, y_max + dy)
            self.canvas.draw_idle()



def create_plot_window(fig):
    root = tk.Tk()
    root.title("Ploting Milca System")
    root.geometry("1200x800")
    
    # Set window icon
    try:
        root.iconbitmap("frontend/assets/milca.ico")
    except tk.TclError:
        print("Warning: Could not load icon file 'milca.ico'")
    
    frame = MatplotlibCanvas(root, fig)
    frame.pack(fill=tk.BOTH, expand=True)
    return root

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create example plot
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    
    # Create and run window
    root = create_plot_window(fig)
    root.mainloop()