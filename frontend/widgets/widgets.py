import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Dict, Optional
from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from numpy.typing import NDArray
from numpy import interp

def plot_gradient_fill(
    ax: Axes,
    x: NDArray[np.float64],
    y1: NDArray[np.float64],
    y2: Optional[NDArray[np.float64]] = None,
    cmap: str = "jet", # plasma, jet - , coolwarm, YlGnBu, rainbow - , winter, cool
    alpha: float = 0.6,
    decimals: int = 2,
) -> None:
    """
    Plotea un gradiente con un sombreado entre dos curvas.

    Parámetros:
    - ax: Eje de Matplotlib donde se realizará el gráfico.
    - x: Array con los valores de x (longitud de la viga).
    - y1: Array con los valores de la primera curva (momento flector).
    - y2: Array con los valores de la segunda curva o línea base (por defecto 0).
    - cmap: Nombre del mapa de colores para el gradiente (por defecto 'coolwarm').
    - alpha: Transparencia del sombreado (por defecto 0.6).
    - decimals: Número de decimales en los ticks de la barra de colores (por defecto 2).
    """
    y2 = np.zeros_like(x) if y2 is None else y2

    mask = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    x, y1, y2 = x[mask], y1[mask], y2[mask]

    # Construcción eficiente de vértices
    verts = np.array([
        [[x[i], y2[i]], [x[i], y1[i]], [x[i + 1], y1[i + 1]], [x[i + 1], y2[i + 1]]]
        for i in range(len(x) - 1)
    ])

    # Normalización avanzada de colores
    # norm = Normalize(vmin=y1.min(), vmax=y1.max())
    norm = Normalize(vmin=np.min(y1), vmax=np.max(y1) if np.max(y1) != np.min(y1) else np.min(y1) + 1)


    # Creación del PolyCollection optimizada
    poly = PolyCollection(
        verts, array=y1[:-1], cmap=cmap, norm=norm, edgecolor="none", alpha=alpha
    )
    ax.add_collection(poly)

    # Barra de colores optimizada
    cbar = plt.colorbar(poly, ax=ax,                # Eje al que se asocia la barra de colores.
                        orientation="vertical",     # Orientación de la barra de colores. Puede ser 'vertical' u 'horizontal'.
                        norm=norm,                  # Normalización de los colores.(puede ser Normalize o LogNorm)
                        fraction=0.05,              # Tamaño relativo de la barra de colores respecto al gráfico.
                        pad=-0.03,                   # Espaciado entre la barra de colores y el gráfico (valor negativo para que se acerque al gráfico).
                        aspect=20,                  # Relación de aspecto de la barra de colores. (ancho/alto)
                        shrink=0.9,                 # Factor de escala de la barra de colores (valor menor a 1 reduce el tamaño).
                        label="",                   # Etiqueta de la barra de colores.
                        format=f"%.{decimals}f",              # Formato de los valores en los ticks de la barra de colores (por ejemplo, "1.0f" para 1 decimal).
                        ticks = None,               # Lista de valores de los ticks de la barra de colores (si es None, se calculan automáticamente).
                        # boundaries=list(np.linspace(np.min(y1), np.max(y1), 7)),   # Valores que definen los límites de los colores en la barra.
                        # boundaries = list(np.linspace(y1.min(), y1.max(), 7)) if y1.max() > y1.min() else [y1.min()],   # Valores que definen los límites de los colores en la barra.
                        boundaries = list(np.linspace(y1.min(), y1.max(), 7)) if y1.max() > y1.min() else None,   # Valores que definen los límites de los colores en la barra.

                        extend="both",              # Controla si la barra de colores debe extenderse más allá de los límites:
                                                    # "neither" (sin extensión), "both" (extensión en ambos extremos),
                                                    # "min" (extensión solo en el mínimo), "max" (extensión solo en el máximo).
                        spacing='proportional',     # Controla el espaciado de las marcas en la barra:
                                                    # 'uniform' (espaciado uniforme) o 'proportional' (espaciado proporcional a los valores).
                        # drawedges=True,             # Si es True, dibuja bordes alrededor de las celdas de la barra de colores.
                        location = 'left',         # Ubicación de la barra de colores ('top', 'bottom', 'left', 'right'
                        )


    # Configuración avanzada de la barra de colores
    cbar.ax.tick_params(axis='y',                   # El eje al que se aplican los cambios, puede ser 'x', 'y' o 'both'
                        labelsize=8,                # Tamaño de las etiquetas de los ticks (números de los ticks)
                        labelcolor='b',             # Color de las etiquetas de los ticks (números)
                        direction='inout',          # Dirección de los ticks. Puede ser 'in', 'out', 'inout'
                        length=4,                   # Longitud de los ticks
                        width=1.5,                    # Grosor de los ticks
                        colors='r',                 # Color de los ticks en sí
                        grid_color='g',             # Color de la línea de la cuadrícula (si se dibuja)
                        grid_alpha=1,               # Transparencia de la cuadrícula (0 completamente transparente, 1 completamente opaca)
                        labelrotation=0,            # Rotación de las etiquetas de los ticks en grados
                        pad=2,                      # Distancia entre los ticks y las etiquetas
                        bottom=False,                # Mostrar u ocultar los ticks en el eje inferior (True o False)
                        top=False,                  # Mostrar u ocultar los ticks en el eje superior
                        left=False,                 # Mostrar u ocultar los ticks en el eje izquierdo
                        right=True,                # Mostrar u ocultar los ticks en el eje derecho
                        labelbottom=False,           # Mostrar u ocultar las etiquetas del eje inferior
                        labeltop=False,             # Mostrar u ocultar las etiquetas del eje superior
                        labelleft=False,             # Mostrar u ocultar las etiquetas del eje izquierdo
                        labelright=True,           # Mostrar u ocultar las etiquetas del eje derecho
    )


    # Configuración avanzada de los bordes de la barra de colores
    cbar.outline.set_edgecolor('k')       # Cambiar color del borde
    cbar.outline.set_linewidth(1)         # Cambiar grosor del borde
    cbar.outline.set_alpha(0)            # Cambiar transparencia del borde (0 transparente, 1 opaco)
    cbar.outline.set_linestyle('solid')    # Cambiar estilo del borde ('solid', 'dashed', 'dashdot', 'dotted')


@dataclass
class DiagramConfig:
    """
    Clase de configuración para un diagrama de esfuerzo interno.
    
    Esta clase almacena la configuración necesaria para crear y mostrar un diagrama
    de esfuerzo interno, incluyendo los valores, unidades y opciones de formato.

    Atributos:
        name (str): Nombre descriptivo del diagrama.
        values (NDArray[np.float_]): Array con los valores del esfuerzo interno.
        units (str): Unidades del esfuerzo interno.
        precision (float): Número de decimales para mostrar los valores. Default: 2.
        color (str): Color para el diagrama en formato matplotlib. Default: 'blue'.
        cmap (str): Mapa de colores para el gradiente. Default: 'jet'.

    Métodos:
        format_value(value: float) -> str: Formatea un valor según la precisión especificada.
    """
    name: str
    values: NDArray[np.float64]
    units: str
    precision: float = 2  # Valor por defecto
    color: str = 'blue'
    cmap: str = 'jet'

    def format_value(self, value: float) -> str:
        """
        Formatea un valor numérico según la precisión especificada.
        
        Args:
            value (float): Valor numérico a formatear.
            
        Returns:
            str: Cadena con el valor formateado con la precisión especificada.
        """
        decimals = abs(int(self.precision))
        return f"{value:.{decimals}f}"


class InternalForceDiagramWidget:
    """
    Widget interactivo para visualizar diagramas de esfuerzos internos en elementos estructurales.
    
    Esta clase crea una interfaz gráfica que permite visualizar múltiples diagramas de esfuerzos
    internos simultáneamente, con capacidades interactivas para mostrar valores específicos
    en cualquier punto a lo largo del elemento.

    Atributos:
        elem (int): Número identificador del elemento estructural.
        x (List[float]): Lista de posiciones x a lo largo del elemento.
        diagrams (Dict[str, DiagramConfig]): Diccionario de configuraciones para cada diagrama.
        L (float): Longitud total del elemento.
        grafigcalor (bool): Indica si se usa visualización con gradiente de calor.
        cmap (str): Nombre del mapa de colores para el gradiente.

    Args:
        elem (int): Número del elemento.
        x (List[float]): Array con las posiciones x.
        diagrams (Dict[str, DiagramConfig]): Diccionario con los diagramas de esfuerzos internos.
        seccion (str, optional): Sección de la viga. Default: '(35x45)'.
        grafigcalor (bool, optional): Indica si se utiliza un gráfico de calor. Default: False.
        cmap (str, optional): Mapa de colores para el gradiente. Default: 'jet'.
    """
    def __init__(self, elem: int, x: List[float], diagrams: Dict[str, DiagramConfig], seccion='(35x45)', grafigcalor = False, cmap='jet'):
        self.grafigcalor = grafigcalor
        self.elem = elem
        self.x = x
        self.diagrams = diagrams
        self.L = x[-1] #+ (self.x[1] - self.x[0])
        self.cmap = cmap

        # Inicializar diccionario para elementos interactivos
        self.interactive_elements = {}

        # Crear la ventana principal
        self.root = tk.Tk()
        self.root.geometry("800x750")
        self.root.title(
            f"Diagramas de esfuerzos internos, barra Nro {self.elem} {seccion}")

        # Crear la interfaz
        self.create_main_frame()
        self.create_grid_layout()

        # Iniciar la interfaz gráfica
        self.root.mainloop()

    def create_main_frame(self):
        """
        Crea el marco principal de la interfaz gráfica.
        
        Este método inicializa el frame principal que contendrá todos los elementos
        de la interfaz, configurándolo para expandirse y llenar el espacio disponible.
        """
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    def create_grid_layout(self):
        """
        Configura la disposición en cuadrícula de los elementos de la interfaz.
        
        Este método establece la estructura base de la interfaz, configurando las filas
        y columnas para los diagramas y sus valores asociados. La cuadrícula se ajusta
        automáticamente según el número de diagramas a mostrar.
        """
        # Configurar el grid
        num_diagrams = len(self.diagrams)
        for i in range(num_diagrams):
            self.main_frame.grid_rowconfigure(i, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=3)  # Diagrama
        self.main_frame.grid_columnconfigure(1, weight=1)  # Valores

        # Crear cada fila de diagrama y valores
        for i, (diagram_key, config) in enumerate(self.diagrams.items()):
            self.create_diagram_row(i, diagram_key, config)

    def create_diagram_row(self, row: int, diagram_key: str, config: DiagramConfig):
        """
        Crea una fila completa para un diagrama específico en la interfaz.
        
        Este método genera todos los elementos visuales necesarios para mostrar un diagrama,
        incluyendo el gráfico, las etiquetas y los elementos interactivos.

        Args:
            row (int): Número de fila en la cuadrícula donde se ubicará el diagrama.
            diagram_key (str): Clave identificadora del diagrama en el diccionario.
            config (DiagramConfig): Configuración del diagrama a crear.
        """
        # Frame para el diagrama
        diagram_frame = ttk.LabelFrame(
            self.main_frame,
            text=f"{config.name} ({config.units})"
        )
        diagram_frame.grid(row=row, column=0, sticky="nsew", padx=4, pady=4)
        diagram_frame.configure(labelwidget=tk.Label(
            diagram_frame,
            text=diagram_frame.cget("text"),
            foreground="blue",
            font=("Calibri Light", 10, "bold")
        ))

        # Crear figura y gráfico
        fig, ax = plt.subplots(figsize=(6, 1.6))
        ax.plot(self.x, config.values,
                color=config.color, linewidth=0.7, alpha=1)
        ax.plot([self.x[0], self.L], [0, 0], 'k', linewidth=0.9)
        if self.grafigcalor:
            plot_gradient_fill(ax, self.x, config.values, decimals=config.precision, cmap=self.cmap)

        else:
            ax.fill_between(self.x, config.values, 0, where=config.values >= 0,
                            color='b', alpha=0.1)
            ax.fill_between(self.x, config.values, 0, where=config.values < 0,
                            color="r", alpha=0.1)

        # Configuración del gráfico
        self.setup_plot_style(ax)
        self.adjust_plot_margins(ax)

        # Añadir texto para coordenadas xy en la esquina superior derecha
        xy_text = ax.text(0.98, 0.95, "", transform=ax.transAxes,
                          ha='right', va='top', fontsize=8,
                          bbox=dict(facecolor='#deeff5', alpha=0.4, edgecolor='none'))

        # Canvas para el gráfico
        canvas = FigureCanvasTkAgg(fig, master=diagram_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Elementos interactivos
        interactive = {
            'ax': ax,
            'canvas': canvas,

            'line': ax.axvline(x=self.x[0], color="#e34f6f", linestyle="-", alpha=1, linewidth=0.8),
            'point': ax.plot([self.x[0]], [config.values[0]], 'o', markersize=2, color='#00d068', alpha=1)[0],
            'label': ax.text(self.x[0], config.values[0], "", color="blue", fontsize=8, fontstyle='italic', fontfamily='serif'),

            'click_line': ax.axvline(x=self.x[0], color='#91a6a6', linestyle='-', alpha=0.8, linewidth=0.8),
            'click_point': ax.plot([], [], 'ro', markersize=2)[0],
            'click_label': ax.text(0, 0, "", color="k", fontsize=8, ha="left", va="bottom", fontstyle='italic', fontfamily='serif'),

            'values': config.values,
            'xy_text': xy_text  # Añadir referencia al texto xy
        }

        # Guardar referencias usando diagram_key
        self.interactive_elements[diagram_key] = interactive

        # Conectar eventos
        fig.canvas.mpl_connect("motion_notify_event",
                               lambda event: self.on_hover(event, diagram_key))
        fig.canvas.mpl_connect("button_press_event",
                               lambda event: self.on_click(event))

        # Frame para valores
        values_frame = self.create_values_frame(row, diagram_key, config)
        values_frame.grid(row=row, column=1, sticky="nsew", padx=4, pady=4)

        fig.tight_layout(pad=0)

    def create_values_frame(self, row: int, name: str, config: DiagramConfig) -> ttk.LabelFrame:
        """
        Crea un frame para mostrar los valores asociados a un diagrama.
        
        Este método genera un panel lateral que muestra valores estáticos y dinámicos
        relacionados con el diagrama.

        Args:
            row (int): Número de fila en la cuadrícula.
            name (str): Nombre del diagrama.
            config (DiagramConfig): Configuración del diagrama.

        Returns:
            ttk.LabelFrame: Frame contenedor de los valores del diagrama.
        """
        values_frame = ttk.LabelFrame(self.main_frame, text=f"Valores {name}")
        values_frame.configure(labelwidget=tk.Label(
            values_frame,
            text=values_frame.cget("text"),
            foreground="blue",
            font=("Calibri Light", 10, "bold")
        ))

        # Valores estáticos con precisión específica
        self.add_label(values_frame, f"Longitud = {self.L:.2f} m")
        self.add_label(
            values_frame, f"Max Value = {config.format_value(np.max(config.values))} {config.units}")
        self.add_label(
            values_frame, f"Min Value = {config.format_value(np.min(config.values))} {config.units}")

        # Espacio en blanco
        self.add_label(values_frame, " ")

        # Valores dinámicos (sin xy)
        dynamic_labels = {
            'value': self.add_label(values_frame, "-", dynamic=True),
            'position': self.add_label(values_frame, "-", dynamic=True),
        }

        # Guardar referencias a las etiquetas dinámicas
        self.interactive_elements[name].update({
            'labels': dynamic_labels,
            'config': config  # Guardar la configuración para acceder a la precisión
        })

        return values_frame

    def setup_plot_style(self, ax):
        """
        Configura el estilo visual del gráfico.
        
        Establece los parámetros de visualización básicos para el gráfico, como
        visibilidad de ejes, color de fondo y rejilla.

        Args:
            ax (matplotlib.axes.Axes): Eje del gráfico a configurar.
        """
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0.4)
        ax.set_facecolor('lightblue')
        ax.grid(False)

    def adjust_plot_margins(self, ax):
        """
        Ajusta los márgenes del gráfico para una mejor visualización.
        
        Calcula y aplica márgenes apropiados alrededor del gráfico, considerando
        las transformaciones entre coordenadas de píxeles y datos.

        Args:
            ax (matplotlib.axes.Axes): Eje del gráfico a ajustar.
        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        trans = ax.transData.transform
        inv_trans = ax.transData.inverted().transform

        ymin_pixel = trans((0, ymin))[1]
        ymax_pixel = trans((0, ymax))[1]
        xmin_pixel = trans((xmin, 0))[0]
        xmax_pixel = trans((xmax, 0))[0]

        ymin_pixel -= 20
        ymax_pixel += 20
        xmin_pixel -= 20
        xmax_pixel += 20

        ymin_new = inv_trans((0, ymin_pixel))[1]
        ymax_new = inv_trans((0, ymax_pixel))[1]
        xmin_new = inv_trans((xmin_pixel, 0))[0]
        xmax_new = inv_trans((xmax_pixel, 0))[0]

        ax.set_ylim(ymin_new, ymax_new)
        ax.set_xlim(xmin_new, xmax_new)

    def add_label(self, parent: ttk.LabelFrame, text: str, dynamic: bool = False) -> ttk.Label:
        """
        Añade una etiqueta al frame especificado.
        
        Crea y configura una nueva etiqueta con el texto proporcionado.

        Args:
            parent (ttk.LabelFrame): Frame contenedor de la etiqueta.
            text (str): Texto a mostrar en la etiqueta.
            dynamic (bool, optional): Indica si la etiqueta será dinámica. Default: False.

        Returns:
            ttk.Label: Referencia a la etiqueta creada si es dinámica, None en caso contrario.
        """
        label = ttk.Label(parent, text=text, font=(
            "Calibri Light", 10), foreground="black")
        label.pack(anchor="w", padx=0, pady=0)
        return label if dynamic else None

    def on_hover(self, event, current_diagram: str):
        """
        Maneja los eventos de movimiento del mouse sobre los gráficos.
        
        Actualiza la visualización de valores y elementos interactivos según la posición
        del mouse, incluso cuando está fuera del canvas de Matplotlib.

        Args:
            event: Evento de movimiento del mouse.
            current_diagram (str): Identificador del diagrama actual.
        """
        if event.inaxes is not None:
            x_val = event.xdata
            
            # Ajustar x_val al rango válido
            if x_val < self.x[0]:
                x_val = self.x[0]
            elif x_val > self.x[-1]:
                x_val = self.x[-1]

            # Actualizar todos los diagramas
            for name, elements in self.interactive_elements.items():
                # Usar interpolación para obtener el valor y
                y_val = interp(x_val, self.x, elements['values'])
                config = elements['config']

                # Actualizar elementos visuales
                elements['line'].set_xdata([x_val])
                elements['point'].set_data([x_val], [y_val])
                elements['label'].set_text(
                    f"  {config.format_value(y_val)}")
                elements['label'].set_position((x_val, y_val))

                # Actualizar texto xy en la gráfica
                elements['xy_text'].set_text(
                    f"{x_val:.2f}, {config.format_value(y_val)}")

                # Redibujar el canvas
                elements['canvas'].draw()

    def on_click(self, event):
        """
        Maneja los eventos de clic del mouse sobre los gráficos.
        
        Registra y visualiza los valores en el punto donde se realizó el clic,
        funcionando incluso cuando el clic ocurre fuera del canvas de Matplotlib.

        Args:
            event: Evento de clic del mouse.
        """
        if event.inaxes is not None:
            x_click = event.xdata
            
            # Ajustar x_click al rango válido
            if x_click < self.x[0]:
                x_click = self.x[0]
            elif x_click > self.x[-1]:
                x_click = self.x[-1]

            # Actualizar todos los diagramas
            for name, elements in self.interactive_elements.items():
                # Usar interpolación para obtener el valor y
                y_click = interp(x_click, self.x, elements['values'])
                config = elements['config']

                # Actualizar punto y línea de clic
                elements['click_point'].set_data([x_click], [y_click])
                elements['click_label'].set_text(
                    f" {config.format_value(y_click)}")
                elements['click_label'].set_position((x_click, y_click))
                elements['click_line'].set_xdata([x_click])

                # Actualizar etiquetas de valores
                elements['labels']['value'].configure(
                    text=f"{config.format_value(y_click)} {config.units}")
                elements['labels']['position'].configure(
                    text=f"at {x_click:.4f} m")

                # Redibujar el canvas
                elements['canvas'].draw()


if __name__ == "__main__":
    # Ejemplo de uso con múltiples funciones
    x = np.linspace(0, 17, 10)
    def N(x):
        return - 8 + 10*x
    def V(x):
        return - 8 + 10*x + x**2
    def M(x):
        return - 8*x + 5*x**2 + x**3
    def θ(x):
        return - 8*x + 5*x**2 + x**3 + x**4
    def y(x):
        return - 8*x + 5*x**2 + x**3 + x**4 + x**5

    diagrams = {
        'N(x)': DiagramConfig(
            name='Diagrama de Fuerza Normal',
            values=N(x),
            units='tonf',
        ),
        'V(x)': DiagramConfig(
            name='Diagrama de Fuerza Cortante',
            values=V(x),
            units='tonf',
        ),
        'M(x)': DiagramConfig(
            name='Diagrama de Momento Flector',
            values=M(x),
            units='tonf-m',
        ),
    'θ(x)': DiagramConfig(
            name='Diagrama de Rotación',
            values=θ(x),
            units='rad',
        ),
        'y(x)': DiagramConfig(
            name='Diagrama de Deflexión',
            values=y(x),
            units='cm',
        )
    }

    app = InternalForceDiagramWidget(1, x, diagrams, grafigcalor=True, cmap='rainbow')
