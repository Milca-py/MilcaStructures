import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import sys

class ExcelStyleArrayViewer(tk.Tk):
    def __init__(self, numpy_array=None, zero_threshold=False, round_values=False, decimal_places=4):
        super().__init__()
        self.title("Visualizador de Arrays NumPy")
        self.geometry("800x600")
        
        # Nuevos parámetros
        self.zero_threshold = zero_threshold  # Si True, valores cercanos a 0 se muestran como 0
        self.round_values = round_values      # Si True, redondear valores
        self.decimal_places = decimal_places  # Número de decimales para redondeo
        
        # Configuración de estilos
        self.style = ttk.Style()
        self.style.configure("Treeview", 
                            background="#ffffff",
                            foreground="#333333", 
                            rowheight=25,
                            font=('Segoe UI', 10))
        self.style.configure("Treeview.Heading", 
                           font=('Segoe UI', 10, 'bold'),
                           background="#e0e0e0",
                           relief="raised")
        self.style.map("Treeview", 
                     background=[('selected', '#d3e3fd')])
        
        # Colores y estilos
        self.accent_color = "#4a86e8"
        self.selected_color = "#d3e3fd"
        self.bg_color = "#f9f9f9"
        self.config(bg=self.bg_color)
        
        # Eventos globales de teclado para cargar/generar datos
        self.bind('<Control-o>', lambda e: self.load_array())
        self.bind('<Control-g>', lambda e: self.create_random_array())
        
        # Marco principal (ocupa toda la ventana)
        self.main_frame = ttk.Frame(self, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ÁREA PRINCIPAL: Visualización del array (ocupa toda la pantalla)
        self.table_container = ttk.Frame(self.main_frame)
        self.table_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars con estilo mejorado
        self.y_scrollbar = ttk.Scrollbar(self.table_container)
        self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.x_scrollbar = ttk.Scrollbar(self.table_container, orient=tk.HORIZONTAL)
        self.x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview para mostrar los datos en formato tabla estilo Excel
        self.tree = ttk.Treeview(self.table_container, 
                                 yscrollcommand=self.y_scrollbar.set,
                                 xscrollcommand=self.x_scrollbar.set,
                                 selectmode="extended",
                                 padding=2)
        
        self.y_scrollbar.config(command=self.tree.yview)
        self.x_scrollbar.config(command=self.tree.xview)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Configuración para estilos de celdas
        self.tree.tag_configure('oddrow', background="#f5f9ff")
        self.tree.tag_configure('evenrow', background="#ffffff")
        self.tree.tag_configure('selected_row', background=self.selected_color)
        self.tree.tag_configure('selected_col', background=self.selected_color)
        self.tree.tag_configure('zero_value', foreground="#888888", font=('Segoe UI', 10, 'italic'))
        
        # Información detallada en la barra de estado
        self.status_bar = ttk.Frame(self.main_frame, relief=tk.GROOVE, padding=(5, 2))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        
        # Etiquetas para estadísticas diversas
        self.info_label = ttk.Label(self.status_bar, text="No hay array cargado")
        self.info_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.stats_frame = ttk.Frame(self.status_bar)
        self.stats_frame.pack(side=tk.RIGHT)
        
        self.min_label = ttk.Label(self.stats_frame, text="Min: —")
        self.min_label.pack(side=tk.LEFT, padx=8)
        
        self.max_label = ttk.Label(self.stats_frame, text="Max: —")
        self.max_label.pack(side=tk.LEFT, padx=8)
        
        self.mean_label = ttk.Label(self.stats_frame, text="Prom: —")
        self.mean_label.pack(side=tk.LEFT, padx=8)
        
        self.std_label = ttk.Label(self.stats_frame, text="Desv: —")
        self.std_label.pack(side=tk.LEFT, padx=8)
        
        # Instructivo discreto en la parte central
        self.hint_label = ttk.Label(self.status_bar, 
                                   text="Ctrl+O: Abrir | Ctrl+G: Generar | Clic: Seleccionar",
                                   foreground="#888888")
        self.hint_label.pack(side=tk.LEFT, padx=10)
        
        # Indicadores de formatos activos
        self.format_frame = ttk.Frame(self.status_bar)
        self.format_frame.pack(side=tk.RIGHT, padx=(15, 0))
        
        format_text = ""
        if self.zero_threshold:
            format_text += "Ceros-Auto "
        if self.round_values:
            format_text += f"Redondeo-{self.decimal_places} "
        
        if format_text:
            self.format_label = ttk.Label(self.format_frame, 
                                        text=f"Formato: {format_text}", 
                                        foreground="#4a86e8")
            self.format_label.pack(side=tk.RIGHT)
        
        # Mostrar array inicial o mensaje inicial
        if numpy_array is not None:
            self.display_array(numpy_array)
        else:
            # Configuración inicial con mensaje instructivo
            self.tree["columns"] = ("mensaje",)
            self.tree.column("#0", width=80, anchor="center")
            self.tree.heading("#0", text="Índice")
            self.tree.column("mensaje", width=720, anchor="center")
            self.tree.heading("mensaje", text="Datos")
            self.tree.insert('', tk.END, text="",
                           values=("Presione Ctrl+O para cargar un array o Ctrl+G para generar uno aleatorio",))
    
    def display_array(self, array):
        # Limpiar tabla existente
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Si es 1D, convertir a 2D para mostrar uniformemente
        original_shape = array.shape
        if len(array.shape) == 1:
            array = array.reshape(-1, 1)
        
        # Actualizar información básica y estadísticas
        shape_info = f"1D: {original_shape[0]}" if len(original_shape) == 1 else f"2D: {array.shape[0]}×{array.shape[1]}"
        
        # Calcular estadísticas
        min_val = np.min(array)
        max_val = np.max(array)
        mean_val = np.mean(array)
        std_val = np.std(array)
        
        # Actualizar etiquetas de estadísticas
        self.info_label.config(text=f"Forma: {shape_info}")
        self.min_label.config(text=f"Min: {min_val:.4g}")
        self.max_label.config(text=f"Max: {max_val:.4g}")
        self.mean_label.config(text=f"Prom: {mean_val:.4g}")
        self.std_label.config(text=f"Desv: {std_val:.4g}")
        
        # Configurar columnas con índice de fila en la primera columna (estilo Excel)
        self.tree["columns"] = tuple(range(array.shape[1]))
        
        # Configurar columna de índices con mejor formato
        self.tree.column("#0", width=60, anchor="center", stretch=False)
        self.tree.heading("#0", text="#", command=lambda: self.clear_selection())
        
        # Configurar encabezados con mejor formato
        column_width = 100
        for i in range(array.shape[1]):
            self.tree.column(i, width=column_width, anchor="center")
            self.tree.heading(i, text=f"{i+1}", 
                             command=lambda col=i: self.select_column(col))
        
        # Insertar datos con filas alternadas e índices
        for i, row in enumerate(array):
            tag = 'evenrow' if i % 2 == 0 else 'oddrow'
            formatted_row = []
            cell_tags = [tag] * len(row)  # Lista para almacenar las etiquetas de cada celda
            
            for j, val in enumerate(row):
                # Aplicar formato según los parámetros
                if isinstance(val, (float, np.float16, np.float32, np.float64)):
                    # Convertir valores cercanos a cero en cero si está activado
                    if self.zero_threshold and abs(val) < 1e-10:
                        formatted_val = "0"
                        cell_tags[j] = [tag, 'zero_value']  # Añadir etiqueta para valores cero
                    elif self.round_values:
                        # Aplicar redondeo según el número de decimales especificado
                        val_rounded = round(val, self.decimal_places)
                        if abs(val_rounded) < 0.001 or abs(val_rounded) >= 10000:
                            formatted_val = f"{val_rounded:.{self.decimal_places}e}"
                        else:
                            formatted_val = f"{val_rounded:.{self.decimal_places}f}"
                    else:
                        # Formato estándar para mostrar valores
                        if abs(val) < 0.001 or abs(val) >= 10000:
                            formatted_val = f"{val:.4e}"
                        else:
                            formatted_val = f"{val:.4f}"
                else:
                    formatted_val = str(val)
                
                formatted_row.append(formatted_val)
            
            # Índice numérico en la primera columna
            item_id = self.tree.insert("", tk.END, text=f"{i+1}", values=tuple(formatted_row), tags=(tag,), iid=f"row_{i}")
            
            # Aplicar etiquetas específicas por celda si es necesario
            for j, tags in enumerate(cell_tags):
                if isinstance(tags, list) and len(tags) > 1:
                    self.tree.set_cell_tags(item_id, j, tags)
        
        # Configurar eventos para selección
        self.tree.bind("<ButtonRelease-1>", self.handle_click)
    
    def handle_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        
        # Seleccionar fila completa al hacer clic en el índice
        if region == "tree":
            selected_item = self.tree.identify_row(event.y)
            if selected_item:
                self.select_row(selected_item)
    
    def select_row(self, item_id):
        # Limpiar selecciones previas
        self.clear_selection()
        
        # Aplicar estilo de selección a la fila
        self.tree.item(item_id, tags=('selected_row',))
        
        # Obtener datos de la fila seleccionada para estadísticas
        row_values = self.tree.item(item_id, "values")
        if row_values and all(v != "" for v in row_values):
            try:
                num_values = [float(v) for v in row_values]
                row_min = min(num_values)
                row_max = max(num_values)
                row_mean = sum(num_values) / len(num_values)
                
                # Calcular desviación estándar
                variance = sum((x - row_mean) ** 2 for x in num_values) / len(num_values)
                row_std = variance ** 0.5
                
                # Actualizar etiquetas de estadísticas para esta fila
                row_num = int(item_id.split('_')[1]) + 1 if '_' in item_id else "?"
                self.info_label.config(text=f"Fila {row_num} seleccionada")
                self.min_label.config(text=f"Min: {row_min:.4g}")
                self.max_label.config(text=f"Max: {row_max:.4g}")
                self.mean_label.config(text=f"Prom: {row_mean:.4g}")
                self.std_label.config(text=f"Desv: {row_std:.4g}")
            except:
                pass
    
    def select_column(self, col_num):
        # Limpiar selecciones previas
        self.clear_selection()
        
        # Recolectar valores de columna para estadísticas
        col_values = []
        
        # Aplicar estilo de selección a todas las celdas de la columna
        for item_id in self.tree.get_children():
            current_tags = self.tree.item(item_id, "tags")
            if current_tags and 'selected_row' not in current_tags:
                base_tag = 'evenrow' if 'evenrow' in current_tags else 'oddrow'
                self.tree.item(item_id, tags=(base_tag, 'selected_col'))
            
            # Obtener valor de esta celda en la columna
            try:
                value = float(self.tree.item(item_id, "values")[col_num])
                col_values.append(value)
            except:
                pass
        
        # Calcular estadísticas de la columna seleccionada
        if col_values:
            col_min = min(col_values)
            col_max = max(col_values)
            col_mean = sum(col_values) / len(col_values)
            
            # Calcular desviación estándar
            variance = sum((x - col_mean) ** 2 for x in col_values) / len(col_values)
            col_std = variance ** 0.5
            
            # Actualizar etiquetas de estadísticas para esta columna
            self.info_label.config(text=f"Columna {col_num} seleccionada")
            self.min_label.config(text=f"Min: {col_min:.4g}")
            self.max_label.config(text=f"Max: {col_max:.4g}")
            self.mean_label.config(text=f"Prom: {col_mean:.4g}")
            self.std_label.config(text=f"Desv: {col_std:.4g}")
    
    def clear_selection(self):
        # Restaurar estilos originales
        for item_id in self.tree.get_children():
            row_num = int(item_id.split('_')[1]) if '_' in item_id else 0
            tag = 'evenrow' if row_num % 2 == 0 else 'oddrow'
            self.tree.item(item_id, tags=(tag,))
    
    def load_array(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo NumPy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                array = np.load(file_path)
                # Manejar arrays de diferentes dimensiones
                if len(array.shape) > 2:
                    messagebox.showwarning("Aviso", "Los arrays con más de 2 dimensiones se aplanarán a 2D.")
                    array = array.reshape(-1, array.shape[-1] if len(array.shape) > 1 else 1)
                elif len(array.shape) == 0:
                    array = np.array([array])
                
                # Abrir ventana de configuración para los parámetros de visualización
                self.show_display_options(array)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo: {str(e)}")
    
    def show_display_options(self, array):
        """Muestra una ventana para configurar las opciones de visualización"""
        options_window = tk.Toplevel(self)
        options_window.title("Opciones de Visualización")
        options_window.geometry("350x200")
        options_window.resizable(False, False)
        options_window.transient(self)
        options_window.grab_set()
        
        frame = ttk.Frame(options_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Opciones de visualización
        zero_threshold_var = tk.BooleanVar(value=self.zero_threshold)
        ttk.Checkbutton(frame, text="Convertir valores cercanos a cero en cero", 
                      variable=zero_threshold_var).grid(row=0, column=0, columnspan=2, 
                                                      padx=5, pady=5, sticky=tk.W)
        
        round_values_var = tk.BooleanVar(value=self.round_values)
        round_check = ttk.Checkbutton(frame, text="Redondear valores", 
                                    variable=round_values_var)
        round_check.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(frame, text="Decimales:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        decimal_places_var = tk.IntVar(value=self.decimal_places)
        decimal_places_entry = ttk.Spinbox(frame, from_=0, to=10, width=5, 
                                         textvariable=decimal_places_var)
        decimal_places_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Función para actualizar el estado de entrada de decimales
        def update_decimal_entry_state():
            decimal_places_entry.config(state="normal" if round_values_var.get() else "disabled")
        
        # Configurar la función de actualización
        round_check.config(command=update_decimal_entry_state)
        # Llamar a la función para establecer el estado inicial
        update_decimal_entry_state()
        
        def apply_options():
            self.zero_threshold = zero_threshold_var.get()
            self.round_values = round_values_var.get()
            self.decimal_places = decimal_places_var.get()
            
            # Actualizar el indicador de formato en la barra de estado
            format_text = ""
            if self.zero_threshold:
                format_text += "Ceros-Auto "
            if self.round_values:
                format_text += f"Redondeo-{self.decimal_places} "
            
            if hasattr(self, 'format_label'):
                if format_text:
                    self.format_label.config(text=f"Formato: {format_text}")
                    if not self.format_label.winfo_ismapped():
                        self.format_label.pack(side=tk.RIGHT)
                else:
                    self.format_label.pack_forget()
            elif format_text:
                self.format_label = ttk.Label(self.format_frame, 
                                            text=f"Formato: {format_text}", 
                                            foreground="#4a86e8")
                self.format_label.pack(side=tk.RIGHT)
            
            self.display_array(array)
            options_window.destroy()
        
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Cancelar", command=options_window.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Aplicar", command=apply_options).pack(side=tk.LEFT, padx=5)
    
    def create_random_array(self):
        # Ventana simple de configuración
        config_window = tk.Toplevel(self)
        config_window.title("Generar Array")
        config_window.geometry("300x250")  # Aumentado para acomodar nuevas opciones
        config_window.resizable(False, False)
        config_window.transient(self)
        config_window.grab_set()
        
        frame = ttk.Frame(config_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Parámetros básicos
        ttk.Label(frame, text="Filas:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        rows_entry = ttk.Entry(frame, width=8)
        rows_entry.grid(row=0, column=1, padx=5, pady=5)
        rows_entry.insert(0, "10")
        
        ttk.Label(frame, text="Columnas:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        cols_entry = ttk.Entry(frame, width=8)
        cols_entry.grid(row=1, column=1, padx=5, pady=5)
        cols_entry.insert(0, "5")
        
        ttk.Label(frame, text="Tipo:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        data_type = tk.StringVar(value="float")
        type_combo = ttk.Combobox(frame, textvariable=data_type, width=8, 
                                 values=["float", "int"], state="readonly")
        type_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # Nuevas opciones de visualización
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        
        zero_threshold_var = tk.BooleanVar(value=self.zero_threshold)
        ttk.Checkbutton(frame, text="Convertir valores cercanos a cero en cero", 
                       variable=zero_threshold_var).grid(row=4, column=0, columnspan=2, 
                                                       padx=5, pady=5, sticky=tk.W)
        
        round_values_var = tk.BooleanVar(value=self.round_values)
        round_check = ttk.Checkbutton(frame, text="Redondear valores", 
                                     variable=round_values_var)
        round_check.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(frame, text="Decimales:").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        decimal_places_var = tk.IntVar(value=self.decimal_places)
        decimal_places_entry = ttk.Spinbox(frame, from_=0, to=10, width=5, 
                                          textvariable=decimal_places_var)
        decimal_places_entry.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Función para actualizar el estado de entrada de decimales
        def update_decimal_entry_state():
            decimal_places_entry.config(state="normal" if round_values_var.get() else "disabled")
        
        # Configurar la función de actualización
        round_check.config(command=update_decimal_entry_state)
        # Llamar a la función para establecer el estado inicial
        update_decimal_entry_state()
        
        def generate():
            try:
                rows = int(rows_entry.get())
                cols = int(cols_entry.get())
                
                if rows <= 0 or cols <= 0:
                    raise ValueError("Las dimensiones deben ser positivas")
                
                if data_type.get() == "float":
                    array = np.random.rand(rows, cols) * 100 - 50  # Valores entre -50 y 50
                else:
                    array = np.random.randint(-50, 50, (rows, cols))
                
                # Actualizar parámetros de visualización
                self.zero_threshold = zero_threshold_var.get()
                self.round_values = round_values_var.get()
                self.decimal_places = decimal_places_var.get()
                
                # Actualizar el indicador de formato en la barra de estado
                format_text = ""
                if self.zero_threshold:
                    format_text += "Ceros-Auto "
                if self.round_values:
                    format_text += f"Redondeo-{self.decimal_places} "
                
                if hasattr(self, 'format_label'):
                    if format_text:
                        self.format_label.config(text=f"Formato: {format_text}")
                        if not self.format_label.winfo_ismapped():
                            self.format_label.pack(side=tk.RIGHT)
                    else:
                        self.format_label.pack_forget()
                elif format_text:
                    self.format_label = ttk.Label(self.format_frame, 
                                                text=f"Formato: {format_text}", 
                                                foreground="#4a86e8")
                    self.format_label.pack(side=tk.RIGHT)
                
                self.display_array(array)
                config_window.destroy()
            except ValueError as e:
                messagebox.showerror("Error", str(e))
        
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Cancelar", command=config_window.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Generar", command=generate).pack(side=tk.LEFT, padx=5)

# Añadir método para establecer etiquetas por celda (tkinter no lo soporta nativamente)
def set_cell_tags(self, item_id, column, tags):
    """Método para simular etiquetas por celda en Treeview"""
    pass

# Añadir el método al treeview
ttk.Treeview.set_cell_tags = set_cell_tags

def mostrar_array(array, zero_threshold=False, round_values=False, decimal_places=4):
    """
    Muestra un array NumPy en una interfaz gráfica
    
    Parámetros:
    - array: El array NumPy a visualizar
    - zero_threshold: Si True, valores cercanos a cero se muestran como 0
    - round_values: Si True, redondea los valores según decimal_places
    - decimal_places: Número de decimales para redondeo (solo aplica si round_values=True)
    """
    app = ExcelStyleArrayViewer(array, zero_threshold, round_values, decimal_places)
    app.mainloop()