from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QWidget, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QComboBox, QColorDialog, QLineEdit
)
import sys

class PlotterOptionsDialog(QDialog):
    """Ventana emergente para configurar opciones de gráficos con pestañas."""
    
    def __init__(self, plotter_options):
        super().__init__()
        self.setWindowTitle("Opciones de Gráfico")
        self.setMinimumSize(400, 300)

        self.plotter_options = plotter_options  # Referencia a opciones actuales
        
        layout = QVBoxLayout(self)
        
        # Tabs
        self.tabs = QTabWidget()
        self.general_tab = self.create_general_tab()
        self.visualization_tab = self.create_visualization_tab()
        self.results_tab = self.create_results_tab()

        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.visualization_tab, "Estructura")
        self.tabs.addTab(self.results_tab, "Resultados")

        layout.addWidget(self.tabs)

        # Botones
        btn_layout = QHBoxLayout()
        self.apply_button = QPushButton("Aplicar")
        self.cancel_button = QPushButton("Cancelar")
        
        self.apply_button.clicked.connect(self.apply_changes)
        self.cancel_button.clicked.connect(self.reject)

        btn_layout.addWidget(self.apply_button)
        btn_layout.addWidget(self.cancel_button)
        layout.addLayout(btn_layout)

    def create_general_tab(self):
        """Pestaña para opciones generales."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Tamaño de figura
        layout.addWidget(QLabel("Tamaño de figura:"))
        self.figure_size = QLineEdit(f"{self.plotter_options.figure_size}")
        layout.addWidget(self.figure_size)

        # DPI
        layout.addWidget(QLabel("DPI:"))
        self.dpi = QSpinBox()
        self.dpi.setRange(50, 500)
        self.dpi.setValue(self.plotter_options.dpi)
        layout.addWidget(self.dpi)

        # Color de fondo
        self.bg_color_button = QPushButton("Seleccionar color de fondo")
        self.bg_color_button.clicked.connect(self.choose_bg_color)
        layout.addWidget(self.bg_color_button)

        # Mostrar Grid
        self.grid_checkbox = QCheckBox("Mostrar cuadrícula")
        self.grid_checkbox.setChecked(self.plotter_options.grid)
        layout.addWidget(self.grid_checkbox)

        tab.setLayout(layout)
        return tab

    def create_visualization_tab(self):
        """Pestaña para opciones de visualización de la estructura."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Tamaño de nodos
        layout.addWidget(QLabel("Tamaño de nodos:"))
        self.node_size = QSpinBox()
        self.node_size.setRange(1, 20)
        self.node_size.setValue(self.plotter_options.node_size)
        layout.addWidget(self.node_size)

        # Color de nodos
        self.node_color_button = QPushButton("Seleccionar color de nodo")
        self.node_color_button.clicked.connect(self.choose_node_color)
        layout.addWidget(self.node_color_button)

        # Color de elementos
        self.element_color_button = QPushButton("Seleccionar color de elementos")
        self.element_color_button.clicked.connect(self.choose_element_color)
        layout.addWidget(self.element_color_button)

        tab.setLayout(layout)
        return tab

    def create_results_tab(self):
        """Pestaña para opciones de visualización de resultados."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Escala de deformación
        layout.addWidget(QLabel("Escala de deformación:"))
        self.deformation_scale = QDoubleSpinBox()
        self.deformation_scale.setRange(1, 100)
        self.deformation_scale.setValue(self.plotter_options.deformation_scale)
        layout.addWidget(self.deformation_scale)

        # Mostrar estructura sin deformar
        self.undeformed_checkbox = QCheckBox("Mostrar estructura sin deformar")
        self.undeformed_checkbox.setChecked(self.plotter_options.show_undeformed)
        layout.addWidget(self.undeformed_checkbox)

        # Color de deformaciones
        self.deformation_color_button = QPushButton("Seleccionar color de deformaciones")
        self.deformation_color_button.clicked.connect(self.choose_deformation_color)
        layout.addWidget(self.deformation_color_button)

        tab.setLayout(layout)
        return tab

    def choose_bg_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.plotter_options.background_color = color.name()

    def choose_node_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.plotter_options.node_color = color.name()

    def choose_element_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.plotter_options.element_color = color.name()

    def choose_deformation_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.plotter_options.deformation_color = color.name()

    def apply_changes(self):
        """Guarda los cambios y cierra la ventana."""
        self.plotter_options.figure_size = eval(self.figure_size.text())  # Convertir a tupla
        self.plotter_options.dpi = self.dpi.value()
        self.plotter_options.grid = self.grid_checkbox.isChecked()
        self.plotter_options.node_size = self.node_size.value()
        self.plotter_options.deformation_scale = self.deformation_scale.value()
        self.plotter_options.show_undeformed = self.undeformed_checkbox.isChecked()

        self.accept()  # Cierra la ventana

class MainWindow(QMainWindow):
    """Ventana principal con opción para abrir configuración."""
    
    def __init__(self, plotter_options):
        super().__init__()
        self.setWindowTitle("Configurador de Gráficos")
        self.setGeometry(100, 100, 600, 400)

        self.plotter_options = plotter_options

        self.config_button = QPushButton("Abrir Configuración", self)
        self.config_button.setGeometry(200, 150, 200, 50)
        self.config_button.clicked.connect(self.open_config)

    def open_config(self):
        """Abre la ventana de configuración."""
        dialog = PlotterOptionsDialog(self.plotter_options)
        if dialog.exec():
            print("Opciones actualizadas:", vars(self.plotter_options))

class PlotterOptions:
    """Clase base con opciones predeterminadas."""
    
    def __init__(self):
        self.figure_size = (10, 8)
        self.dpi = 100
        self.background_color = 'white'
        self.grid = False
        self.node_size = 4
        self.node_color = 'blue'
        self.element_color = 'blue'
        self.deformation_scale = 40
        self.show_undeformed = False
        self.deformation_color = '#007acc'
