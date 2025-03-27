import sys
from trace import Trace

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QShortcut, QIcon, QDoubleValidator
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QCheckBox, QComboBox, QLineEdit, QColorDialog, QWidget, QWidgetAction
)
from milcapy.plotter.options import PlotterOptions
from milcapy.model.model import SystemMilcaModel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas



# Clase para mostrar un gráfico de Matplotlib en un widget de Qt
class MatplotlibCanvas(QWidget):
    def __init__(self, parent, model: 'SystemMilcaModel'):
        super().__init__(parent)
        ############################################################
        self.model = model
        self.figure = model.plotter.figure
        self.plotter_options = model.plotter_options
        self.axes = self.figure.axes  # Obtener todos los ejes
        ############################################################

        # Almacenar límites originales de cada eje
        self.original_limits = {
            ax: (ax.get_xlim(), ax.get_ylim()) for ax in self.axes}

        # Crear el lienzo de Matplotlib
        self.canvas = FigureCanvas(self.figure)
        self.canvas.draw()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Variables de paneo
        self._pan_start = None
        self.pan_active = False
        self.active_ax = None  # Eje activo

        # Conectar eventos de ratón
        self._connect_events()

    @property
    def current_load_pattern(self) -> str | None:
        return self.model.current_load_pattern

    def _connect_events(self):
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event',
                                self._on_button_release)
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



# Clase para la ventana de opciones de gráfico (ventas emergente)
class GraphicOptionsDialog(QDialog):
    """Ventana emergente para opciones de gráfico."""
    def __init__(self, parent=None, model: 'SystemMilcaModel' = None, options=None):
        super().__init__(parent)
        self.setWindowTitle("Opciones de Gráfico")
        self.setMinimumSize(400, 500)
        self.model = model
        self.plotter_options = model.plotter.plotter_options
        self.options = options  # Diccionario para almacenar opciones seleccionadas

        main_layout = QVBoxLayout()

        # Secciones de opciones
        main_layout.addWidget(self.create_general_options())
        main_layout.addWidget(self.create_node_options())
        main_layout.addWidget(self.create_member_options())
        main_layout.addWidget(self.create_load_options())
        main_layout.addWidget(self.create_deformed_shape_options())
        main_layout.addWidget(self.create_internal_forces_options())

        # Botones Aceptar / Restaurar / Aplicar / Cancelar
        button_layout = QHBoxLayout()
        accept_button = QPushButton("Aceptar")
        restore_button = QPushButton("Restaurar Valores")
        apply_button = QPushButton("Aplicar")
        cancel_button = QPushButton("Cancelar")

        accept_button.clicked.connect(self.accept_changes)
        restore_button.clicked.connect(self.restore_defaults)
        apply_button.clicked.connect(self.apply_changes)
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(accept_button)
        button_layout.addWidget(restore_button)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    @property
    def current_load_pattern(self) -> str | None:
        return self.model.current_load_pattern

    def create_general_options(self):
        """Opciones generales como color de fondo."""
        group = QGroupBox("Generales")
        layout = QVBoxLayout()

        self.background_color_button = QPushButton(
            "Seleccionar Color de Fondo")
        self.background_color_button.clicked.connect(
            self._select_background_color)
        layout.addWidget(self.background_color_button)

        group.setLayout(layout)
        return group

    def create_node_options(self):
        """Opciones para nodos."""
        group = QGroupBox("Nodos")
        layout = QVBoxLayout()

        self.show_nodes_checkbox = QCheckBox("Mostrar Nodos")
        self.node_labels_checkbox = QCheckBox("Mostrar Etiquetas de Nodos")

        layout.addWidget(self.show_nodes_checkbox)
        layout.addWidget(self.node_labels_checkbox)
        group.setLayout(layout)
        return group

    def create_member_options(self):
        """Opciones para miembros (elementos)."""
        group = QGroupBox("Miembros")
        layout = QVBoxLayout()

        self.show_members_checkbox = QCheckBox("Mostrar Miembros")
        self.member_labels_checkbox = QCheckBox(
            "Mostrar Etiquetas de Miembros")

        layout.addWidget(self.show_members_checkbox)
        layout.addWidget(self.member_labels_checkbox)
        group.setLayout(layout)
        return group

    def create_load_options(self):
        """Opciones para visualización de cargas."""
        group = QGroupBox("Cargas")
        layout = QVBoxLayout()

        self.show_loads_checkbox = QCheckBox("Mostrar Cargas")

        layout.addWidget(self.show_loads_checkbox)
        group.setLayout(layout)
        return group

    def create_deformed_shape_options(self):
        """Opciones para visualización de la forma deformada."""
        group = QGroupBox("Forma Deformada")
        layout = QVBoxLayout()

        self.deformation_scale_input = QLineEdit()
        self.deformation_scale_input.setPlaceholderText(
            "Escala de deformación")
        self.deformation_scale_input.setValidator(
            QDoubleValidator(0.01, 1000.0, 2))

        layout.addWidget(QLabel("Escala de Deformación"))
        layout.addWidget(self.deformation_scale_input)
        group.setLayout(layout)
        return group

    def create_internal_forces_options(self):
        """Opciones para fuerzas internas y colormap."""
        group = QGroupBox("Fuerzas Internas")
        layout = QVBoxLayout()

        # Tipo de relleno
        self.filling_type_combo = QComboBox()
        self.filling_type_combo.addItems(["Sólido", "Sin Relleno", "Colormap"])

        # Colormap
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["Jet", "Viridis", "Coolwarm"])

        # Checkbox para barra de colores
        self.show_colorbar_checkbox = QCheckBox("Mostrar Barra de Colores")

        layout.addWidget(QLabel("Tipo de Relleno"))
        layout.addWidget(self.filling_type_combo)
        layout.addWidget(QLabel("Colormap"))
        layout.addWidget(self.colormap_combo)
        layout.addWidget(self.show_colorbar_checkbox)

        group.setLayout(layout)
        return group

    def _select_background_color(self):
        """Abre el selector de color para el fondo."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.options["UI_background_color"] = color.name()

    def restore_defaults(self):
        """Restablece las opciones a los valores por defecto."""
        self.show_nodes_checkbox.setChecked(False)
        self.node_labels_checkbox.setChecked(False)
        self.show_members_checkbox.setChecked(True)
        self.member_labels_checkbox.setChecked(False)
        self.show_loads_checkbox.setChecked(False)
        self.deformation_scale_input.setText("40")
        self.filling_type_combo.setCurrentText("Sólido")
        self.colormap_combo.setCurrentText("Jet")
        self.show_colorbar_checkbox.setChecked(True)
        self.__recover_values()
        self.__transfer_changes()
        self.update_changer()
        print("Restaurado a valores por defecto.")

    def accept_changes(self):
        """Guarda las opciones seleccionadas y las imprime."""
        self.__recover_values()
        self.__transfer_changes()
        self.update_changer()
        self.accept()

    def apply_changes(self):
        """Aplica las opciones seleccionadas pero no cierra la ventana."""
        self.__recover_values()
        self.__transfer_changes()
        self.update_changer()
        print(self.model.plotter_options.UI_load)

    def __recover_values(self):
        """Recupera los valores de los widgets."""
        self.options["UI_show_nodes"] = self.show_nodes_checkbox.isChecked()
        self.options["UI_node_labels"] = self.node_labels_checkbox.isChecked()
        self.options["UI_show_members"] = self.show_members_checkbox.isChecked()
        self.options["UI_member_labels"] = self.member_labels_checkbox.isChecked()
        self.options["UI_load"] = self.show_loads_checkbox.isChecked()
        deformation_scale_text = self.deformation_scale_input.text()
        self.options["UI_deformation_scale"] = float(
            deformation_scale_text) if deformation_scale_text else 40
        self.options["UI_filling_type"] = self.filling_type_combo.currentText()
        self.options["UI_colormap"] = self.colormap_combo.currentText()
        self.options["UI_show_colorbar"] = self.show_colorbar_checkbox.isChecked()

    def __transfer_changes(self):
        """Transfiere las opciones seleccionadas al objeto PlotterOptions."""
        op = self.plotter_options
        op.UI_background_color = self.options.get("UI_background_color", "white")
        op.UI_show_nodes = self.options.get("UI_show_nodes", True)
        op.UI_show_members = self.options.get("UI_show_members", True)
        op.UI_node_labels = self.options.get("UI_node_labels", False)
        op.UI_member_labels = self.options.get("UI_member_labels", False)
        op.UI_load = self.options.get("UI_load", True)
        op.UI_deformation_scale = self.options.get("UI_deformation_scale", 40)
        if self.options.get("UI_filling_type", "Sólido") == "Sin Relleno":
            op.UI_fill_diagram = False
        else:
            op.UI_filling_type = self.options.get("UI_filling_type", "Sólido")
        op.UI_colormap = self.options.get("UI_colormap", "Jet")
        op.UI_show_colorbar = self.options.get("UI_show_colorbar", True)

    def _keep_data(self):
        """asigna los valores de las opciones anteriores (mantiene) a los widgets."""
        self.show_nodes_checkbox.setChecked(self.options.get("UI_show_nodes", True))
        self.node_labels_checkbox.setChecked(self.options.get("UI_node_labels", False))
        self.show_members_checkbox.setChecked(self.options.get("UI_show_members", True))
        self.member_labels_checkbox.setChecked(self.options.get("UI_member_labels", False))
        self.show_loads_checkbox.setChecked(self.options.get("UI_load", True))
        self.deformation_scale_input.setText(str(self.options.get("UI_deformation_scale", 40)))
        self.filling_type_combo.setCurrentText(self.options.get("UI_filling_type", "Sólido"))
        self.colormap_combo.setCurrentText(self.options.get("UI_colormap", "Jet"))
        self.show_colorbar_checkbox.setChecked(self.options.get("UI_show_colorbar", True))

    def update_changer(self):
        """Actualiza los cambios al figura principal y su vista inmediata."""
        self.model.plotter.change_background_color()    # cambia el color de fondo
        self.model.plotter.update_nodes()               # actualiza los nodos
        self.model.plotter.update_members()             # actualiza los miembros
        self.model.plotter.update_node_labels()         # actualiza los labels de los nodos
        self.model.plotter.update_member_labels()       # actualiza los labels de los miembros
        self.model.plotter.update_supports()            # actualiza los soportes
        self.model.plotter.update_point_load()
        self.model.plotter.update_point_load_labels()
        self.model.plotter.update_distributed_loads()
        self.model.plotter.update_distributed_load_labels()

        # if self.model.plotter.rigid_deformed_shape != {}:
        # self.model.plotter.rigid_deformed_shape = {}
        # for lp_name in self.model.results.keys():
        #     self.model.plotter.plot_rigid_deformed(escala=self.options.get("UI_deformation_scale", 40))



# Clase para la ventana principal
class MainWindow(QMainWindow):
    def __init__(self, model: 'SystemMilcaModel'):
        super().__init__()
        self.setWindowTitle("MILCApy")
        self.setGeometry(100, 100, 800, 800)
        self.setWindowIcon(QIcon("milcapy/plotter/assets/milca.ico"))

        self.model = model

        # Crear el widget de Matplotlib
        self.plot_widget = MatplotlibCanvas(self, self.model)
        self.setCentralWidget(self.plot_widget)

        # Crear la barra de herramientas
        self.toolbar = self.addToolBar("Herramientas")

        # Acción para guardar IMAGENES
        action_save = QAction(QIcon("otros/logo.png"), "Guardar", self)
        action_save.triggered.connect(self.on_save_figure)
        self.toolbar.addAction(action_save)

        # SEPARADOR
        self.toolbar.addSeparator()

        # Acción para OPCIONES DE GRAFICO
        action_save = QAction(
            QIcon("otros/opciones_de_grafico.png"), "Opciones de gráfico", self)
        action_save.triggered.connect(self.abrir_opciones_grafico)
        self.toolbar.addAction(action_save)

        # SEPARADOR
        self.toolbar.addSeparator()

        # SELECCIONAR LOAD PATERNS
        self.combo = QComboBox()
        lista_patterns = list(self.model.results.keys())
        self.combo.addItems(lista_patterns)
        self.combo.currentTextChanged.connect(self.on_pattern_selected)

        # Envolver el ComboBox en un QWidgetAction
        combo_action = QWidgetAction(self)
        combo_action.setDefaultWidget(self.combo)

        # Agregar a la barra de herramientas
        self.toolbar.addAction(combo_action)
        self.toolbar.addSeparator()

        # CASILLAS DE VERIFICACION:
        self.DFA = QCheckBox("DFA")
        self.DFA.stateChanged.connect(self.mostrar_fuerzas_axiales)
        self.toolbar.addWidget(self.DFA)
        self.toolbar.addSeparator()
        self.DFC = QCheckBox("DFC")
        self.DFC.stateChanged.connect(self.mostrar_fuerzas_cortantes)
        self.toolbar.addWidget(self.DFC)
        self.toolbar.addSeparator()
        self.DMF = QCheckBox("DMF")
        self.DMF.stateChanged.connect(self.mostrar_fuerzas_momentos)
        self.toolbar.addWidget(self.DMF)
        self.toolbar.addSeparator()
        self.REACIONES = QCheckBox("Reacciones")
        self.REACIONES.stateChanged.connect(self.mostrar_reacciones)
        self.toolbar.addWidget(self.REACIONES)
        self.toolbar.addSeparator()
        self.DEFORMADA = QCheckBox("Deformada")
        self.DEFORMADA.stateChanged.connect(self.mostrar_deformada)
        self.toolbar.addWidget(self.DEFORMADA)
        self.toolbar.addSeparator()
        self.DEFORMADA_RIGIDA = QCheckBox("Deformada rígida")
        self.DEFORMADA_RIGIDA.stateChanged.connect(
            self.mostrar_deformada_rigida)
        self.toolbar.addWidget(self.DEFORMADA_RIGIDA)

        # Atajo de teclado (Ctrl + H) para mostrar/ocultar la barra de herramientas
        self.toggle_toolbar_shortcut = QShortcut(Qt.CTRL | Qt.Key_H, self)
        self.toggle_toolbar_shortcut.activated.connect(self.toggle_toolbar)

        # Inicialmente oculta la barra de herramientas
        self.toolbar.setVisible(True)

        #####################################################
        self.options_values = {
            "UI_background_color": "white",
            "UI_show_nodes": True,
            "UI_node_labels": False,
            "UI_show_members": True,
            "UI_member_labels": False,
            "UI_load": True,
            "UI_deformation_scale": 40,
            "UI_internal_forces_scale": 0.03,
            "UI_filling_type": "Sólido",
            "UI_colormap": "Jet",
            "UI_show_colorbar": True,
        }
        #####################################################

    def toggle_toolbar(self):
        """Activa/Desactiva la barra de herramientas"""
        self.toolbar.setVisible(not self.toolbar.isVisible())

    def on_save_figure(self):
        self.model.plotter.figure.savefig("figure.png", dpi=self.model.plotter_options.save_fig_dpi, bbox_inches="tight")

    def abrir_opciones_grafico(self):
        """Abre la ventana emergente de opciones de gráfico"""
        dialog = GraphicOptionsDialog(self, self.model, self.options_values)
        dialog._keep_data()
        dialog.show()

    def on_pattern_selected(self, value):
        self.NOTIFICATION(value)
        print(f"Seleccionaste el patron: {value}")
        self.DFA.setChecked(False)
        self.DFC.setChecked(False)
        self.DMF.setChecked(False)
        self.REACIONES.setChecked(False)
        self.DEFORMADA.setChecked(False)
        self.DEFORMADA_RIGIDA.setChecked(False)

    def NOTIFICATION(self, current_load_pattern):
        self.model.current_load_pattern = current_load_pattern
        self.model.plotter.update_change()

    def mostrar_fuerzas_axiales(self, state):
        """Muestra las fuerzas axiales"""
        if state == 2:
            print("Fuerzas axiales mostradas")
        elif state == 0:
            print("Fuerzas axiales ocultadas")

    def mostrar_fuerzas_cortantes(self, state):
        """Muestra las fuerzas cortantes"""
        if state == 2:
            print("Fuerzas cortantes mostradas")
        elif state == 0:
            print("Fuerzas cortantes ocultadas")

    def mostrar_fuerzas_momentos(self, state):
        """Muestra las fuerzas momentos"""
        if state == 2:
            print("Fuerzas momentos mostradas")
        elif state == 0:
            print("Fuerzas momentos ocultadas")

    def mostrar_reacciones(self, state):
        """Muestra las reacciones"""
        if state == 2:
            print("Reacciones mostradas")
        elif state == 0:
            print("Reacciones ocultadas")

    def mostrar_deformada(self, state):
        """Muestra la deformada"""
        if state == 2:
            print("Deformada mostrada")
        elif state == 0:
            print("Deformada ocultada")

    def mostrar_deformada_rigida(self, state):
        """Muestra la deformada rígida"""
        if state == 2:
            print("Deformada rígida mostrada")
            self.model.plotter_options.UI_rigid_deformed = True
            self.model.plotter.update_rigid_deformed()
        elif state == 0:
            print("Deformada rígida ocultada")
            self.model.plotter_options.UI_rigid_deformed = False
            self.model.plotter.update_rigid_deformed()




def main_window(model: 'SystemMilcaModel'):
    app = QApplication.instance()
    if app is None:  # Si no existe una instancia, créala
        app = QApplication(sys.argv)
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec())