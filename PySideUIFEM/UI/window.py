from PySide6.QtWidgets import QMainWindow, QCheckBox
from PySide6.QtGui import QAction, QShortcut, QIcon, QFont
from PySide6.QtCore import Qt

from UI.options import PlotterOptionsDialog
from UI.matplotlib_canvas import MatplotlibCanvas


class MainWindow(QMainWindow):
    """Ventana principal con barra de menús"""
    def __init__(self, fig):
        super().__init__()
        self.setWindowTitle("Aplicación con Menú")
        self.setGeometry(100, 100, 1700, 800)
        self.setWindowIcon(QIcon("milcapy/plotter/assets/milca.ico"))

        # Crear el widget de Matplotlib
        self.plot_widget = MatplotlibCanvas(self, fig)
        self.setCentralWidget(self.plot_widget)

        # Crear la barra de herramientas
        self.toolbar = self.addToolBar("Herramientas")

        # Acción para guardar IMAGENES
        action_save = QAction(QIcon("otros/logo.png"), "Guardar", self)
        action_save.triggered.connect(self.guardar_archivo)
        self.toolbar.addAction(action_save)

        # SEPARADOR
        self.toolbar.addSeparator()

        # Acción para OPCIONES DE GRAFICO
        action_save = QAction(QIcon("otros/opciones_de_grafico.png"), "Opciones de gráfico", self)
        action_save.triggered.connect(self.abrir_opciones_grafico)
        self.toolbar.addAction(action_save)

        # SEPARADOR
        self.toolbar.addSeparator()

        # CASILLAS DE VERIFICACION:
        DFA = QCheckBox("DFA")
        DFA.stateChanged.connect(self.mostrar_fuerzas_axiales)
        DFA.setFont(QFont("Arial", 8))
        DFA.setStyleSheet("""
                          QCheckBox { spacing: 4px; margin: 2px; background-color: white; color: blue}
                          QCheckBox::indicator {
                              width: 15px;  /* Ancho del cuadrito */
                              height: 15px; /* Alto del cuadrito */
                          }
                          """)
        self.toolbar.addWidget(DFA)
        self.toolbar.addSeparator()
        DFC = QCheckBox("DFC")
        DFC.stateChanged.connect(self.mostrar_fuerzas_cortantes)
        self.toolbar.addWidget(DFC)
        self.toolbar.addSeparator()
        DMF = QCheckBox("DMF")
        DMF.stateChanged.connect(self.mostrar_fuerzas_momentos)
        self.toolbar.addWidget(DMF)
        self.toolbar.addSeparator()
        REACIONES = QCheckBox("Reacciones")
        REACIONES.stateChanged.connect(self.mostrar_reacciones)
        self.toolbar.addWidget(REACIONES)
        self.toolbar.addSeparator()
        DEFORMADA = QCheckBox("Deformada")
        DEFORMADA.stateChanged.connect(self.mostrar_deformada)
        self.toolbar.addWidget(DEFORMADA)
        self.toolbar.addSeparator()
        DEFORMADA_RIGIDA = QCheckBox("Deformada rígida")
        DEFORMADA_RIGIDA.stateChanged.connect(self.mostrar_deformada_rigida)
        self.toolbar.addWidget(DEFORMADA_RIGIDA)

        # Atajo de teclado (Ctrl + H) para mostrar/ocultar la barra de herramientas
        self.toggle_toolbar_shortcut = QShortcut(Qt.CTRL | Qt.Key_H, self)
        self.toggle_toolbar_shortcut.activated.connect(self.toggle_toolbar)

        # Inicialmente oculta la barra de herramientas
        self.toolbar.setVisible(True)

    def toggle_toolbar(self):
        """Activa/Desactiva la barra de herramientas"""
        self.toolbar.setVisible(not self.toolbar.isVisible())

    def abrir_opciones_grafico(self):
        """Abre la ventana emergente de opciones de gráfico"""
        dialog = PlotterOptionsDialog(self)
        dialog.exec()

    def guardar_archivo(self):
        """Guarda la imagen del gráfico"""
        print("======  Guardando imagen del gráfico ............. =================")

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
        elif state == 0:
            print("Deformada rígida ocultada")
