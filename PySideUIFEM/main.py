import sys
from PySide6.QtWidgets import QApplication
from UI.window import MainWindow
from UI.figureex import get_figure



app = QApplication.instance()
if app is None:  # Si no existe una instancia, créala
    app = QApplication(sys.argv)
fig = get_figure()
window = MainWindow(fig)
window.show()
sys.exit(app.exec()) 