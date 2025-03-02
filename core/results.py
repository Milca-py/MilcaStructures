from typing import List, Dict, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from core.system import SystemMilcaModel


class Results:
    "Clase que extrae, organiza y almacena los resultados de las simulaciones."
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system
        self.results = {}
