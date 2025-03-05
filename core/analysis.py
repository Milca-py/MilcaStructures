from typing import List, Dict, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from core.system import SystemMilcaModel
    from utils import TypeAnalysis


class AnalysisOptions:
    "Clase que contiene las opciones de análisis."
    def __init__(self):
        self.status = False



class Analysis:
    "Clase que extrae, organiza y almacena los resultados de las simulaciones."
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system
        self.options: AnalysisOptions = AnalysisOptions()



class StaticAnalysis:
    "Clase que contiene las opciones de análisis estático."
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system

