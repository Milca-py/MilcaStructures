from typing import List, Dict, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from core.system import SystemMilcaModel
    from utils import TypeAnalysis



class Analysis:
    "Clase que extrae, organiza y almacena los resultados de las simulaciones."
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system
        self.options: AnalysisOptions = None
        self.results = {}

class AnalysisOptions:
    "Clase que contiene las opciones de análisis."
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system
        self.type: TypeAnalysis = None
        self.time = 0

class DynamicsAnalysis:
    "Clase que contiene las opciones de análisis dinámico."
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system
        self.type = 'dynamics'
        self.time = 0
        self.time_step = 0
        self.time_steps

class StaticAnalysis:
    "Clase que contiene las opciones de análisis estático."
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system
        self.type = 'static'
        self.time = 0
        self.time_step = 0
        self.time_steps

class OrderAnalysis:
    "Clase que contiene las opciones de análisis de orden."
    def __init__(self, system: 'SystemMilcaModel'):
        self.system = system
        self.type = 'first_order'
        self.time = 0
        self.time_step = 0
        self.time_steps