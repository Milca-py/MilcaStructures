from typing import List, Dict, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from milcapy.elements.system import SystemMilcaModel
    from milcapy.utils import TypeAnalysis


class AnalysisOptions:
    """Clase que contiene las opciones de análisis."""
    
    def __init__(self):
        """Inicializa las opciones de análisis con valores predeterminados."""
        self.status = False


class Analysis:
    """Clase que extrae, organiza y almacena los resultados de las simulaciones."""
    
    def __init__(self, system: 'SystemMilcaModel'):
        """
        Inicializa el análisis para un sistema estructural.
        
        Args:
            system: Sistema estructural a analizar
        """
        self.system = system
        self.options: AnalysisOptions = AnalysisOptions()


class StaticAnalysis:
    """Clase que contiene las opciones de análisis estático."""
    
    def __init__(self, system: 'SystemMilcaModel'):
        """
        Inicializa el análisis estático para un sistema estructural.
        
        Args:
            system: Sistema estructural a analizar
        """
        self.system = system