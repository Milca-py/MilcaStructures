from typing import TYPE_CHECKING, Dict
from milcapy.core.results import Results
from milcapy.postprocess.post_processing import PostProcessing
from milcapy.analysis.static import LinearStaticAnalysis
if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel



class AnalysisManager:
    """Clase mananger para el análisis estructural.
    maneja los tipos de análisis y opciones de análisis estructural.
    realiza el análisis estructural para todos las condiciones de carga."""
    
    def __init__(
        self,
        model: "SystemMilcaModel",
    ) -> None:
        """
        Inicializa el análisis estructural.

        Args:
            model: Sistema estructural a analizar.
        """
        self.model = model
        self.load_patterns = self.model.load_pattern_map.values()

        # guardar resultados del análisis para cada condición de carga en diferentes objetos
        self.loadpattern_results: Dict[str, Results] = self.model.loadpattern_results



    def run(self) -> None:
        """Ejecuta el análisis estructural para todas las condiciones de carga."""
        
        # solucionar para load pattern
        for load_pattern in self.load_patterns:
            # if load_pattern != self.load_patterns[0]: # si no es la primera condición de carga
            #     print("Reseteando el modelo...")
            #     self.model.reset()
            
            # Asignar las cargas a los nodos y elementos almacenados en el patrón de carga
            load_pattern.assign_loads_to_nodes(self.model)
            load_pattern.assign_loads_to_elements(self.model)
            # Compilar las matrices locales y de transformación de cada elemento
            for element in self.model.element_map.values():
                element.compile()

            # resolver el modelo
            analysis = LinearStaticAnalysis(self.model, self.model.analysis_options, self.model.solver_options)
            analysis.run()

            # Crear un objeto de resultados para almacenar los resultados del análisis
            self.loadpattern_results[load_pattern.name] = Results(self.model, self.model.results_options)
            
            # crear un objeto de post-procesamiento para procesar los resultados
            post_processing = PostProcessing(self.model, self.loadpattern_results[load_pattern.name], self.model.postprocessing_options)
            
            post_processing.process_all_elements()

            # actualizar el estado de analisis en LoadPattern
            load_pattern.analyzed = True
            