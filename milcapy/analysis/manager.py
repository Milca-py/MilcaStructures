from typing import TYPE_CHECKING, Dict
from milcapy.core.results import Results
from milcapy.postprocess.post_processing import PostProcessing
from milcapy.analysis.linear_static import LinearStaticAnalysis
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
        self.load_patterns = self.model.load_patterns.values()

        # guardar resultados del análisis para cada condición de carga en diferentes objetos
        self.load_pattern_results: Dict[str, Results] = self.model.load_pattern_results



    def run(self) -> None:
        """Ejecuta el análisis estructural para todas las condiciones de carga."""
        
        # solucionar para cada load pattern
        for load_pattern in self.load_patterns:

            # Asignar las cargas a los nodos y elementos almacenados en el patrón de carga
            load_pattern.assign_loads_to_nodes()
            load_pattern.assign_loads_to_elements()
            # Compilar las matrices y vectores de cada elemento
            for element in self.model.elements.values():
                element.compile_transformation_matrix()
                element.compile_local_stiffness_matrix()
                element.compile_global_stiffness_matrix()
                element.compile_local_load_vector()
                element.compile_global_load_vector()

            # resolver el modelo
            analysis = LinearStaticAnalysis(self.model, self.model.analysis_options)
            analysis.run()

            # Crear un objeto de resultados para almacenar los resultados del análisis
            self.load_pattern_results[load_pattern.name] = Results(self.model, self.model.results_options)
            
            # crear un objeto de post-procesamiento para procesar los resultados
            post_processing = PostProcessing(self.model, self.load_pattern_results[load_pattern.name], self.model.postprocessing_options)
            post_processing.process_all_elements()

            # actualizar el estado de analisis en LoadPattern
            load_pattern.analyzed = True