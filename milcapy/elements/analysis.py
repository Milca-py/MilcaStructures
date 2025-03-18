from typing import TYPE_CHECKING, Dict
from abc import ABC, abstractmethod
from milcapy.elements.solver import DirectStiffnessSolver
from milcapy.postprocess.results import Results
from milcapy.postprocess.post_processing import PostProcessing

if TYPE_CHECKING:
    from milcapy.elements.system import SystemMilcaModel
    from milcapy.elements.solver import SolutionMethod
    from milcapy.elements.solver import DirectStiffnessSolverrOptions, SolverOptions




# ==================================================================================================
# Clase para definir las opciones del análisis estructural
# ==================================================================================================


class AnalysisOptions(ABC):
    """
    Clase base abstracta para opciones de análisis estructural.
    Define atributos comunes y métodos generales.
    """

    def __init__(
        self,
        include_shear_deformations: bool,
        shear_coefficient_method: str,
    ) -> None:
        """
        Inicializa las opciones de análisis estructural.

        Args:
            include_shear_deformations (bool): Indica si se consideran deformaciones por cortante.
            shear_coefficient_method (str): Método para el coeficiente de Timoshenko. 
                                            Opciones: "Timoshenko", "Cowper".
        """
        self.include_shear_deformations = include_shear_deformations
        self.shear_coefficient_method = shear_coefficient_method

    @property
    @abstractmethod
    def analysis_type(self) -> str:
        """Atributo para definir el tipo de análisis."""
        pass

    def validate(self) -> bool:
        """
        Valida la configuración del análisis estructural.

        Returns:
            bool: True si la configuración es válida, False en caso contrario.
        """
        valid_methods = {"Timoshenko", "Cowper"}

        if self.shear_coefficient_method not in valid_methods:
            print(
                f"Error: Método de coeficiente no válido: {self.shear_coefficient_method}. Opciones válidas: {valid_methods}")
            return False

        return True


class LinearStaticOptions(AnalysisOptions):
    """Opciones específicas para análisis estático lineal."""

    def __init__(
        self,
        include_shear_deformations: bool = True,
        shear_coefficient_method: str = "Timoshenko" # v=0
    ) -> None:
        super().__init__(include_shear_deformations, shear_coefficient_method)

    @property
    def analysis_type(self) -> str:
        """Retorna el tipo de análisis."""
        return "linear_static"


# ==================================================================================================
# Clase para definir los metodos de análisis estructural
# ==================================================================================================

class Analysis(ABC):
    """
    Clase base para análisis estructural.
    Se encarga de analizar las condiciones en Load Pattern, Load Case y Load Combination.
    """

    def __init__(
        self,
        system: "SystemMilcaModel",
        analysis_options: "AnalysisOptions",
        solver_options: "SolverOptions"
    ) -> None:
        """
        Inicializa el análisis para un sistema estructural.

        Args:
            system: Sistema estructural a analizar.
            options: Opciones de configuración del análisis.
        """
        self.system = system
        self.analysis_options = analysis_options
        self.solver_options = solver_options

    @abstractmethod
    def run(self):
        """Método abstracto que define la ejecución del análisis."""
        pass


class LinearStaticAnalysis(Analysis):
    """Análisis estructural para un sistema estático lineal."""

    def __init__(
        self,
        system: "SystemMilcaModel",
        analysis_options: "LinearStaticOptions",
        solver_options: "SolverOptions"
    ) -> None:
        """
        Inicializa el análisis estático lineal.

        Args:
            system: Sistema estructural a analizar.
            options: Opciones específicas para el análisis estático lineal.
        """
        super().__init__(system, analysis_options, solver_options)

    def run(self):
        """Ejecuta el análisis estático lineal."""
        solution = DirectStiffnessSolver(self.system, self.analysis_options, self.solver_options)
        solution.assemble_global_load_vector()
        solution.assemble_global_stiffness_matrix()
        solution.solve()




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
            