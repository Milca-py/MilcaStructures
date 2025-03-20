from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from milcapy.solvers.direct_solver import DirectStiffnessSolver

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from milcapy.solvers.direct_solver import SolverOptions
    from milcapy.analysis.options import AnalysisOptions, LinearStaticOptions



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
