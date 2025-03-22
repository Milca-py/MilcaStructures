from typing import TYPE_CHECKING
from milcapy.solvers.direct_solver import DirectStiffnessSolver

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from milcapy.analysis.options import LinearStaticOptions



class LinearStaticAnalysis:
    """Análisis estructural para un sistema estático lineal."""

    def __init__(
        self,
        system: "SystemMilcaModel",
        analysis_options: "LinearStaticOptions",
    ) -> None:
        """
        Inicializa el análisis estático lineal.

        Args:
            system: Sistema estructural a analizar.
            options: Opciones específicas para el análisis estático lineal.
        """
        self.system = system
        self.analysis_options = analysis_options

    def run(self):
        """Ejecuta el análisis estático lineal."""
        solution = DirectStiffnessSolver(self.system, self.analysis_options)
        solution.assemble_global_load_vector()
        solution.assemble_global_stiffness_matrix()
        solution.solve()
