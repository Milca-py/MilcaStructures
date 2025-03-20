from abc import ABC, abstractmethod

class SolverOptions(ABC):
    """
    Clase abstracta para definir opciones generales de cualquier solucionador.
    """

    def __init__(self, solver_type: str) -> None:
        """
        Inicializa las opciones generales del solucionador.

        Args:
            solver_type (str): Tipo de solucionador.
        """
        self.solver_type = solver_type

    @abstractmethod
    def validate(self) -> bool:
        """Método abstracto para validar opciones específicas de cada solucionador."""
        pass


class DirectStiffnessSolverrOptions(SolverOptions):
    """
    Opciones específicas para solucionador de rigidez directa.
    """

    def __init__(self, method: str = "numpy") -> None:
        """
        Inicializa las opciones del solucionador directo.

        Args:
            method (str): Método de factorización directa ("lu", "cholesky", "qr", "numpy").
        """
        super().__init__(solver_type="direct")
        self.method = method.upper()

    def validate(self) -> bool:
        """Valida si el método de solución directa es válido."""
        valid_methods = {"lU", "CHOLESKY", "QR", "NUMPY"}
        if self.method not in valid_methods:
            print(f"Error: Método de solución no válido: {self.method}. Opciones válidas: {valid_methods}")
            return False
        return True
