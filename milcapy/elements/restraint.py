from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milcapy.elements.node import Node

class Restraint:
    """Clase que define las restricciones de un nodo."""
    
    def __init__(
        self,
        node: "Node",
        ux: bool = False,
        uy: bool = False,
        rz: bool = False
    ) -> None:
        """
        Inicializa las restricciones de un nodo.
        
        Args:
            node: Nodo al que se le aplican las restricciones.
            ux: Restricción en dirección x.
            uy: Restricción en dirección y.
            rz: Restricción de rotación en z.
        """
        self.node = node
        self.ux = ux
        self.uy = uy
        self.rz = rz
    
    def to_tuple(self) -> tuple:
        """Devuelve las restricciones en formato de tupla."""
        return (self.ux, self.uy, self.rz)