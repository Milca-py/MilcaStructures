from .model.model import SystemMilcaModel
from .utils.types import (
    BeamTheoriesType,
    CoordinateSystemType,
    DirectionType,
    StateType,
    LoadType,
    FieldType,
    ConstitutiveModelType,
    IntegrationType,
    )
from typing_extensions import Self


class SystemModel(SystemMilcaModel):
    """
    Clase que representa el modelo estructural.
    Colecciona:
    - Materiales
    - Secciones
    - Nodos
    - Modelos de elementos finitos
    - Patrones de Carga
    - Resultados
    """
    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> Self:
        super().__init__()
        return self


def model_viewer(model: SystemModel) -> None:
    """
    Muestra la interfaz gráfica para visualizar el modelo.

    Parameters
    ----------
    model : SystemModel
        Modelo estructural a visualizar.
    """
    model.show()



__all__ = [
    "SystemModel",
    "model_viewer",
    "BeamTheoriesType",
    "CoordinateSystemType",
    "DirectionType",
    "StateType",
    "LoadType",
    "FieldType",
    "ConstitutiveModelType",
    "IntegrationType",
]