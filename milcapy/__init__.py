from .model.model import SystemMilcaModel
from .utils.types import (
    BeamTheoriesType,
    CoordinateSystemType,
    DirectionType,
    StateType,
    LoadType,
    FieldTypeMembrane,
    ConstitutiveModel,
    IntegrationType,
    )

class SystemModel(SystemMilcaModel):
    """
    Class to create a model for the MILCA software.

    Attributes:
        model (SystemMilcaModel): Instance of the SystemMilcaModel class.
    """

    def __init__(self):
        super().__init__()

def model_viewer(
    model: SystemMilcaModel
):
    """
    Muestra la interfaz gráfica para visualizar el modelo.
    """
    model.show()