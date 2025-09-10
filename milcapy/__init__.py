from .model.model import SystemMilcaModel
from .utils.types import (
    BeamTheoriesType,
    CoordinateSystemType,
    DirectionType,
    StateType,
    LoadType,
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
    from milcapy.plotter.UIdisplay import main_window
    from milcapy.plotter.plotter import Plotter
    model.plotter = Plotter(model)
    model.plotter.initialize_plot()
    main_window(model)