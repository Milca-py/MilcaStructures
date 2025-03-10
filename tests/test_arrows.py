from milcapy.display.load import graphic_one_arrow

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

def test_graphic_one_arrow(
    x: float,
    y: float,
    load: float,
    length_arrow: float,
    angle: float,
    ax: "Axes",
    color: str = "blue",
    label: bool = True,
    color_label: str = "black"
):
    graphic_one_arrow(
        x=x,
        y=y,
        load=load,
        length_arrow=length_arrow,
        angle=angle,
        ax=ax,
        color=color,
        label=label,
        color_label=color_label
    )