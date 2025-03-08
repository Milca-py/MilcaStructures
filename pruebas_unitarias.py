from tests import test_arrows

import matplotlib.pyplot as plt
from frontend.widgets.UIdisplay import create_plot_window


fig, ax = plt.subplots()


test_arrows.graphic_one_arrow(
    x=0,
    y=0,
    load=-10,  
    length_arrow=50,
    angle=0,
    ax=ax,
    color="blue",
    label=True,
    color_label="black"
)

root = create_plot_window(fig)
root.mainloop()
