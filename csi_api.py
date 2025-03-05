
from core.system import SystemMilcaModel
from frontend.widgets.UIdisplay import create_plot_window
from frontend.widgets.display_array import mostrar_array

model = SystemMilcaModel()

model.add_material(
    name="concreto",
    modulus_elasticity=2e6,
    poisson_ratio=0.2
)

model.add_rectangular_section(
    name="seccion1",
    material_name="concreto",
    base=0.4,
    height=0.5
)

model.add_node(1, (0, 0))
model.add_node(2, (5, 0))

model.add_element(id=1, node_i_id=1, node_j_id=2, section_name="seccion1", type="FRAME")

model.add_restraint(1, (True, True, True))
model.add_restraint(2, (True, True, True))

model.add_load_pattern("Live")
model.add_distributed_load(1, "Live", "LOCAL", -10, -20)
model.add_distributed_load(1, "Live", "LOCAL", -10, -20, direction="LOCAL_1")


model.solve()

print(model.reactions)
print(model.displacements)
model.show_structure(show=False)
root = create_plot_window(model.plotter.fig)
root.mainloop()

