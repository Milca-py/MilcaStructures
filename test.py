
from core.system import SystemMilcaModel
from frontend.widgets.UIdisplay import create_plot_window
from frontend.widgets.display_array import mostrar_array


model = SystemMilcaModel()

model.add_material(
    name="concreto",
    modulus_elasticity=2.1e6,
    poisson_ratio=0.2
)

model.add_rectangular_section(
    name="seccion1",
    material_name="concreto",
    base=0.3,
    height=0.5
)
model.add_rectangular_section(
    name="seccion2",
    material_name="concreto",
    base=0.5,
    height=0.5
)
model.add_rectangular_section(
    name="seccion3",
    material_name="concreto",
    base=0.6,
    height=0.6
)

nodes = {
    1: (0, 0),
    2: (0, 5),
    3: (0, 8.5),
    4: (0, 12),
    5: (7, 0),
    6: (7, 5),
    7: (7, 8.5),
    8: (7, 12),
}

for key, value in nodes.items():
    model.add_node(key, value)

elements = {
    1: (1, 2, "seccion3"),
    2: (2, 3, "seccion3"),
    3: (3, 4, "seccion3"),
    4: (5, 6, "seccion2"),
    5: (6, 7, "seccion2"),
    6: (7, 8, "seccion2"),
    7: (2, 6, "seccion1"),
    8: (3, 7, "seccion1"),
    9: (4, 8, "seccion1"),
}

for key, value in elements.items():
    model.add_element(key, *value)

model.add_restraint(1, (True, True, True))
model.add_restraint(5, (True, True, True))

model.add_load_pattern(name="Live Load")
model.add_point_load(2, "Live Load", "GLOBAL", 5, 0, 0)
model.add_point_load(3, "Live Load", "GLOBAL", 10, 0, 0)
model.add_point_load(4, "Live Load", "GLOBAL", 20, 0, 0)

model.add_distributed_load(7, "Live Load", "LOCAL", -5, -5)
model.add_distributed_load(8, "Live Load", "LOCAL", -2, -6)
model.add_distributed_load(9, "Live Load", "LOCAL", -4, -3)

model.solve()

model.show_structure(show=False)
# root = create_plot_window(model.plotter.fig)
# root.mainloop()


k_global = model.global_stiffness_matrix

reactions = model.results.reactions


mostrar_array(reactions, True, True, 4)