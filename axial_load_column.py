from core.system import SystemMilcaModel
from frontend.widgets.UIdisplay import create_plot_window


model = SystemMilcaModel()

model.add_material(
    name="concreto",
    modulus_elasticity=2.1e6,
    poisson_ratio=0.2
)

model.add_rectangular_section(
    name="seccion",
    material_name="concreto",
    base=0.3,
    height=0.5
)

nodes = {
    1: (0, 0),
    2: (0, 5),
}
for key, value in nodes.items():
    model.add_node(key, value)

elements = {
    1: (1, 2, "seccion"),

}

for key, value in elements.items():
    model.add_element(key, *value)

model.add_restraint(1, (True, True, True))

model.add_load_pattern(name="Live Load")
model.add_point_load(2, "Live Load", "GLOBAL", -400, 100000, 0)

model.solve()

scale = 1
model.show_structure(show=False)
model.plotter.show_deformed(escala=scale, show=False)
model.plotter.show_rigid_deformed(escala=scale, show=False)

root = create_plot_window(model.plotter.fig)
root.mainloop()
