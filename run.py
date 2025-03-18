
from milcapy import SystemMilcaModel
from milcapy import create_plot_window
# --------------------------------------------------
# 1. Definición del modelo y secciones
# --------------------------------------------------

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

# --------------------------------------------------
# 2. Definición de nodos
# --------------------------------------------------
nodes = {
    1: (0, 0),
    2: (0, 5),
    3: (0, 8.5),
    4: (0, 12),
    5: (7, 0),
    6: (7, 5),
    7: (7, 8.5),
    8: (7, 12),
    # 9: (3.5, 15)  # Desactivado en este ejemplo
}

for key, value in nodes.items():
    model.add_node(key, value)

# --------------------------------------------------
# 3. Definición de elementos
# --------------------------------------------------
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
    # 10: (4, 9, "seccion1"),
    # 11: (8, 9, "seccion1")
}

for key, value in elements.items():
    model.add_element(key, *value)

# --------------------------------------------------
# 4. Restricciones y cargas
# --------------------------------------------------
model.add_restraint(1, (True, True, True))
model.add_restraint(5, (True, True, True))

model.add_load_pattern(name="Live Load")
model.add_point_load(2, "Live Load", "GLOBAL", 5, 0, 0)
model.add_point_load(3, "Live Load", "GLOBAL", 10, 0, 0)
model.add_point_load(4, "Live Load", "GLOBAL", 20, 0, 7)

# model.add_distributed_load(7, "Live Load", "GLOBAL", 5, 5, direction="GRAVITY_PROJ")
# model.add_distributed_load(5, "Live Load", "LOCAL", 5, 5)
model.add_distributed_load(7, "Live Load", "LOCAL", -5, -5)
model.add_distributed_load(8, "Live Load", "LOCAL", -2, -6)
model.add_distributed_load(9, "Live Load", "LOCAL", -4, -3)

# --------------------------------------------------
# 5. Resolución del modelo
# --------------------------------------------------
model.postprocessing_options.n = 40
model.solve()

# --------------------------------------------------
# 6. Mostrar la estructura (opcional)
# --------------------------------------------------
model.inicialize_plotter()
model.plotter_options.internal_forces_scale = 40
model.plotter.set_load_pattern_name("Live Load")
# model.plotter.plot_nodes()
# model.plotter.plot_elements()
# model.plotter.plot_supports()
# model.plotter.plot_node_labels()
# model.plotter.plot_element_labels()
# model.plotter.plot_point_loads()
# model.plotter.plot_distributed_loads()
# model.plotter.plot_axial_force()
# model.plotter.plot_shear_force()
# model.plotter.plot_bending_moment()
# model.plotter.plot_slope()
# model.plotter.plot_deflection()
# model.plotter.plot_deformed()
# model.plotter.plot_rigid_deformed()
model.plotter.plot_structure()
model.plotter.show()
