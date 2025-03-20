from milcapy import SystemMilcaModel
# UNIDADES: (kN, m)

sys = SystemMilcaModel()
sys.add_material("concreto", 2.2e7, 0.3)
sys.add_rectangular_section("sec1", "concreto", 0.5, 0.5)
sys.add_rectangular_section("sec2", "concreto", 0.3, 0.65)

nodes = {
    1: (0, 0),
    2: (0, 4),
    3: (4, 6),
    4: (8, 4),
    5: (8, 0)
}

for node, coord in nodes.items():
    sys.add_node(node, coord)

elements = {
    1: (1, 2, "sec1"),
    2: (2, 3, "sec2"),
    3: (4, 3, "sec2"),
    4: (5, 4, "sec1")
}

for element, (node1, node2, section) in elements.items():
    sys.add_element(element, node1, node2, section)

sys.add_restraint(1, (True, True, True))
sys.add_restraint(5, (True, True, True))

sys.add_load_pattern("CARGA1")
sys.add_distributed_load(2, "CARGA1", "LOCAL", -5, -5)
sys.add_distributed_load(3, "CARGA1", "GLOBAL", 7, 7, direction="GRAVITY")

sys.add_load_pattern("CARGA2")
sys.add_distributed_load(1, "CARGA2", "LOCAL", -5, -5)
sys.add_distributed_load(2, "CARGA2", "LOCAL", -5, -5)
sys.add_distributed_load(3, "CARGA2", "LOCAL", 5, 5)
sys.add_distributed_load(4, "CARGA2", "LOCAL", 5, 5)

# --------------------------------------------------
# 5. Resolución del modelo
# --------------------------------------------------
sys.postprocessing_options.n = 40
sys.solve()

# --------------------------------------------------
# 6. Mostrar la estructura (opcional)
# --------------------------------------------------
sys.inicialize_plotter()
sys.plotter_options.internal_forces_scale = 10000
sys.plotter_options.deformation_scale = 1000
sys.plotter.set_load_pattern_name("CARGA2")
sys.plotter.plot_elements()
sys.plotter.plot_supports()
# sys.plotter.plot_nodes()
# sys.plotter.plot_element_labels()
# sys.plotter.plot_point_loads()
# sys.plotter.plot_distributed_loads()
# sys.plotter.plot_axial_force()
# sys.plotter.plot_shear_force()
# sys.plotter.plot_bending_moment()
# sys.plotter.plot_slope()
# sys.plotter.plot_deflection()
sys.plotter.plot_deformed()
# sys.plotter.plot_rigid_deformed()
sys.plotter.show()