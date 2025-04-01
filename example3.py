from milcapy import SystemMilcaModel
from milcapy.plotter.UIdisplay import main_window
# UNIDADES: (kN, m)

portico = SystemMilcaModel()
portico.add_material("concreto", 2.2e7, 0.3)
portico.add_rectangular_section("sec1", "concreto", 0.5, 0.5)

nodes = {
    1: (0, 0),
    2: (0, 4),
    3: (4, 0),
    4: (4, 4),
    5: (4, 8),
    6: (8, 8),
    7: (8, 0)
}

for node, coord in nodes.items():
    portico.add_node(node, *coord)

elements = {
    1: (1, 2, "sec1"),
    2: (2, 4, "sec1"),
    3: (4, 3, "sec1"),
    4: (4, 5, "sec1"),
    5: (5, 6, "sec1"),
    6: (6, 7, "sec1")
}

for element, (node1, node2, section) in elements.items():
    portico.add_member(element, node1, node2, section)

portico.add_restraint(1, (False, True, False))
portico.add_restraint(2, (False, False, False))
portico.add_restraint(3, (True, True, False))
portico.add_restraint(7, (True, True, False))

portico.add_load_pattern("CARGA1")
portico.add_point_load(6, "CARGA1", -10)
portico.add_distributed_load(2, "CARGA1", -5, -5)
portico.add_distributed_load(5, "CARGA1", -5, -5)

# --------------------------------------------------
# 5. Resolución del modelo
# --------------------------------------------------
portico.postprocessing_options.n = 40
portico.analysis_options.include_shear_deformations = True
portico.solve()

# --------------------------------------------------
# 6. Resultados
# --------------------------------------------------
portico._inicialize_plotter()
portico.plotter.initialize_plot()
main_window(portico)