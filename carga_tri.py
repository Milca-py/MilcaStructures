from milcapy import SystemMilcaModel
from milcapy.plotter.UIdisplay import main_window
# UNIDADES: (kN, m)

portico = SystemMilcaModel()
portico.add_material("concreto", 2.2e7, 0.3)
portico.add_rectangular_section("sec1", "concreto", 0.5, 0.5)
portico.add_rectangular_section("sec2", "concreto", 30000, 30000)

nodes = {
    1: (0, 0),
    2: (0, 4),
    3: (1, 4),
    4: (2, 4),
    5: (3, 4),
    6: (4, 4),
    7: (4, 0)
}

for node, coord in nodes.items():
    portico.add_node(node, *coord)

elements = {
    1: (1, 2, "sec1"),
    2: (2, 3, "sec2"),
    3: (3, 4, "sec1"),
    4: (4, 5, "sec1"),
    5: (5, 6, "sec2"),
    6: (6, 7, "sec1")
}

for element, (node1, node2, section) in elements.items():
    portico.add_member(element, node1, node2, section)

portico.add_restraint(1, (True, True, True))
portico.add_restraint(7, (True, True, True))

portico.add_load_pattern("CARGA1")
portico.add_point_load(2, "CARGA1", 1)
portico.add_distributed_load(3, "CARGA1", -0, -5)
portico.add_distributed_load(4, "CARGA1", -5, -0)

# --------------------------------------------------
# 5. Resolución del modelo
# --------------------------------------------------
portico.postprocessing_options.n = 40
portico.analysis_options.include_shear_deformations = True
portico.solve()

# --------------------------------------------------
# 6. Resultados
# --------------------------------------------------
portico.show()