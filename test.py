from milcapy import SystemMilcaModel
from milcapy.plotter.UIdisplay import main_window
import numpy as np
import matplotlib.pyplot as plt
# UNIDADES: (kN, m)

portico = SystemMilcaModel()
portico.add_material("concreto", 2.2e7, 0.3)
portico.add_rectangular_section("sec1", "concreto", 0.5, 0.5)
portico.add_rectangular_section("sec2", "concreto", 0.3, 0.65)

nodes = {
    1: (0, 0),
    2: (0, 4),
    3: (4, 6),
    4: (8, 4),
    5: (8, 0)
}

for node, coord in nodes.items():
    portico.add_node(node, *coord)

elements = {
    1: (1, 2, "sec1"),
    2: (2, 3, "sec2"),
    3: (4, 3, "sec2"),
    4: (5, 4, "sec1")
}

for element, (node1, node2, section) in elements.items():
    portico.add_member(element, node1, node2, section)

portico.add_restraint(1, (True, True, True))
portico.add_restraint(5, (True, True, True))

portico.add_load_pattern("CARGA1")
portico.add_distributed_load(2, "CARGA1", -5, -5)
portico.add_distributed_load(3, "CARGA1", 7, 7, "GLOBAL", direction="GRAVITY")

portico.add_load_pattern("CARGA2")
portico.add_distributed_load(1, "CARGA2", -5, -5)
portico.add_distributed_load(2, "CARGA2", -5, -5)
portico.add_distributed_load(3, "CARGA2", -5, -5)
portico.add_distributed_load(4, "CARGA2", -5, -5)

# --------------------------------------------------
# 5. Resolución del modelo
# --------------------------------------------------
portico.postprocessing_options.n = 40
portico.solve()

# --------------------------------------------------
# 6. Resultados
# --------------------------------------------------
result_member = portico.results["CARGA1"].get_member_axial_force(3)
x = np.linspace(0, portico.members[3].length(), len(result_member))
fig, ax = plt.subplots()
ax.plot(x, result_member)
main_window(fig)