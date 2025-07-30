from milcapy import SystemMilcaModel

model = SystemMilcaModel()
model.add_material("concreto", 2.1e6, 0.2)
model.add_rectangular_section("seccion1", "concreto", 0.5, 0.3)
model.add_node(1, 0, 0)
model.add_node(2, 4, 0)
model.add_member(1, 1, 2, "seccion1")
model.add_restraint(1, (True, True, True))
# model.add_restraint(2, (True, True, True))

model.add_load_pattern(name="Live Load")
model.add_distributed_load(1, "Live Load", -10000, -10000) ###

model.postprocessing_options.n = 11
model.solve()
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
print(model.results["Live Load"].get_member_deflection(1))
model.show()

