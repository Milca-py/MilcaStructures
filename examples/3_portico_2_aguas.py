from milcapy import SystemModel

model = SystemModel()

model.add_material(name="concreto", modulus_elasticity=2.1e6, poisson_ratio=0.2)
model.add_rectangular_section(name="seccion1", material_name="concreto", base=0.3, height=0.5)

model.add_node(1, 0, 0)
model.add_node(2, 0, 5)
model.add_node(3, 3.5, 8.5)
model.add_node(4, 7, 5)
model.add_node(5, 7, 0)

model.add_member(1, 1, 2, "seccion1")
model.add_member(2, 3, 2, "seccion1")
model.add_member(3, 3, 4, "seccion1")
model.add_member(4, 5, 4, "seccion1")

model.add_restraint(1, *(True, True, True))
model.add_restraint(5, *(True, True, True))

model.add_load_pattern("Live Load")
# model.add_point_load(3, "Live Load", 0, -50, 0)
model.add_distributed_load(2, "Live Load", 5, 5)
model.add_distributed_load(3, "Live Load", -7, -7, "GLOBAL","Y")

model.solve()
model.show()

