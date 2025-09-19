from milcapy import SystemModel

model = SystemModel()

model.add_material(name="concreto", modulus_elasticity=2100000, poisson_ratio=0.2)
model.add_rectangular_section(name="seccion1", material_name="concreto", base=0.3, height=0.5)
model.add_rectangular_section(name="seccion2", material_name="concreto", base=0.5, height=0.5)
model.add_rectangular_section(name="seccion3", material_name="concreto", base=0.6, height=0.6)

model.add_node(1, 0, 0)
model.add_node(2, 0, 5)
model.add_node(3, 0, 8.5)
model.add_node(4, 0, 12)
model.add_node(5, 7, 0)
model.add_node(6, 7, 5)
model.add_node(7, 7, 8.5)
model.add_node(8, 7, 12)

model.add_member(1, 1, 2, "seccion3")
model.add_member(2, 2, 3, "seccion3")
model.add_member(3, 3, 4, "seccion3")
model.add_member(4, 5, 6, "seccion2")
model.add_member(5, 6, 7, "seccion2")
model.add_member(6, 7, 8, "seccion2")
model.add_member(7, 2, 6, "seccion1")
model.add_member(8, 3, 7, "seccion1")
model.add_member(9, 4, 8, "seccion1")

model.add_restraint(1, *(True, True, True))
model.add_restraint(5, *(True, True, True))

model.add_load_pattern("Live Load")
model.add_point_load(2, "Live Load", 5, 0, 0, "GLOBAL")
model.add_point_load(3, "Live Load", 10, 0, 0, "GLOBAL")
model.add_point_load(4, "Live Load", 20, 0, 0, "GLOBAL")

model.add_distributed_load(7, "Live Load", -5, -5)
model.add_distributed_load(8, "Live Load", -2, -6)
model.add_distributed_load(9, "Live Load", -4, -3)

model.add_releases(member_id=9, mi=True, mj=True)
model.add_releases(member_id=8, mi=False, mj=True)
model.add_releases(member_id=7, mi=True, mj=False)

model.solve()
model.show()

