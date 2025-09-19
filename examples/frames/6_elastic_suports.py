from milcapy import SystemModel, BeamTheoriesType


theory = BeamTheoriesType.TIMOSHENKO

model = SystemModel()


model.add_material("concreto", 2.2*10**7, 0.3)
model.add_rectangular_section("sec1", "concreto", 0.4, 0.3)
model.add_rectangular_section("sec2", "concreto", 0.3, 0.5)
model.add_node(1, 0, 1.2)
model.add_node(2, 0, 4.7)
model.add_node(3, 6, 4.7)
model.add_node(4, 6, 0.0)
model.add_member(1, 1, 2, "sec1",theory)
model.add_member(2, 2, 3, "sec2",theory)
model.add_member(3, 4, 3, "sec1",theory)
model.add_restraint(1, *(True, True, False))
model.add_restraint(4, *(True, False, False))
model.add_elastic_support(4, ky=200)
model.add_load_pattern("dead")
model.add_distributed_load(1, "dead", 0, -15)
model.add_distributed_load(2, "dead", 0, -20)
model.solve()
model.show()
