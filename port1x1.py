from milcapy import SystemModel, BeamTheoriesType, model_viewer

model = SystemModel()
b, h = 5, 3

model.add_material("concreto", 2.1e6, 0.2, 100)
model.add_rectangular_section("vigas", "concreto", 0.3, 0.5)

model.add_node(1, 0, 0)
model.add_node(2, 0, h)
model.add_node(3, b, h)
model.add_node(4, b, 0)

model.add_member(1, 1, 2, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_member(2, 2, 3, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_member(3, 3, 4, "vigas", BeamTheoriesType.EULER_BERNOULLI)

model.add_end_length_offset(2, la=0.5, lb=0.5)
model.add_releases(2, mi=True, mj=True)

model.add_restraint(1, *(True, True, True))
model.add_restraint(4, *(True, True, True))

model.add_node(5, 2*b, 0)
model.add_node(6, 2*b, h)
model.add_node(7, 3*b, h)
model.add_node(8, 3*b, 0)

model.add_member(4, 5, 6, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_member(5, 6, 7, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_member(6, 7, 8, "vigas", BeamTheoriesType.EULER_BERNOULLI)

model.add_end_length_offset(2, la=0.5, lb=0.5)
model.add_releases(5, mi=True, mj=True)

model.add_restraint(5, *(True, True, True))
model.add_restraint(8, *(True, True, True))

model.add_node(9, 4*b, 0)
model.add_node(10, 5*b, 0)

model.add_member(7, 9, 10, "vigas", BeamTheoriesType.EULER_BERNOULLI)

model.add_end_length_offset(7, la=0.5, lb=0.5)
model.add_releases(7, mi=True, mj=True)

model.add_restraint(9, *(True, True, True))
model.add_restraint(10, *(True, True, True))

model.add_node(11, 7*b, 0)
model.add_node(12, 8*b, 0)

model.add_member(11, 11, 12, "vigas", BeamTheoriesType.EULER_BERNOULLI)

model.add_end_length_offset(11, la=0.5, lb=0.5)

model.add_restraint(11, *(True, True, False))
model.add_restraint(12, *(True, True, False))


model.add_load_pattern("Live Load")
model.add_self_weight("Live Load")
model.postprocessing_options.n = 100
model.solve()
model_viewer(model)
