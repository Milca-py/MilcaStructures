from milcapy import SystemModel, BeamTheoriesType, model_viewer

model = SystemModel()
b, h = 5, 3

model.add_material(name="concreto", modulus_elasticity=2.1e6, poisson_ratio=0.2, specific_weight=10)
model.add_rectangular_section(name="vigas", material_name="concreto", base=0.3, height=0.5)

model.add_node(1, 0, 0)
model.add_node(2, 0, h)
model.add_node(3, b, h)
model.add_node(4, b, 0)

model.add_member(1, 1, 2, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_member(2, 2, 3, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_member(3, 3, 4, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_truss(4, 1, 3, "vigas")

model.add_restraint(1, *(True, True, True))
model.add_restraint(4, *(False, True, True))

model.add_local_axis_for_node(4, +10*3.1416/180)
model.add_elastic_support(4, kx=10, CSys="LOCAL")

lengthOffset = 0.25

model.add_end_length_offset(2, la=lengthOffset, lb=lengthOffset, qla=False, qlb=False)

model.add_load_pattern("Live Load")
model.add_self_weight("Live Load", 1)
model.add_point_load(2, "Live Load", fx=100)

model.add_prescribed_dof(1, "Live Load", uy=-0.001, CSys="LOCAL")

model.postprocessing_options.n = 100

model.solve()

model_viewer(model)
