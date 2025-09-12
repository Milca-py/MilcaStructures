from milcapy import SystemModel, BeamTheoriesType, model_viewer

model = SystemModel()

model.add_material(name="concreto", modulus_elasticity=2.1e6, poisson_ratio=0.2)
model.add_rectangular_section(name="vigas", material_name="concreto", base=0.3, height=0.5)
model.add_rectangular_section(name="muros", material_name="concreto", base=0.3, height=2.0)

model.add_node(1, 0, 0)
model.add_node(2, 0, 5)
model.add_node(3, 7, 8.5)
model.add_node(4, 14, 5)
model.add_node(5, 14, 0)

model.add_member(1, 1, 2, "muros", BeamTheoriesType.TIMOSHENKO)
model.add_member(2, 2, 3, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_member(3, 3, 4, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_member(4, 4, 5, "muros", BeamTheoriesType.TIMOSHENKO)
model.add_member(5, 2, 4, "vigas", BeamTheoriesType.EULER_BERNOULLI)

model.add_restraint(1, (False, True, True))
model.add_restraint(5, (False, True, True))

model.add_local_axis_for_node(1, -37*3.1416/180)
model.add_local_axis_for_node(5, +37*3.1416/180)

lengthOffset = 1

model.add_elastic_support(3, ky=10)
model.add_end_length_offset(2, la=lengthOffset, qla=True)
model.add_end_length_offset(3, lb=lengthOffset, qlb=True)
model.add_end_length_offset(5, la=lengthOffset, lb=lengthOffset, qla=True)

model.add_releases(5, mi=True, mj=True)

model.add_load_pattern("Live Load")
model.add_point_load(3, "Live Load", 0, -50, 0)
model.add_distributed_load(2, "Live Load", -10, -5)
model.add_distributed_load(3, "Live Load", -5, -10)
model.add_distributed_load(5, "Live Load", -5, -5)
model.add_load_pattern("Dead Load")
model.add_point_load(3, "Dead Load", 40, -50, 10)
model.add_distributed_load(5, "Dead Load", -5, -5)

model.add_prescribed_dof(1, "Live Load", uy=-0.01, CSys="LOCAL")
model.add_prescribed_dof(5, "Live Load", uy=-0.01, CSys="LOCAL")

model.postprocessing_options.n = 100

model.solve()

model_viewer(model)
