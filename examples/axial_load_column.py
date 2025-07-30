from milcapy import SystemModel


model = SystemModel()

model.add_material(
    name="concreto",
    modulus_elasticity=2.1e6,
    poisson_ratio=0.2
)

model.add_rectangular_section(name="seccion", material_name="concreto", base=0.3, height=0.5)

nodes = {
    1: (0, 0),
    2: (0, 5),
}
for key, value in nodes.items():
    model.add_node(key, *value)

elements = {
    1: (1, 2, "seccion"),

}

for key, value in elements.items():
    model.add_member(key, *value)

model.add_restraint(1, (True, True, True))

model.add_load_pattern(name="Live Load")
model.add_point_load(2, "Live Load", 100, -10000, 0, "GLOBAL")

model.add_distributed_load(member_id=1, load_pattern_name="Live Load", CSys="LOCAL", load_start=-100, load_end=-50, replace=False, direction="LOCAL_1", load_type="FORCE")

model.solve()
model.show()

