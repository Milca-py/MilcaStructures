from milcapy import SystemModel
nt = 7
l = 1
beam = SystemModel()
beam.add_material("concreto", 2e6, 0.15, 10)
beam.add_rectangular_section("vig40x40", "concreto", 0.4, 0.4)

for i in range(nt+1):
    beam.add_node(i+1, i*l, 0)
    beam.add_elastic_support(i+1, ky=100)

for i in range(nt):
    beam.add_member(i+1, i+1, i+2, "vig40x40")

# beam.add_restraint(1, *(True, True, False))
# beam.add_restraint(nt+1, *(True, True, False))

beam.add_load_pattern("Live Load")
beam.add_self_weight("Live Load")
beam.solve()

beam.show()