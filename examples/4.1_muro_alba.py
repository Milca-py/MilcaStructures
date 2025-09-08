from milcapy import SystemModel


portico = SystemModel()

Ec = 2173706.512
Ea = 175000
v = 0.25
t = 0.25
Ipe = 3.255 * 10 ** -4
Ie = 3.269
portico.add_material("concreto",Ec,v)
portico.add_material("albañileria",Ea,v)
portico.add_rectangular_section("col","concreto",t,t)
portico.add_generic_section("muro","albañileria",t*3,Ie,5/6)
portico.add_node(1, 0, 0)
portico.add_node(2, 0, 3)
portico.add_node(3, 4.5, 3)
portico.add_node(4, 4.5, 0)
portico.add_member(1, 1, 2, "muro")
portico.add_member(2, 2, 3, "col")
portico.add_member(3, 3, 4, "col")
portico.add_restraint(1, (True, True, True))
portico.add_restraint(4, (True, True, True))
portico.add_end_length_offset(2, 1.5, 0, False, False)
portico.add_load_pattern("Live Load")
portico.add_point_load(2, "Live Load", fx=10)
portico.solve()
portico.show()

