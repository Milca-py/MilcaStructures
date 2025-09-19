from milcapy import SystemMilcaModel

E = 25  # MPa
v = 0.20

l = 2000  # mm

h = 600  # mm
t = 400  # mm

fy = 20 # kN
model = SystemMilcaModel()
model.add_material("concreto", E, v)
model.add_shell_section("muro", "concreto", t)
model.add_node(1, 0, 0)
model.add_node(2, l, 0)
model.add_node(3, l, h)
model.add_node(4, 0, h)
model.add_membrane_q6(1, 1, 2, 3, 4, "muro")
model.add_restraint(1, *(True, True, False))
model.add_restraint(4, *(True, True, False))
model.add_load_pattern("carga")
model.add_point_load(2, "carga", fy=-0.5*fy)
model.add_point_load(3, "carga", fy=-0.5*fy)
model.solve()
model.show()
