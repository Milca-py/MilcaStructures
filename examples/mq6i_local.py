import milcapy as mp
from enum import Enum, auto


class ElementType(Enum):
    MQ4 = auto()
    MQ6 = auto()
    MQ8 = auto()
    MQ6I = auto()
    MQ6IMod = auto()


element_type = ElementType.MQ4
t = 0.4

E = 2.1e6
v = 0.2

p = 10


model = mp.SystemModel()
model.add_material("concreto", E, v)
model.add_shell_section("viga", "concreto", t)
# modelo tetha=45
# model.add_node(1, 1, 0)
# model.add_node(2, 3, 2)
# model.add_node(3, 2, 3)
# model.add_node(4, 0, 1)
# modelo tetha=135
# model.add_node(1, 3, 2)
# model.add_node(2, 2, 3)
# model.add_node(3, 0, 1)
# model.add_node(4, 1, 0)
# model.add_restraint(3, *(True, True, False))
# model.add_restraint(4, *(True, True, False))
# model.add_load_pattern("carga")
# model.add_point_load(1, "carga", fy=-0.5*p)
# model.add_point_load(2, "carga", fy=-0.5*p)
# modelo tetha=atn(0.2)
# model.add_node(1, 2.2, 1.8)
# model.add_node(2, 0, 1.4)
# model.add_node(3, 0.2, 0)
# model.add_node(4, 2.4, 0.4)
# model.add_restraint(2, *(True, True, False))
# model.add_restraint(3, *(True, True, False))
# model.add_load_pattern("carga")
# model.add_point_load(1, "carga", fy=-0.5*p)
# model.add_point_load(4, "carga", fy=-0.5*p)
# modelo tetha=0
# model.add_node(1, 0, 0)
# model.add_node(2, 2, 0)
# model.add_node(3, 2, 0.6)
# model.add_node(4, 0, 0.6)
# modelo tetha=0, CUDRILATERAL
# model.add_node(1, 0, 0)
# model.add_node(2, 2, 0)
# model.add_node(3, 3, 1.6)
# model.add_node(4, 1, 0.6)
# modelo tetha=90
# model.add_node(1, 2, 0)
# model.add_node(2, 2, 0.6)
# model.add_node(3, 0, 0.6)
# model.add_node(4, 0, 0)
# model.add_restraint(3, *(True, True, False))
# model.add_restraint(4, *(True, True, False))
# model.add_load_pattern("carga")
# model.add_point_load(1, "carga", fy=-0.5*p)
# model.add_point_load(2, "carga", fy=-0.5*p)
# # modelo tetha=alpha
# model.add_node(1, 3, 1)
# model.add_node(2, 3, 2)
# model.add_node(3, 0, 1)
# model.add_node(4, 1, 0)
# model.add_restraint(3, *(True, True, False))
# model.add_restraint(4, *(True, True, False))
# model.add_load_pattern("carga")
# model.add_point_load(1, "carga", fy=-0.5*p)
# model.add_point_load(2, "carga", fy=-0.5*p)
# modelo VIGA RECTANGULAR
model.add_node(1, 2, 0.5)
model.add_node(2, 0, 0.5)
model.add_node(3, 0, 0)
model.add_node(4, 2, 0)
model.add_restraint(2, *(True, True, False))
model.add_restraint(3, *(True, True, False))
model.add_load_pattern("carga")
model.add_point_load(1, "carga", fy=-0.5*p)
model.add_point_load(4, "carga", fy=-0.5*p)



if element_type == ElementType.MQ6:
    model.add_membrane_q6(1, 1, 2, 3, 4, "viga")
if element_type == ElementType.MQ6I:
    model.add_membrane_q6i(1, 1, 2, 3, 4, "viga")
if element_type == ElementType.MQ6IMod:
    model.add_membrane_q6i_mod(1, 1, 2, 3, 4, "viga")
if element_type == ElementType.MQ4:
    model.add_membrane_q4(1, 1, 2, 3, 4, "viga")
if element_type == ElementType.MQ8:
    model.add_membrane_q8(1, 1, 2, 3, 4, "viga")
# model.add_restraint(4, *(True, True, False))
# model.add_restraint(1, *(True, True, False))
# model.add_load_pattern("carga")
# model.add_point_load(2, "carga", fy=-0.5*p)
# model.add_point_load(3, "carga", fy=-0.5*p)
model.solve()
model.show()


