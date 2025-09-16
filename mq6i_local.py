import milcapy as mp
from enum import Enum, auto


class ElementType(Enum):
    MQ6 = auto()
    MQ6I = auto()
    MQ6IMod = auto()


element_type = ElementType.MQ6I
t = 0.4

E = 2e6
v = 0.2

p = 20


model = mp.SystemModel()
model.add_node(4, 0, 1)
model.add_node(1, 1, 0)
model.add_node(2, 3, 1)
model.add_node(3, 3, 2)
model.add_material("concreto", E, v)
model.add_shell_section("viga", "concreto", t)
if element_type == ElementType.MQ6:
    model.add_membrane_q6(1, 1, 2, 3, 4, "viga")
if element_type == ElementType.MQ6I:
    model.add_membrane_q6i(1, 1, 2, 3, 4, "viga")
if element_type == ElementType.MQ6IMod:
    model.add_membrane_q6i_mod(1, 1, 2, 3, 4, "viga")
model.add_restraint(4, *(True, True, False))
model.add_restraint(1, *(True, True, False))
model.add_load_pattern("carga")
model.add_point_load(2, "carga", fy=-0.5*p)
model.add_point_load(3, "carga", fy=-0.5*p)
model.solve()
model.show()


