from milcapy import SystemModel

portal = SystemModel()
# UNITS: tonf, m
lBahias = 4
lPisos = 3.5
secVig = [0.3, 0.5]
secCol = [0.5, 0.5]
secBra = [0.1, 0.1]
matConc = [2.24e6, 0.2, 2.4]
matBra = [2.04e7, 0.3, 7.8]
portal.add_material("concreto", *matConc)
portal.add_rectangular_section("vig", "concreto", *secVig)
portal.add_rectangular_section("col", "concreto", *secCol)
portal.add_rectangular_section("bra", "concreto", *secBra)
portal.add_node(1,  0*lBahias, 0*lPisos)
portal.add_node(2,  0*lBahias, 1*lPisos)
portal.add_node(3,  0*lBahias, 2*lPisos)
portal.add_node(4,  0*lBahias, 3*lPisos)
portal.add_node(5,  0*lBahias, 4*lPisos)
portal.add_node(6,  0*lBahias, 5*lPisos)
portal.add_node(7,  0*lBahias, 6*lPisos)
portal.add_node(8,  0*lBahias, 7*lPisos)
portal.add_node(9,  1*lBahias, 0*lPisos)
portal.add_node(10, 1*lBahias, 1*lPisos)
portal.add_node(11, 1*lBahias, 2*lPisos)
portal.add_node(12, 1*lBahias, 3*lPisos)
portal.add_node(13, 1*lBahias, 4*lPisos)
portal.add_node(14, 1*lBahias, 5*lPisos)
portal.add_node(15, 1*lBahias, 6*lPisos)
portal.add_node(16, 1*lBahias, 7*lPisos)
portal.add_node(17, 2*lBahias, 0*lPisos)
portal.add_node(18, 2*lBahias, 1*lPisos)
portal.add_node(19, 2*lBahias, 2*lPisos)
portal.add_node(20, 2*lBahias, 3*lPisos)
portal.add_node(21, 2*lBahias, 4*lPisos)
portal.add_node(22, 2*lBahias, 5*lPisos)
portal.add_node(23, 2*lBahias, 6*lPisos)
portal.add_node(24, 2*lBahias, 7*lPisos)
portal.add_node(25, 3*lBahias, 0*lPisos)
portal.add_node(26, 3*lBahias, 1*lPisos)
portal.add_node(27, 3*lBahias, 2*lPisos)
portal.add_node(28, 3*lBahias, 3*lPisos)
portal.add_node(29, 3*lBahias, 4*lPisos)
portal.add_node(30, 3*lBahias, 5*lPisos)
portal.add_node(31, 3*lBahias, 6*lPisos)
portal.add_node(32, 3*lBahias, 7*lPisos)
for i in range(1, 8):
    # COLUMNAS
    portal.add_member(i,    i,      i+1,    "col")
    portal.add_member(i+7,  i+8,    i+9,    "col")
    portal.add_member(i+14, i+16,   i+17,   "col")
    portal.add_member(i+21, i+24,   i+25,   "col")
    # VIGAS
    portal.add_member(i+28, i+1,   i+1+8,   "col")
    portal.add_member(i+35, i+1+8,   i+1+16,   "col")
    portal.add_member(i+42, i+1+16,   i+1+24,   "col")
    # BRACOS
    portal.add_truss(i+49, i+8,   i+17,   "bra")
    portal.add_truss(i+56, i+9,   i+16,   "bra")

portal.add_restraint(1,  *(True, True, True))
portal.add_restraint(9,  *(True, True, True))
portal.add_restraint(17, *(True, True, True))
portal.add_restraint(25, *(True, True, True))

portal.add_load_pattern("Live Load")
portal.add_self_weight("Live Load")
for i in range(1, 9):
    portal.add_point_load(i, "Live Load", fx=i)
portal.solve()
portal.show()
# portal.plot_model("Live Load")
