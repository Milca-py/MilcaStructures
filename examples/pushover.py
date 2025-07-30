from milcapy import SystemMilcaModel
# UNIDADES: (kN, m)

def Hognestad(ec, fc, Ec):
    e0 = 2*fc/Ec
    if ec < e0:
        fc_hg = fc*((2*ec/e0)-(ec/e0)**2)
    elif ec <= 0.0038:
        fc_hg = 0.85*fc+(0.15*fc)*(ec-0.0038)/(e0-0.0038)
    else: fc_hg = 0.01*fc

    return fc_hg

sys = SystemMilcaModel()
sys.add_material("concreto", 2.2e7, 0.3)
sys.add_rectangular_section("sec1", "concreto", 0.5, 0.5)

nodes = {
    1: (0, 0),
    2: (0, 4),
    3: (4, 6),
    4: (8, 4),
    5: (8, 0)
}
for node, coord in nodes.items():
    sys.add_node(node, *coord)

elements = {
    1: (1, 2, "sec1"),
    2: (2, 3, "sec1"),
    3: (4, 3, "sec1"),
    4: (5, 4, "sec1")
}
for element, (node1, node2, section) in elements.items():
    sys.add_member(element, node1, node2, section)

sys.add_restraint(1, (True, True, True))
sys.add_restraint(5, (True, True, True))

sys.add_load_pattern("CARGA")
sys.add_point_load(3, "CARGA", fx=100)
sys.add_point_load(2, "CARGA", fx=100)

sys.solve()

# control de dezplazamiento
dfx = sys.results["CARGA"].get_node_displacements(3)[0]

dezplaments_lateral = []
shear_basal = []

i = 2
while abs(dfx) < 0.5:
    sys.add_point_load(3, "CARGA", fx=0.002*i)
    sys.add_point_load(2, "CARGA", fx=0.001*i)
    sys.solve()
    dfx = sys.results["CARGA"].get_node_displacements(3)[0]
    v_basal = sys.results["CARGA"].get_node_reactions(1)[0] + sys.results["CARGA"].get_node_reactions(5)[0]
    dezplaments_lateral.append(dfx)
    shear_basal.append(v_basal)
    sys.materials["concreto"].E = Hognestad(dfx/1000, 3e6, 2.1e6)
    i += 1





import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(dezplaments_lateral, shear_basal)
ax.set(ylabel='Fuerza (kN)', xlabel='Desplazamiento lateral (m)',
    title='Curva de pushover')
plt.tight_layout()
plt.show()



