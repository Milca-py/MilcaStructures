from milcapy import SystemMilcaModel
# --------------------------------------------------
# 1. Definición del modelo y secciones
# --------------------------------------------------

model = SystemMilcaModel()

model.add_material(
    name="concreto",
    modulus_elasticity=2.1e6,
    poisson_ratio=0.2
)

model.add_rectangular_section(
    name="V30x40",
    material_name="concreto",
    base=0.3,
    height=0.4
)
model.add_rectangular_section(
    name="C40x40",
    material_name="concreto",
    base=0.4,
    height=0.4
)

# --------------------------------------------------
# 2. Definición de nodos
# --------------------------------------------------
b = 7 # bahía
h = 4 # altura

nodes = {}
cont = 1
for i in range(0, 11):
    for j in range(0, 5):
        nodes[cont] = (i*b, j*h)
        cont += 1

for i in range(3, 8):
    for j in range(5, 15):
        nodes[cont] = (i*b, j*h)
        cont += 1

members = {}
cont = 1
# columnas:
for i in range(1, 55):
    if i % 5 == 0:
        continue
    members[cont] = (i, i+1, "C40x40")
    cont += 1
# Vigas:
for i in range(1, 52):
    if i == 1 or (i+4) % 5 == 0:
        continue
    members[cont] = (i, i+5, "V30x40")
    cont += 1
# columnas parte superior
for i in range(56, 105):
    if (i-65) % 10 == 0 or i == 65:
        continue
    members[cont] = (i, i+1, "C40x40")
    cont += 1
# vigas parte superior
for i in range(56, 96):
    members[cont] = (i, i+10, "V30x40")
    cont += 1
# conesctar subestructura
sup = [56, 66, 76, 86, 96]
inf = [20, 25, 30, 35, 40]
for i in range(5):
    members[cont] = (sup[i], inf[i], "C40x40")
    cont += 1

for key, value in nodes.items():
    model.add_node(key, *value)

for key, value in members.items():
    model.add_member(key, *value)

for i in range(1, 56, 5):
    model.add_restraint(i, (True, True, True))

model.add_load_pattern(name="Live Load")
for i in range(1, 6):
    model.add_point_load(i, "Live Load", round(0.29*i, 2), 0, 0, "GLOBAL")

for i in range(56, 66):
    model.add_point_load(i, "Live Load", round(0.29*(i-50), 2), 0, 0, "GLOBAL")

model.add_load_pattern(name="own Weight")
for el in model.members.keys():
    model.add_distributed_load(el, "own Weight", 4, 4, "GLOBAL", "GRAVITY")

model.postprocessing_options.n=17
model.solve()




# obtencion de la derivas
pisos = [i for i in range(1, 6)] + [i+56 for i in range(10)]
desp = []
for i in pisos:
    desp.append(float(model.results["Live Load"].get_node_displacements(i)[0]))
derivas = [(abs(desp[i+1] - desp[i]))/h for i in range(len(desp)-1)]
derivas.insert(0, 0)
print(desp)
import matplotlib.pyplot as plt
# plotear derivas
fig, ax = plt.subplots(figsize=(4.5, 7))
ax.plot(derivas, range(0, 15), 'r.-')
ax.vlines(0.007, 0, 14, colors='b', linestyles='dashed')
ax.text(0.0069, 1, "Δmax = 0.007", fontsize=10, fontweight='bold', color='b', rotation=90, ha='right', va='bottom')
ax.set_xlabel("Pisos")
ax.set_ylabel("Deriva (m/m)")
ax.set_title("Derivas del portico", fontsize=12, fontweight='bold')
plt.tight_layout()
# plt.show()



model.show()

