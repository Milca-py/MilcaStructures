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
    name="seccion1",
    material_name="concreto",
    base=0.3,
    height=0.5
)
model.add_rectangular_section(
    name="seccion2",
    material_name="concreto",
    base=0.5,
    height=0.5
)
model.add_rectangular_section(
    name="seccion3",
    material_name="concreto",
    base=0.6,
    height=0.6
)

# --------------------------------------------------
# 2. Definición de nodos
# --------------------------------------------------
nodes = {
    1: (0, 0),
    2: (0, 5),
    3: (0, 8.5),
    4: (0, 12),
    5: (7, 0),
    6: (7, 5),
    7: (7, 8.5),
    8: (7, 12),
}

for key, value in nodes.items():
    model.add_node(key, *value)

# --------------------------------------------------
# 3. Definición de elementos
# --------------------------------------------------
elements = {
    1: (1, 2, "seccion3"),
    2: (2, 3, "seccion3"),
    3: (3, 4, "seccion3"),
    4: (5, 6, "seccion2"),
    5: (6, 7, "seccion2"),
    6: (7, 8, "seccion2"),
    7: (2, 6, "seccion1"),
    8: (3, 7, "seccion1"),
    9: (4, 8, "seccion1"),
}

for key, value in elements.items():
    model.add_member(key, *value)

# --------------------------------------------------
# 4. Restricciones y cargas
# --------------------------------------------------
model.add_restraint(1, (True, True, True))
model.add_restraint(5, (True, True, True))

model.add_load_pattern(name="Live Load")
model.add_point_load(2, "Live Load", 5, 0, 0, "GLOBAL")
model.add_point_load(3, "Live Load", 10, 0, 0, "GLOBAL")
model.add_point_load(4, "Live Load", 20, 0, 0, "GLOBAL")

model.add_distributed_load(7, "Live Load", -5, -5, "LOCAL")
model.add_distributed_load(8, "Live Load", -2, -6, "LOCAL")
model.add_distributed_load(9, "Live Load", -4, -3, "LOCAL")

model.add_load_pattern(name="Live")
model.add_point_load(2, "Live", -5, 0, 0, "GLOBAL")
model.add_point_load(3, "Live", -10, 0, 0, "GLOBAL")
model.add_point_load(4, "Live", -20, 0, 0, "GLOBAL")

model.add_distributed_load(7, "Live", 5, 5, "LOCAL")
model.add_distributed_load(8, "Live", 2, 6, "LOCAL")
model.add_distributed_load(9, "Live", 4, 3, "LOCAL")

model.add_load_pattern(name="Dead Load")
for i in range(1, 10):
    model.add_distributed_load(i, "Dead Load", 25, 25, "GLOBAL", direction="GRAVITY")

model.add_load_pattern(name="self Dead")
for i in range(1, 10):
    model.add_distributed_load(i, "self Dead", -25, -25, "GLOBAL", direction="GRAVITY")


model.add_load_pattern(name="SismoX +")
for i in range(2, 5):
    model.add_point_load(i, "SismoX +", 5*i, 0, 0, "GLOBAL")
    model.add_distributed_load(5+i, "SismoX +", 5, 5, "GLOBAL", direction="GRAVITY")

model.add_load_pattern(name="SismoX -")
for i in range(6, 9):
    model.add_point_load(i, "SismoX -", 20-5*i, 0, 0, "GLOBAL")
    model.add_distributed_load(1+i, "SismoX -", 5, 5, "GLOBAL", direction="GRAVITY")

model.add_load_pattern(name="Peso Propio")
for i in range(1, 10):
    model.add_distributed_load(i, "Peso Propio", 25, 25, "GLOBAL", direction="GRAVITY")
# --------------------------------------------------
# 5. Resolución del modelo
# --------------------------------------------------
model.solve()

# calculo de la envolvente [1.4LL + 1.6*L + 0.6*DL + 0.6*SL]
mpt1 = 1.4 * model.results["Live Load"].get_member_bending_moment(7)
mpt2 = 1.6 * model.results["Live"].get_member_bending_moment(7)
mpt3 = 0.6 * model.results["Dead Load"].get_member_bending_moment(7)
mpt4 = 0.6 * model.results["self Dead"].get_member_bending_moment(7)
# conm 2 (1.4Sx(+) + 1.4Sx(-) + 0.4Pp)
Sx = 1.4 * model.results["SismoX +"].get_member_bending_moment(7)
Sy = 1.4 * model.results["SismoX -"].get_member_bending_moment(7)
pp = 0.4 * model.results["Peso Propio"].get_member_bending_moment(7)
envelop1 = [[], []]
envelop2 = [[], []]
for m1, m2, m3, m4 in zip(mpt1, mpt2, mpt3, mpt4):
    envelop1[0].append(max(m1, m2, m3, m4))
    envelop1[1].append(min(m1, m2, m3, m4))

for m1, m2, m3 in zip(Sx, Sy, pp):
    envelop2[0].append(max(m1, m2, m3))
    envelop2[1].append(min(m1, m2, m3))

import numpy as np
x = np.linspace(0, model.members[7].length(), len(envelop1[0]))
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].set_title("1.4LL + 1.6L + 0.6DL + 0.6SL", fontsize=12, fontweight='bold')
axs[1].set_title("1.4Sx(+) + 1.4Sx(-) + 0.4Pp", fontsize=12, fontweight='bold')
axs[0].plot(x, envelop1[0], color='blue', label='Máximo')
axs[0].plot(x, envelop1[1], color='blue', label='Mínimo')
axs[0].fill_between(x, envelop1[1], envelop1[0], color='#badfca', alpha=0.5)

axs[1].plot(x, envelop2[0], color='red', label='Máximo')
axs[1].plot(x, envelop2[1], color='red', label='Mínimo')
axs[1].fill_between(x, envelop2[1], envelop2[0], color='#f59292', alpha=0.5)
# ajustar separacion dentre los axes:
axs[0].set_ylabel("Momento (kNm)")
axs[1].set_ylabel("Momento (kNm)")
axs[1].set_xlabel("Longitud (m)")

# plt.show()

# --------------------------------------------------
# 6. Mostrar la ventana con la figura
# --------------------------------------------------
model.show()
