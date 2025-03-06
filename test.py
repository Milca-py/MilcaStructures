
import matplotlib.pyplot as plt
from core.post_processing import (
    values_axial_force, 
    values_shear_force, 
    values_bending_moment,
    values_spin,
    values_deflection
    )
from components.element_components import local_load_vector, load_vector
from core.system import SystemMilcaModel
from frontend.widgets.UIdisplay import create_plot_window
from frontend.widgets.display_array import mostrar_array

from utils import rotate_xy, traslate_xy
import numpy as np

# units are in "tonf" and "m"

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

nodes = {
    1: (0, 0),
    2: (0, 5),
    3: (0, 8.5),
    4: (0, 12),
    5: (7, 0),
    6: (7, 5),
    7: (7, 8.5),
    8: (7, 12),
    9: (3.5, 15)
}

for key, value in nodes.items():
    model.add_node(key, value)

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
    10: (4, 9, "seccion1"),
    11: (8, 9, "seccion1")
}

for key, value in elements.items():
    model.add_element(key, *value)

model.add_restraint(1, (True, True, True))
model.add_restraint(5, (True, True, True))

model.add_load_pattern(name="Live Load")
model.add_point_load(2, "Live Load", "GLOBAL", 5, 0, 0)
model.add_point_load(3, "Live Load", "GLOBAL", 10, 0, 0)
model.add_point_load(4, "Live Load", "GLOBAL", 20, 0, 0)

model.add_distributed_load(7, "Live Load", "LOCAL", -5, -5)
model.add_distributed_load(8, "Live Load", "LOCAL", -2, -6)
model.add_distributed_load(9, "Live Load", "LOCAL", -4, -3)

model.solve()

model.show_structure(show=False)



# reactions = model.reactions

# mostrar_array(reactions, True, True, 7)


# element = model.element_map[1]
for element in model.element_map.values():
    npp = 8
    escala = 100

    # x, n = values_axial_force(element, escala, npp)
    # x, n = values_shear_force(element, escala, npp)
    # x, n = values_bending_moment(element, escala, npp)
    # x, n = values_spin(element, 1, npp)
    x, n = values_deflection(element, escala, npp)

    ax = model.plotter.axes[0]

    coord_elem = np.array([np.array([element.node_i.vertex.coordinates]), 
                        np.array([element.node_j.vertex.coordinates])])

    Nxy = np.column_stack((x, n))

    print(Nxy[:, 0])

    Nxy = rotate_xy(Nxy, element.angle_x, 0, 0)
    Nxy = traslate_xy(Nxy, *element.node_i.vertex.coordinates)

    Nxy = np.insert(Nxy, 0, coord_elem[0], axis=0)
    Nxy = np.append(Nxy, coord_elem[1],axis=0)

    y2 = np.linspace(Nxy[0], Nxy[-1], npp)
    y2 = np.insert(y2, 0, coord_elem[0], axis=0)
    y2 = np.append(y2, coord_elem[1],axis=0)
    y2 = y2[:, 1]

    print(y2)



    ax.plot(Nxy[:, 0], Nxy[:, 1], label="Axial Force", lw=1)
    # ax.plot(Nxy[:, 0], y2, label="Axial Force", lw=1)
    # plt.fill_between(Nxy[:, 0], Nxy[:, 1], y2, color='#f4f1bb', alpha=0.5)

root = create_plot_window(model.plotter.fig)
root.mainloop()




# print(f"""
#     Axial Force:
#     {n}
#     shear force:
#     {q}
#     bending moment:
#     {m}
#     spin:
#     {t}
#     deflection:
#     {u}
#       """)

# fig, axs = plt.subplots(3, 2, figsize=(10, 10))
# axs[0, 0].plot(x, n)
# axs[0, 0].set_title("Axial Force")
# axs[0, 0].set_xlabel("x [m]")
# axs[0, 0].set_ylabel("N [tonf]")
# axs[0, 0].grid()

# axs[0, 1].plot(x, q)
# axs[0, 1].set_title("Shear Force")
# axs[0, 1].set_xlabel("x [m]")
# axs[0, 1].set_ylabel("V [tonf]")
# axs[0, 1].grid()

# axs[1, 0].plot(x, m)
# axs[1, 0].set_title("Bending Moment")
# axs[1, 0].set_xlabel("x [m]")
# axs[1, 0].set_ylabel("M [tonf*m]")
# axs[1, 0].grid()

# axs[1, 1].plot(x, t)
# axs[1, 1].set_title("Spin")
# axs[1, 1].set_xlabel("x [m]")
# axs[1, 1].set_ylabel("theta [rad]")
# axs[1, 1].grid()

# axs[2, 0].plot(x, u)
# axs[2, 0].set_title("Deflection")
# axs[2, 0].set_xlabel("x [m]")
# axs[2, 0].set_ylabel("u [m]")
# axs[2, 0].grid()

# plt.show()



