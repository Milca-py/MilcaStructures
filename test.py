
from core.system import SystemMilcaModel
from frontend.widgets.UIdisplay import create_plot_window
from frontend.widgets.display_array import mostrar_array


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
}

for key, value in elements.items():
    model.add_element(key, "FRAME", *value)

model.add_restraint(1, (True, True, True))
model.add_restraint(5, (True, True, True))

model.add_load_pattern(name="Live Load")
model.add_point_load(2, "Live Load", "GLOBAL", 5, 0, 0)
model.add_point_load(3, "Live Load", "GLOBAL", 10, 0, 0)
model.add_point_load(4, "Live Load", "GLOBAL", 20, 0, 0)

model.add_distributed_load(7, "Live Load", "LOCAL", -5, -5)
model.add_distributed_load(8, "Live Load", "LOCAL", 2, -6)
model.add_distributed_load(9, "Live Load", "LOCAL", -4, -3)

model.solve()

model.show_structure(show=False)
root = create_plot_window(model.plotter.fig)
root.mainloop()


k_global = model.global_stiffness_matrix











# array = k_global
# app = mostrar_array(array, zero_threshold=True, round_values=True, decimal_places=0)



# # crear el modelo estructural
# model = SystemMilcaModel()


# E = 30
# v = 0.15
# g = 0
# b = 40
# h = 50

# B = 500
# H = 300
# Hh = 100

# # agregar materiales
# model.add_material(name="concreto", modulus_elasticity=E,
#                    poisson_ratio=v, specific_weight=g)

# # # agregar secciones
# model.add_rectangular_section("seccion1", "concreto", b, h)

# # # agregar nodos
# model.add_node(1, (0, 0))
# model.add_node(2, (0, H))
# model.add_node(3, (B/2, H+Hh))
# model.add_node(4, (B, H))
# model.add_node(5, (B, 0))

# model.add_node(6, (B + B/2, H + Hh))
# model.add_node(7, (B + B, H))
# model.add_node(8, (B + B, 0))


# # # agregar elementos
# model.add_element(1, "FRAME", 1, 2, "seccion1")
# model.add_element(2, "FRAME", 2, 3, "seccion1")
# model.add_element(3, "FRAME", 3, 4, "seccion1")
# model.add_element(4, "FRAME", 4, 5, "seccion1")

# model.add_element(5, "FRAME", 4, 6, "seccion1")
# model.add_element(6, "FRAME", 6, 7, "seccion1")
# model.add_element(7, "FRAME", 7, 8, "seccion1")

# # # agregar restricciones
# model.add_restraint(1, (True, True, True))
# model.add_restraint(2, (True, False, False))
# model.add_restraint(5, (True, True, False))
# model.add_restraint(8, (False, False, True))

# # # agregar patrones de carga
# model.add_load_pattern(name="muerta")

# # agregar cargas puntuales
# model.add_point_load(6, "muerta", "GLOBAL", 50, 50, 50)
# model.add_point_load(7, "muerta", "LOCAL", 50, 50, 50, angle_rot=0)
# # agregar cargas distribuidas
# model.add_distributed_load(2, "muerta", "LOCAL",-50, -50)
# model.add_distributed_load(3, "muerta", "LOCAL", -50, -50)
# # model.add_distributed_load(5, "muerta", "GLOBAL", 50, 50, direction="GRAVITY")
# # model.add_distributed_load(6, "muerta", "GLOBAL", 50, 50, direction="GRAVITY_PROJ")

# # resolver el modelo
# model.solve()



# # # Mostrar la estructura sin mostrar la figura inmediatamente
# model.show_structure(show=False)
# # for ax in model.plotter.fig.axes:
# #     ax.axis("off")  # Ocultar ejes
# # model.show_structure()


# # Mostrar la estructura en frontend
# root = create_plot_window(model.plotter.fig)
# root.mainloop()