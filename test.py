
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
    # 9: (3.5, 15)
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
    # 10: (4, 9, "seccion1"),
    # 11: (8, 9, "seccion1")
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
# model.plotter.show_diagrams(type="spin", show=False)
# model.plotter.show_deformed(escala=10, show=False)



# reactions = model.reactions

# # mostrar_array(reactions, True, True, 7)





# root = create_plot_window(model.plotter.fig)
# root.mainloop()


from core.post_processing import PostProcessing, values_slope

post = PostProcessing(model)

element = model.element_map[7]

print(element.integration_coefficients[0])
print(element.bending_moment)
print(values_slope(element, 1, 40)[1])

# print(model.analysis.options.status)
# print(model.post_processing.integration_coefficients_elements)
# print(model.post_processing.integration_coefficients_elements)
# print(model.element_map[1].deflection)















# from frontend.widgets.widgets import InternalForceDiagramWidget, DiagramConfig
# from core.post_processing import values_axial_force, values_shear_force, values_bending_moment, values_spin, values_deflection
# import numpy as np



# nelem = 7
# element = model.element_map[nelem]
# factor = 1
# npp = 40
# N = values_axial_force(element, factor, npp)
# V = values_shear_force(element, factor, npp)
# M = values_bending_moment(element, factor, npp)
# θ = values_spin(element, factor, npp)
# y = values_deflection(element, factor, npp)
# x = np.linspace(0, element.length, npp)

## mostrar en una widget
# diagrams = {
#     'N(x)': DiagramConfig(
#         name='Diagrama de Fuerza Normal',
#         values=N[1],
#         units='tonf',
#     ),
#     'V(x)': DiagramConfig(
#         name='Diagrama de Fuerza Cortante',
#         values=V[1],
#         units='tonf',
#     ),
#     'M(x)': DiagramConfig(
#         name='Diagrama de Momento Flector',
#         values=M[1],
#         units='tonf-m',

#     ),
# 'θ(x)': DiagramConfig(
#         name='Diagrama de Rotación',
#         values=θ[1],
#         units='rad',
#         precision=5

#     ),
#     'y(x)': DiagramConfig(
#         name='Diagrama de Deflexión',
#         values=y[1],
#         units='cm',
#         precision=5

#     )
# }

# app = InternalForceDiagramWidget(nelem, x, diagrams, grafigcalor=True, cmap='jet')

