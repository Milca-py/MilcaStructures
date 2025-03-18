from milcapy.elements.system import SystemMilcaModel

model = SystemMilcaModel()

model.add_material("concreto", 25e6, 0.2)
model.add_rectangular_section("section", "concreto", 0.3, 0.3)

# parameters
bahia = 5
altura = 3

# nodes
nodes = {
    1: [0, 0],
    2: [0, altura],
    3: [0, 2*altura],
    4: [0, 3*altura],
    5: [bahia, 0],
    6: [bahia, altura],
    7: [bahia, 2*altura],
    8: [bahia, 3*altura],
    9: [2*bahia, 0],
    10: [2*bahia, altura],
    11: [2*bahia, 2*altura],
    12: [2*bahia, 3*altura],
    # 13: [bahia, 6*altura],
}

for node, coord in nodes.items():
    model.add_node(node, coord)

# elements
elements = {
    # columnas
    1: [1, 2],
    2: [2, 3],
    3: [3, 4],
    4: [5, 6],
    5: [6, 7],
    6: [7, 8],
    7: [9, 10],
    8: [10, 11],
    9: [11, 12],
    # vigas
    10: [2, 6],
    11: [6, 10],
    12: [3, 7],
    13: [7, 11],
    14: [4, 8],
    15: [8, 12],
    # arriostramientos
    # 16: [1, 6],
    # 17: [2, 5],
    # 18: [5, 10],
    # 19: [6, 9],
    # 20: [2, 7],
    # 21: [3, 6],
    # 22: [6, 11],
    # 23: [7, 10],
    # 24: [3, 8],
    # 25: [4, 7],
    # 26: [7, 12],
    # 27: [8, 11],
    # aguas
    # 28: [4, 13],
    # 29: [13, 12],
}

for element, nodes in elements.items():
    model.add_element(element, *nodes, "section")

# constraints
model.add_restraint(1, (True, True, True))
model.add_restraint(5, (True, True, True))
model.add_restraint(9, (True, True, True))

# loads
model.add_load_pattern("CARGA")
# model.add_point_load(2, "CARGA", fx=100)
# model.add_point_load(3, "CARGA", fx=500)
# model.add_point_load(4, "CARGA", fx=1000)
# model.add_point_load(13, "CARGA", fy=-10000)
model.add_distributed_load(10,"CARGA", load_start=-50, load_end=-50)
model.add_distributed_load(11,"CARGA", load_start=-50, load_end=-50)
model.add_distributed_load(12,"CARGA", load_start=-50, load_end=-50)
model.add_distributed_load(13,"CARGA", load_start=-50, load_end=-50)
model.add_distributed_load(14,"CARGA", load_start=-50, load_end=-50)
model.add_distributed_load(15,"CARGA", load_start=-50, load_end=-50)

# analysis
model.postprocessing_options.n = 40
model.solve()



model.inicialize_plotter()
model.plotter_options.internal_forces_scale = 0.01
model.plotter.set_load_pattern_name("CARGA")
# model.plotter.plot_nodes()
model.plotter.plot_elements()
model.plotter.plot_supports()
# model.plotter.plot_node_labels()
# model.plotter.plot_element_labels()
model.plotter.plot_point_loads()
model.plotter.plot_distributed_loads()
# model.plotter.plot_axial_force()
# model.plotter.plot_shear_force()
model.plotter.plot_bending_moment()
# model.plotter.plot_slope()
# model.plotter.plot_deflection()
# model.plotter.plot_deformed()
# model.plotter.plot_rigid_deformed()
# model.plotter.plot_structure()
model.plotter.show()
