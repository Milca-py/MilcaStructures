from milcapy import SystemMilcaModel
from milcapy.plotter.UIdisplay import main_window
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
}

for node, coord in nodes.items():
    model.add_node(node, *coord)

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
    16: [1, 6],
    17: [2, 5],
    18: [5, 10],
    19: [6, 9],
    20: [2, 7],
    21: [3, 6],
    22: [6, 11],
    23: [7, 10],
    24: [3, 8],
    25: [4, 7],
    26: [7, 12],
    27: [8, 11],
}

for element, nodes in elements.items():
    model.add_member(element, *nodes, "section")

# constraints
model.add_restraint(1, (True, True, True))
model.add_restraint(5, (True, True, True))
model.add_restraint(9, (True, True, True))

# loads
model.add_load_pattern("CARGA")
model.add_point_load(2, "CARGA", fx=100)
model.add_point_load(3, "CARGA", fx=500)
model.add_point_load(4, "CARGA", fx=1000)
model.add_load_pattern("LOAD")
model.add_distributed_load(10,"LOAD", load_start=-50, load_end=-50)
model.add_distributed_load(11,"LOAD", load_start=-50, load_end=-50)
model.add_distributed_load(12,"LOAD", load_start=-50, load_end=-50)
model.add_distributed_load(13,"LOAD", load_start=-50, load_end=-50)
model.add_distributed_load(14,"LOAD", load_start=-50, load_end=-50)
model.add_distributed_load(15,"LOAD", load_start=-50, load_end=-50)

# analysis
model.postprocessing_options.n = 40
model.solve()


# Mostrar la ventana con la figura
model._inicialize_plotter()
model.plotter.initialize_plot()
main_window(model)
