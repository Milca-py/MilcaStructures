# ==============================
# =====    PointLoad    ======
# ==============================

# from core.node import Node
# from utils.vertex import Vertex
# from loads.load import PointLoad

# n = Node(4, Vertex(0, 0))


# n.add_forces(PointLoad(100, 0, 0))
# n.add_forces(PointLoad(0, -100, 0))
# n.add_forces(PointLoad(0, 50, 100))
# n.add_restraints((True, True, True))



# # print(n.dof)
# # print(n.restraints)
# # print(n.vertex.coordinates[0])

# ==============================
# =====    Vertex    ======
# ==============================
# from utils.vertex import Vertex
# v1 = Vertex(0, 0)
# v2 = Vertex(0, 0)
# v3 = Vertex(3, 4)
# print(
#     v3.coordinates,
#     v3.modulus,
#     v3.unit,
#     v3.__add__(v1),
# )


# ==============================
# =====    LoadPattern    ======
# ==============================
# from loads import LoadPattern
# from loads import PointLoad


# lp = LoadPattern("muerta")

# lp.add_point_load(1, PointLoad(100, 0, 0))
# lp.add_point_load(1, PointLoad(0, -100, 0), replace=True)

# print(lp.point_loads_map)


# ==============================
# =====      MATERIAL     ======
# ==============================
# from core.material import ConcreteMaterial


# concreto = ConcreteMaterial(
#     name="Concreto",
#     modulus_elasticity=210e9,
#     poisson_ratio=0.3,
#     specific_weight=25e3
# )


# print(concreto)
# print(concreto.shear_modulus)


# ==============================
# =====       SECTION     ======
# ==============================

# from core.section import RectangularSection

# Vig30x60 = RectangularSection(
#     name="Viga 30x60",
#     material=concreto,
#     base=0.5,
#     height=0.1
# )



# print(Vig30x60)
# print(Vig30x60.area)
# print(Vig30x60.moment_of_inertia)
# print(Vig30x60.shear_coefficient)





# ==============================
# ===== Carga distribuida ======
# ==============================

# from loads import DistributedLoad

# w = DistributedLoad(20, 40)


# print(w*2)
# print(w/2)
# print(2*w)



# ==============================
# ===== Elemento de viga ======
# ==============================

# from core.element import Element
# from core.node import Node
# from utils import ElementType, Vertex
# from core.section import RectangularSection
# from core.material import ConcreteMaterial
# from loads import DistributedLoad

# el = Element(
#     1,
#     ElementType.FRAME,
#     Node(1,Vertex(0, 0)),
#     Node(2,Vertex(400, 300)),
#     RectangularSection("seccion1", ConcreteMaterial("concreto", 30, 0.15, 2400), 40, 50)
#     )


# print(el.length)
# print(el.angle_x_axis*180/3.1416)
# print(el.node_i.vertex.coordinates)
# print(el.node_j.vertex.coordinates)
# print(el.section.area)
# print(el.node_idi)
# print(el.node_idj)
# print(el.node_map)
# print(el.dof_map)
# print(el.stiffness_matrix)

# print(el.distributed_load.to_dict())

# el.compile_stiffness_matrix()
# print(el._stiffness_matrix)

# import pandas as pd
# pd = pd.DataFrame(el._stiffness_matrix)
# pd.to_excel("stiffness_matrix.xlsx")

# el.compile_transformation_matrix()
# print(el._transformation_matrix)

# el.add_distributed_load(DistributedLoad(20, 20))
# print(el._distributed_load.to_dict())
# el.compile_load_vector()
# print(el._load_vector)

# el.compile_stiffness_matrix_global()
# print(el.stiffness_matrix_global)


# el.compile_load_vector_global()
# print(el.load_vector_global)



# print(el._stiffness_matrix)




# ==============================
# =====       MODELO       =====
# ==============================


from core.system import SystemMilcaModel
import pandas as pd
# from main import SytemMilcaModel

# crear el modelo estructural
model = SystemMilcaModel()


E = 30
v = 0.15
g = 0
b = 40
h = 50


# agregar materiales
model.add_material(name="concreto", modulus_elasticity=E,
                   poisson_ratio=v, specific_weight=g)

# # agregar secciones
model.add_rectangular_section("seccion1", "concreto", b, h)

# # agregar nodos
model.add_node(1, (0, 0))
model.add_node(2, (0, 300))
model.add_node(3, (250, 400))
model.add_node(4, (500, 300))
model.add_node(5, (500, 0))


# # agregar elementos
model.add_element(1, "FRAME", 1, 2, "seccion1")
model.add_element(2, "FRAME", 2, 3, "seccion1")
model.add_element(3, "FRAME", 3, 4, "seccion1")
model.add_element(4, "FRAME", 4, 5, "seccion1")

# # agregar restricciones
model.add_restraint(1, (True, True, True))
model.add_restraint(4, (True, True, True))

# # agregar patrones de carga
model.add_load_pattern(name="muerta")

# agregar cargas puntuales
# model.add_point_load(2, "muerta", fx=100)
model.add_distributed_load(2, "muerta", "LOCAL", -50, -50)
model.add_distributed_load(3, "muerta", "LOCAL", -50, -50)

# resolver el modelo
model.solve()



# ==============================
# =====     PLOTTER       =====
# ==============================

from display.ploter import Plotter
plotter = Plotter(model)

plotter.plot_structure()





