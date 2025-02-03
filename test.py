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
#     Node(2,Vertex(4, 3)),
#     RectangularSection("seccion1", ConcreteMaterial("concreto", 210e9, 0.3, 25e3), 0.3, 0.6)
#     )


# print(el.length)
# print(el.angle_x_axis*180/3.1416)
# # print(el.node_i.vertex.coordinates)
# # print(el.node_j.vertex.coordinates)
# # print(el.section.area)
# # print(el.node_idi)
# # print(el.node_idj)
# # print(el.node_map)
# # print(el.dof_map)
# # print(el.stiffness_matrix)

# # print(el.distributed_load.to_dict())

# el.compile_stiffness_matrix()
# # print(el._stiffness_matrix)

# el.compile_transformation_matrix()
# # print(el._transformation_matrix)

# el.add_distributed_load(DistributedLoad(20, 20))
# # print(el._distributed_load.to_dict())
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

sys = SystemMilcaModel()












