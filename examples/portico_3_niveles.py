from milcapy import SystemModel
# --------------------------------------------------
# 1. Definición del modelo y secciones
# --------------------------------------------------

model = SystemModel()

model.add_material(
    name="concreto",
    modulus_elasticity=2.1e6,
    poisson_ratio=0.2
)
# model.add_material(
#     name="rigid",
#     modulus_elasticity=99999999999,
#     poisson_ratio=0.2
# )

# model.add_rectangular_section(
#     name="rigid",
#     material_name="rigid",
#     base=30,
#     height=5
# )
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

    # 9: (0.3, 5),
    # 10: (0.3, 8.5),
    # 11: (0.3, 12),
    # 12: (6.75, 5),
    # 13: (6.75, 8.5),
    # 14: (6.75, 12),
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
    # 7: (9, 12, "seccion1"),
    # 8: (10, 13, "seccion1"),
    # 9: (11, 14, "seccion1"),

    # 10: (2, 9, "rigid"),
    # 11: (3, 10, "rigid"),
    # 12: (4, 11, "rigid"),
    # 13: (12, 6, "rigid"),
    # 14: (13, 7, "rigid"),
    # 15: (14, 8, "rigid"),
}

for key, value in elements.items():
    model.add_member(key, *value)


# --------------------------------------------------
# 4. Restricciones y cargas
# --------------------------------------------------
model.add_restraint(1, (True, True, True))
model.add_restraint(5, (True, True, True))

model.add_load_pattern(name="Load")
model.add_point_load(2, "Load", 50, 0, 0, "GLOBAL")

model.add_load_pattern(name="Live Load")
model.add_point_load(2, "Live Load", 5, 0, 0, "GLOBAL")
model.add_point_load(3, "Live Load", 10, 0, 0, "GLOBAL")
model.add_point_load(4, "Live Load", 20, 0, 0, "GLOBAL")

model.add_distributed_load(7, "Live Load", -5, -5) ###
model.add_distributed_load(8, "Live Load", -2, -6) ###
model.add_distributed_load(9, "Live Load", -4, -3) ###

# qi = -5
# qj = -5
# def qx(x): return (qj-qi)*x/7 +  qi
# qa = qx(0.3)
# qb = qx(6.75)
# model.add_distributed_load(7, "Live Load", qa, qb) ###
# model.add_distributed_load(10, "Live Load", qi, qa)
# model.add_distributed_load(13, "Live Load", qb, qj)

# qi = -2
# qj = -6
# def qx(x): return (qj-qi)*x/7 +  qi
# qa = qx(0.3)
# qb = qx(6.75)
# model.add_distributed_load(8, "Live Load", qa, qb) ###
# model.add_distributed_load(11, "Live Load", qi, qa)
# model.add_distributed_load(14, "Live Load", qb, qj)

# qi = -4
# qj = -3
# def qx(x): return (qj-qi)*x/7 +  qi
# qa = qx(0.3)
# qb = qx(6.75)
# model.add_distributed_load(9, "Live Load", qa, qb) ###
# model.add_distributed_load(12, "Live Load", qi, qa)
# model.add_distributed_load(15, "Live Load", qb, qj)

model.add_end_length(7, 0.3, 0.25)
model.add_end_length(8, 0.3, 0.25)
model.add_end_length(9, 0.3, 0.25)
# model.add_end_length(1, 2, 0)
# model.add_end_length(4, 2, 0)


# --------------------------------------------------
# 5. Resolución del modelo
# --------------------------------------------------


# model.analysis_options.include_shear_deformations = False
# model.postprocessing_options.n = 10
model.solve()


# ====================================================
# print(model.results["Live Load"].get_member_bending_moment(8))
# ====================================================


# --------------------------------------------------
# 6. Mostrar la ventana con la figura
# --------------------------------------------------
model.show()


# node1 = model.nodes[1]
# node2 = model.members[1].node_i

# print(id(node1), id(node2))
# print(node1 is node2)  # True, because they are the same object


# sec1 = model.sections["seccion1"]
# sec2 = model.members[9].section

# print(id(sec1), id(sec2))
# print(sec1 is sec2)  # True, because they are the same object
