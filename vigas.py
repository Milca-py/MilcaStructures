from milcapy import SystemMilcaModel
# UNIDADES: todo en (kN, m) y derivadas

sys = SystemMilcaModel()
sys.add_material("concreto", 2.2e7, 0.3)
sys.add_rectangular_section("sec1", "concreto", 0.5, 0.5)


# parameters:
l = 6
n_tramos = 4
q = 50
for i in range(1, n_tramos + 2):
    sys.add_node(i, i*l, 0)
    sys.add_restraint(i, (True, True, False))


sys.add_load_pattern("CARGA")
for i in range(1, n_tramos + 1):
    sys.add_member(i, i, i + 1, "sec1")
    sys.add_distributed_load(i, "CARGA", q, q, "GLOBAL", direction="GRAVITY")


# --------------------------------------------------
# 5. Resolución del modelo
# --------------------------------------------------
sys.postprocessing_options.n = 17
sys.solve()
sys.show()
