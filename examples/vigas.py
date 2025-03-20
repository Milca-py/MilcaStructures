from milcapy import SystemMilcaModel
from milcapy import create_plot_window
# UNIDADES: todo en (kN, m) y derivadas

sys = SystemMilcaModel()
sys.add_material("concreto", 2.2e7, 0.3)
sys.add_rectangular_section("sec1", "concreto", 0.5, 0.5)


# parameters:
l = 5
n_tramos = 3
q = 50
for i in range(1, n_tramos + 2):
    sys.add_node(i, (i*l, 0))
    sys.add_restraint(i, (True, True, False))


sys.add_load_pattern("CARGA")
for i in range(1, n_tramos + 1):
    sys.add_element(i, i, i + 1, "sec1")
    sys.add_distributed_load(i, "CARGA", "GLOBAL", q, q, direction="GRAVITY")


sys.add_restraint(1, (True, True, True))
sys.add_restraint(n_tramos+1, (True, True, True))

# --------------------------------------------------
# 5. Resolución del modelo
# --------------------------------------------------
sys.postprocessing_options.n = 40
sys.solve()

# --------------------------------------------------
# 6. Mostrar la estructura (opcional)
# --------------------------------------------------
sys.inicialize_plotter()
sys.plotter_options.internal_forces_scale = 0.003
sys.plotter_options.deformation_scale = 1000
sys.plotter.set_load_pattern_name("CARGA")
# sys.plotter.plot_elements()
# sys.plotter.plot_supports()
# sys.plotter.plot_nodes()
# sys.plotter.plot_element_labels()
# sys.plotter.plot_point_loads()
sys.plotter.plot_distributed_loads()
# sys.plotter.plot_axial_force()
# sys.plotter.plot_shear_force()
sys.plotter.plot_bending_moment()
# sys.plotter.plot_slope()
sys.plotter.plot_deformed()
sys.plotter.show()
