def run():
    from milcapy.elements.system import SystemMilcaModel
    from milcapy.frontend.widgets.UIdisplay import create_plot_window
    # UNIDADES: todo en (kN, m) y derivadas

    sys = SystemMilcaModel()
    sys.add_material("concreto", 2.2e7, 0.3)
    sys.add_rectangular_section("sec1", "concreto", 0.5, 0.5)


    # parameters:
    l = 5
    n_tramos = 4
    q = 50
    for i in range(1, n_tramos + 2):
        sys.add_node(i, (i*l, 0))
        sys.add_restraint(i, (True, True, False))


    sys.add_load_pattern("CARGA")
    for i in range(1, n_tramos + 1):
        sys.add_element(i, i, i + 1, "sec1")
        sys.add_distributed_load(i, "CARGA", "GLOBAL", q, q, direction="GRAVITY")


    # sys.solve()
    sys.node_map[2].restraints = (False, False, False)
    # sys.global_stiffness_matrix[4, 4] += 10000000
    # sys.global_stiffness_matrix[5, 5] += 10000000
    # sys.global_stiffness_matrix[6, 6] += 10000000
    sys.add_point_load(2, "CARGA", "GLOBAL", fy=-100)
    sys.solve()
    sys.plotter.plot_structure(labels_distributed_loads=True, show=False, labels_point_loads=True)
    # sys.plotter.show_diagrams("axial_force", show=False)
    # sys.plotter.show_diagrams("shear_force", show=False, escala=0.003)
    sys.plotter.show_diagrams("bending_moment", show=False, escala=0.001)
    # sys.plotter.show_diagrams("slope", show=False, escala=4000)
    # sys.plotter.show_diagrams("deflection", show=False, escala=4000, fill=False)
    sys.plotter.show_deformed(10, show=False)
    # sys.plotter.show_rigid_deformed(1000, show=False)

    root = create_plot_window(sys.plotter.fig)
    root.mainloop()