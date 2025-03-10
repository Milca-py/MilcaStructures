def run():
    from milcapy.core.system import SystemMilcaModel
    from milcapy.frontend.widgets.UIdisplay import create_plot_window
    # UNIDADES: (kN, m)

    sys = SystemMilcaModel()
    sys.add_material("concreto", 2.2e7, 0.3)
    sys.add_rectangular_section("sec1", "concreto", 0.5, 0.5)
    sys.add_rectangular_section("sec2", "concreto", 0.3, 0.65)

    nodes = {
        1: (0, 0),
        2: (0, 4),
        3: (4, 6),
        4: (8, 4),
        5: (8, 0)
    }

    for node, coord in nodes.items():
        sys.add_node(node, coord)

    elements = {
        1: (1, 2, "sec1"),
        2: (2, 3, "sec2"),
        3: (4, 3, "sec2"),
        4: (5, 4, "sec1")
    }

    for element, (node1, node2, section) in elements.items():
        sys.add_element(element, node1, node2, section)

    sys.add_restraint(1, (True, True, True))
    sys.add_restraint(5, (True, True, True))

    sys.add_load_pattern("CARGA")
    sys.add_distributed_load(2, "CARGA", "LOCAL", -5, -5)
    sys.add_distributed_load(3, "CARGA", "GLOBAL", 7, 7, direction="GRAVITY")

    sys.solve()
    sys.plotter.plot_structure(labels_distributed_loads=True, show=False)
    sys.plotter.show_diagrams("axial_force", show=False)
    # sys.plotter.show_diagrams("shear_force", show=False)
    # sys.plotter.show_diagrams("bending_moment", show=False)
    # sys.plotter.show_diagrams("slope", show=False, escala=4000)
    # sys.plotter.show_diagrams("deflection", show=False, escala=4000, fill=False)
    # sys.plotter.show_deformed(1000, show=False)
    # sys.plotter.show_rigid_deformed(1000, show=False)

    root = create_plot_window(sys.plotter.fig)
    root.mainloop()