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
    sys.add_point_load(3, "CARGA", fx=1000)

    sys.solve()


    dfx = sys.results.global_displacements_nodes[3][0]

    dezplaments_lateral = []
    forces_increment = []
    shear_basal = []

    i = 1
    while abs(dfx) < 0.5:
        sys.add_point_load(3, "CARGA", fx=100*i)
        sys.solve()
        dfx = sys.results.global_displacements_nodes[3][0]
        v_basal = sys.results.reactions[0] + sys.results.reactions[12]
        dezplaments_lateral.append(dfx)
        forces_increment.append(1000*i)
        shear_basal.append(abs(v_basal))
        i += 1



    show = True
    if show == True:
        sys.plotter.show_deformed(1, show=False)
        root = create_plot_window(sys.plotter.fig)
        root.mainloop()


    import matplotlib.pyplot as plt
    # plt.close('all')
    plt.close(1)
    fig, ax = plt.subplots()
    # ax.plot(forces_increment, dezplaments_lateral)
    ax.plot(shear_basal, dezplaments_lateral)
    ax.set(xlabel='Fuerza (kN)', ylabel='Desplazamiento lateral (m)',
        title='Curva de pushover')
    plt.show()