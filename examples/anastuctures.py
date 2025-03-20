def run():
    from anastruct.fem.system import SystemElements
    from milcapy.frontend.widgets.UIdisplay import create_plot_window

    EI1 = 2.1e6 * 0.3 * 0.5 ** 3 / 12
    EI2 = 2.1e6 * 0.5 * 0.5 ** 3 / 12
    EI3 = 2.1e6 * 0.6 * 0.6 ** 3 / 12

    ss = SystemElements()

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
    }
    # --------------------------------------------------
    # 3. Definición de elementos
    # --------------------------------------------------
    elements = {
        2: (2, 3, EI3),
        3: (3, 4, EI3),
        1: (1, 2, EI3),
        4: (5, 6, EI2),
        5: (6, 7, EI2),
        6: (7, 8, EI2),
        7: (2, 6, EI1),
        8: (3, 7, EI1),
        9: (4, 8, EI1),
    }

    for value in elements.values():
        ss.add_element(location=[nodes[value[0]], nodes[value[1]]], EI=value[2])

    ss.add_support_fixed(node_id=4)
    ss.add_support_fixed(node_id=5)

    ss.point_load(node_id=3, Fx=5)
    ss.point_load(node_id=2, Fx=10)
    ss.point_load(node_id=1, Fx=20)

    ss.q_load(q=[-5, -5], element_id=7)
    ss.q_load(q=[-2, -6], element_id=8)
    ss.q_load(q=[-4, -3], element_id=9)

    ss.solve()
    ss.show_structure()
    ss.show_axial_force()
    ss.show_displacement()
    ss.show_shear_force()
    ss.show_bending_moment()
    print(ss.element_map[1].deflection)

    fig = ss.plotter.fig
    root = create_plot_window(fig)
    root.mainloop()

if __name__ == "__main__":
    run()