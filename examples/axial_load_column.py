def run():
    from core.system import SystemMilcaModel
    from frontend.widgets.UIdisplay import create_plot_window


    model = SystemMilcaModel()

    model.add_material(
        name="concreto",
        modulus_elasticity=2.1e6,
        poisson_ratio=0.2
    )

    model.add_rectangular_section(
        name="seccion",
        material_name="concreto",
        base=0.3,
        height=0.5
    )

    nodes = {
        1: (0, 0),
        2: (0, 5),
    }
    for key, value in nodes.items():
        model.add_node(key, value)

    elements = {
        1: (1, 2, "seccion"),

    }

    for key, value in elements.items():
        model.add_element(key, *value)

    model.add_restraint(1, (True, True, True))

    model.add_load_pattern(name="Live Load")
    model.add_point_load(2, "Live Load", "GLOBAL", 100, -10000, 0)
    
    # model.add_distributed_load(element_id=1, load_pattern_name="Live Load",
    #                         CSys="LOCAL", load_start= -100, load_end=-50,
    #                         replace=False, direction="LOCAL_2", load_type="FORCE")
    
    model.add_distributed_load(element_id=1, load_pattern_name="Live Load",
                            CSys="LOCAL", load_start= -100, load_end=-50,
                            replace=False, direction="LOCAL_1", load_type="FORCE")

    model.solve()

    print(model.element_map[1].distributed_load.components)
    scale = 1
    model.show_structure(show=False)
    model.plotter.show_diagrams(type="axial_force", escala=0.00003, show=False)
    # # model.plotter.show_diagrams(type="shear_force", escala=0.003, show=False)
    # # model.plotter.show_diagrams(type="bending_moment", escala=0.0003, show=False)
    model.plotter.show_deformed(escala=scale, show=False)
    model.plotter.show_rigid_deformed(escala=scale, show=False)

    root = create_plot_window(model.plotter.fig)
    root.mainloop()

