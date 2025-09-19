from numpy.matlib import True_
from milcapy import SystemModel, BeamTheoriesType, model_viewer
from examples.frames.param_portal import PortalParametric2D


def portal_parametric(n_bahias: int, n_pisos: int, l_bahia: float, h_piso: float, braces: list[int]) -> None:
    portal = PortalParametric2D(n_bahias, n_pisos, l_bahia, h_piso)
    nodes = portal.nodes()
    members = portal.members()
    model = SystemModel()
    secVig = [0.3, 0.5]
    secCol = [0.5, 0.5]
    secBra = [0.1, 0.1]
    matConc = [2.24e6, 0.2, 2.4]
    matBra = [2.04e7, 0.3, 7.8]
    model.add_material("concreto", *matConc)
    model.add_material("acero", *matBra)
    model.add_rectangular_section("vig", "concreto", *secVig)
    model.add_rectangular_section("col", "concreto", *secCol)
    model.add_rectangular_section("bra", "acero", *secBra)
    model.add_load_pattern("Dead Load")
    model.add_self_weight("Dead Load")
    for node_id, (x, y) in nodes.items():
        model.add_node(node_id, x, y)
    for member_id, (node_i, node_j) in members["viga"].items():
        model.add_member(member_id, node_i, node_j, "vig")
    for member_id, (node_i, node_j) in members["columna"].items():
        model.add_member(member_id, node_i, node_j, "col")
        # model.add_distributed_load(member_id, "Dead Load", )
    for i, bay in enumerate(braces):
        for member_id, (node_i, node_j) in portal.braces(bay).items():
            ID_ELE = member_id + (i + 1) * len(model.members) + 1
            model.add_member(ID_ELE, node_i, node_j, "bra",
                                beam_theory=BeamTheoriesType.EULER_BERNOULLI)
            model.add_releases(ID_ELE, mi=True, mj=True)
            model.set_no_weight("Dead Load", ID_ELE)
    for node_id in portal.nodes_from_story(0):
        model.add_restraint(node_id, True, True, True)
    # model.add_self_weight("Dead Load")
    for i, node_id in enumerate(portal.nodes_for_bay(0)):
        if i == 0:   # saltarse la primera
            continue
        model.add_point_load(node_id, "Dead Load", fx=1*node_id)
    model.solve()
    model.plotter_options.mod_support_size = 1.5
    model.plotter_options.internal_forces_label=True
    model_viewer(model)

portal_parametric(3, 5, 7, 5, [2])
