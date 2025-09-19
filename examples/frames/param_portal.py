from enum import Enum
from milcapy.utils.types import to_enum
from milcapy import SystemModel, BeamTheoriesType, model_viewer


class PortalParametric2D:
    def __init__(self, n_bahias: int, n_pisos: int, l_bahia: float, h_piso: float):
        self.m = n_bahias  # Número de bahías (horizontal)
        self.p = n_pisos  # Número de pisos (vertical)
        self.l = l_bahia  # Longitud de cada bahía
        self.h = h_piso  # Altura de cada piso

    def nodes_from_story(self, story: int) -> list[int]:
        """Devuelve los nodos de un piso específico"""
        nodes = [x for x in range(
            story * (self.m + 1) + 1, (story + 1) * (self.m + 1) + 1)]
        return nodes

    def nodes_for_bay(self, bay: int) -> list[int]:
        """Devuelve todos los nodos de una columna específica (vertical)"""
        if bay < 0 or bay > self.m:
            raise ValueError(
                f"La columna {bay} no existe. Debe estar entre 0 y {self.m}")

        nodes = []
        for story in range(self.p + 1):
            node_id = story * (self.m + 1) + bay + 1
            nodes.append(node_id)
        return nodes

    def nodes(self) -> dict[int, tuple[float, float]]:
        """Devuelve las coordenadas de todos los nodos (solo coordenadas X y Z)"""
        nodes = {}  # {id: (x, y)}
        for i in range(self.p + 1):  # Pisos
            for k in range(self.m + 1):  # Columnas
                node_id = i * (self.m + 1) + k + 1
                nodes[node_id] = (k * self.l, i * self.h)
        return nodes

    def members(self) -> dict[str, dict[int, tuple[int, int]]]:
        """Devuelve todos los miembros (columnas y vigas) del pórtico"""
        members = {"columna": {}, "viga": {}}
        id_counter = 1

        # Columnas (miembros verticales)
        for i in range(self.p):  # Pisos
            for k in range(self.m + 1):  # Columnas
                node_i = i * (self.m + 1) + k + 1
                node_j = (i + 1) * (self.m + 1) + k + 1
                members["columna"][id_counter] = (node_i, node_j)
                id_counter += 1

        # Vigas (miembros horizontales)
        for i in range(1, self.p + 1):  # Pisos (empezando desde el primer piso)
            for k in range(self.m):  # Bahías
                node_i = i * (self.m + 1) + k + 1
                node_j = i * (self.m + 1) + k + 2
                members["viga"][id_counter] = (node_i, node_j)
                id_counter += 1

        return members

    def braces(self, bay: int) -> dict[int, tuple[int, int]]:
        """Devuelve las diagonales (braces) de una bahía específica"""
        if bay < 1 or bay > self.m:
            raise ValueError(
                f"La bahía {bay} no existe. Debe estar entre 1 y {self.m}")

        braces = {}
        brace_id = 1

        # Para cada piso donde se pueden colocar diagonales
        for story in range(self.p):
            # Nodos de la bahía especificada
            left_bottom = story * (self.m + 1) + bay
            right_bottom = story * (self.m + 1) + bay + 1
            left_top = (story + 1) * (self.m + 1) + bay
            right_top = (story + 1) * (self.m + 1) + bay + 1

            # Dos diagonales posibles (en forma de X)
            braces[brace_id] = (left_bottom, right_top)
            brace_id += 1
            braces[brace_id] = (right_bottom, left_top)
            brace_id += 1

        return braces


class EleType(Enum):
    TRUSS = 1
    BEAMR = 2
    FRAME = 3


def portal_parametric(n_bahias: int, n_pisos: int, l_bahia: float, h_piso: float, braces: list[int], ele_type: EleType | str, dual: bool = False) -> None:
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
    for node_id, (x, y) in nodes.items():
        model.add_node(node_id, x, y)
    for member_id, (node_i, node_j) in members["viga"].items():
        model.add_member(member_id, node_i, node_j, "vig")
    for member_id, (node_i, node_j) in members["columna"].items():
        model.add_member(member_id, node_i, node_j, "col")
    for i, bay in enumerate(braces):
        for j, (member_id, (node_i, node_j)) in enumerate(portal.braces(bay).items()):
            if isinstance(ele_type, str):
                ele_type = to_enum(ele_type, EleType)
            if dual:
                if ele_type == EleType.FRAME:
                    model.add_member(member_id + (i + 1) * len(model.members) + 1, node_i,
                                     node_j, "bra", beam_theory=BeamTheoriesType.EULER_BERNOULLI)
                elif ele_type == EleType.BEAMR:
                    ID_ELE = member_id + (i + 1) * len(model.members) + 1
                    model.add_member(ID_ELE, node_i, node_j, "bra",
                                     beam_theory=BeamTheoriesType.EULER_BERNOULLI)
                    model.add_releases(ID_ELE, mi=True, mj=True)
                else:
                    model.add_truss(
                        member_id + (i + 1) * len(model.members) + 1, node_i, node_j, "bra")
            elif j % 2 == 0:
                if ele_type == EleType.FRAME:
                    model.add_member(member_id + (i + 1) * len(model.members) + 1, node_i,
                                     node_j, "bra", beam_theory=BeamTheoriesType.EULER_BERNOULLI)
                elif ele_type == EleType.BEAMR:
                    ID_ELE = member_id + (i + 1) * len(model.members) + 1
                    model.add_member(ID_ELE, node_i, node_j, "bra",
                                     beam_theory=BeamTheoriesType.EULER_BERNOULLI)
                    model.add_releases(ID_ELE, mi=True, mj=True)
                else:
                    model.add_truss(
                        member_id + (i + 1) * len(model.members) + 1, node_i, node_j, "bra")
    for node_id in portal.nodes_from_story(0):
        model.add_restraint(node_id, True, True, True)
    model.add_load_pattern("Dead Load")
    model.add_self_weight("Dead Load")
    for i, node_id in enumerate(portal.nodes_for_bay(0)):
        if i == 0:   # saltarse la primera
            continue
        model.add_point_load(node_id, "Dead Load", fx=0.1*node_id)
    model.solve()
    model.plotter_options.mod_support_size = 2
    model_viewer(model)


if __name__ == "__main__":
    portal_parametric(3, 7, 7, 5, [1,3], EleType.TRUSS, True)
