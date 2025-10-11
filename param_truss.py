import numpy as np

def curvar_coordenadas(arr, R):
    """
    Transforma un conjunto de coordenadas lineales en un arco circunferencial.
    arr: array_like, forma (n,2) con coordenadas (x,y) originales
    R: float, radio del arco
    """
    arr = np.array(arr, dtype=float)

    # longitudes acumuladas a lo largo de la línea
    d = np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))
    L = np.insert(np.cumsum(d), 0, 0.0)
    Ltot = L[-1]

    if Ltot == 0:
        raise ValueError("La longitud total de la línea no puede ser cero.")

    # ángulos a lo largo del arco
    theta_total = Ltot / R
    theta_i = (L / Ltot) * theta_total

    # coordenadas nuevas en el arco (con centro en (0,R))
    x_new = R * np.sin(theta_i)
    y_new = R * (1 - np.cos(theta_i))

    return np.column_stack((x_new, y_new))


class ParametricTruss3D:
    def __init__(self, nt, b, p, h, R=None):
        self.nt = nt
        self.b = b
        self.p = p
        self.h = h
        self.nn = (2 * (self.nt) + 1) * 3
        self.__ne = 9
        self.R = R  # si es None → truss recta

    def nodes(self) -> dict[int, tuple[float, float, float]]:
        nodes = {}
        base_coords = []  # coordenadas (x,0) para curvar

        for i in range(2 * self.nt + 1):
            x = i * self.b
            base_coords.append([x, 0])  # proyección en línea recta

        # aplicar curvatura si corresponde
        if self.R is not None:
            curvados = curvar_coordenadas(base_coords, self.R)
            x_coords, z_coords = curvados[:, 0], curvados[:, 1]
        else:
            x_coords = [c[0] for c in base_coords]
            z_coords = [0.0 for _ in base_coords]

        # construir nodos con IDs fijos
        for i, (x, z) in enumerate(zip(x_coords, z_coords)):
            nodes[3 * i + 1] = (x, 0, z)             # nodo inferior izquierdo
            nodes[3 * i + 2] = (x, self.p / 2, z + self.h)  # nodo superior
            nodes[3 * i + 3] = (x, self.p, z)        # nodo inferior derecho

        return nodes

    def triangulars(self) -> dict[int, tuple[int, int]]:
        triangulars = {}
        for i in range(2 * self.nt + 1):
            triangulars[self.__ne * i + 1] = (3 * i + 1, 3 * i + 2)
            triangulars[self.__ne * i + 2] = (3 * i + 2, 3 * i + 3)
            triangulars[self.__ne * i + 3] = (3 * i + 3, 3 * i + 1)
        return triangulars

    def longitudinal(self) -> dict[int, tuple[int, int]]:
        longitudinal = {}
        for i in range(2 * self.nt + 1):
            if i < 2 * self.nt:
                longitudinal[self.__ne * i + 4] = (3 * i + 1, 3 * i + 4)
                longitudinal[self.__ne * i + 5] = (3 * i + 2, 3 * i + 5)
                longitudinal[self.__ne * i + 6] = (3 * i + 3, 3 * i + 6)
        return longitudinal

    def bracings(self) -> dict[int, tuple[int, int]]:
        bracings = {}
        for i in range(2 * self.nt + 1):
            if i < 2 * self.nt:
                if i < self.nt:
                    bracings[self.__ne * i + 7] = (3 * i + 1, 3 * i + 5)
                    bracings[self.__ne * i + 8] = (3 * i + 3, 3 * i + 4)
                    bracings[self.__ne * i + 9] = (3 * i + 3, 3 * i + 5)
                else:
                    bracings[self.__ne * i + 7] = (3 * i + 2, 3 * i + 4)
                    bracings[self.__ne * i + 8] = (3 * i + 1, 3 * i + 6)
                    bracings[self.__ne * i + 9] = (3 * i + 2, 3 * i + 6)
        return bracings

    def elements(self) -> dict[int, tuple[int, int]]:
        elements = {}
        elements.update(self.triangulars())
        elements.update(self.longitudinal())
        elements.update(self.bracings())
        return elements




if __name__ == "__main__":
    from milcatrusspy import Model

    nt = 4
    b, p, h = 1, 1, 1
    matsec = [2.1e6, 0.3*0.4]

    pt = ParametricTruss3D(nt, b, p, h, R=30)
    model = Model(ndm=3)
    for idn, coord in pt.nodes().items():
        model.add_node(idn, *coord)
    for idel, nodes in pt.elements().items():
        model.add_element(idel, *nodes, *matsec)
    for idn in [1, 2, 3, pt.nn-2, pt.nn-1, pt.nn]:
        model.set_restraints(idn, True, True, True)
    model.set_load(3*pt.nt+2, fz=-10)
    model.solve()
    # model.print_results()
    model.plot_model(labels=False)
    model.plot_deformed(scale=500, labels=False)
    model.plot_reactions()
    model.plot_axial_forces(scale=0.02, labels=True)
