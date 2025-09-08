from milcapy.core.node import Node
from milcapy.section.section import Section
import numpy as np

"""
==============================================
        ELEMENTO CUADRILATERO MEMBRANA
        CON MODOS INCOMPATIBLES (Q4I)
==============================================
|| Nodos                 || 4               ||
|| DOF por nodo          || 2               ||
|| Integracion           || Reducida 4P     ||
|| Shape Function        || Bilineal        ||
|| Formulacion           || Isoparametrica  ||
|| Clase de elemento     || Serendipitos    ||
|| Estado de programacion|| Listo           ||
==============================================
"""

class MembraneQuad6I:
    """Elemento de membrana cuadrilátero de 4 nodos + 2 modos incompatibles adicionales para evitar el shear locking."""
    def __init__(
        self,
        id: int,
        node1: Node,
        node2: Node,
        node3: Node,
        node4: Node,
        section: Section
    ) -> None:
        """
        Inicializa el elemento de membrana cuadrilátero de 4 nodos + 2 modos incompatibles adicionales para evitar el shear locking.

        Args:
            id (int): Identificador del elemento.
            node1 (Node): Primer nodo.
            node2 (Node): Segundo nodo.
            node3 (Node): Tercer nodo.
            node4 (Node): Cuarto nodo.
            section (Section): Sección del elemento tipo area (shell).
        """
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4
        self.E = section.E()
        self.v = section.v()
        self.t = section.t
        self.section = section
        self.dofs = np.concatenate((node1.dofs[:2], node2.dofs[:2], node3.dofs[:2], node4.dofs[:2]))

        r3 = np.sqrt(3)
        self.xi = [-1/r3, -1/r3, 1/r3, 1/r3]
        self.eta = [-1/r3, 1/r3, -1/r3, 1/r3]
        self.w = [1, 1, 1, 1]

    def shape_function(self, xi: float, eta: float):
        """
        Funcion de forma.
        """
        N1 = 1/4 * (1-xi) * (1-eta)
        N2 = 1/4 * (1+xi) * (1-eta)
        N3 = 1/4 * (1+xi) * (1+eta)
        N4 = 1/4 * (1-xi) * (1+eta)
        N5 = 1 + xi**2
        N6 = 1 + eta**2
        return N1, N2, N3, N4, N5, N6

    def coordinates(self, xi: float, eta: float):
        """
        Interpolacion de coordenadas del elemento.
        """
        N1,N2,N3,N4,N5,N6 = self.shape_function(xi, eta)
        x1, y1 = self.node1.vertex.coordinates
        x2, y2 = self.node2.vertex.coordinates
        x3, y3 = self.node3.vertex.coordinates
        x4, y4 = self.node4.vertex.coordinates
        x = N1*x1 + N2*x2 + N3*x3 + N4*x4
        y = N1*y1 + N2*y2 + N3*y3 + N4*y4
        return x, y

    def Ia(self) -> np.ndarray:
        """
        Matriz de unitaria modificada.
        """
        Ia = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        return Ia

    def J(self,xi: float, eta: float) -> np.ndarray:
        """
        Matriz de jacobiano.
        """
        x1, y1 = self.node1.vertex.coordinates
        x2, y2 = self.node2.vertex.coordinates
        x3, y3 = self.node3.vertex.coordinates
        x4, y4 = self.node4.vertex.coordinates

        J = np.array([
            [
                -x1*(1 - eta)/4 + x2*(1 - eta)/4 + x3*(eta + 1)/4 - x4*(eta + 1)/4,
                -y1*(1 - eta)/4 + y2*(1 - eta)/4 + y3*(eta + 1)/4 - y4*(eta + 1)/4
            ],
            [
                -x1*(1 - xi)/4 - x2*(xi + 1)/4 + x3*(xi + 1)/4 + x4*(1 - xi)/4,
                -y1*(1 - xi)/4 - y2*(xi + 1)/4 + y3*(xi + 1)/4 + y4*(1 - xi)/4
            ]
        ])


        return J

    def Gamma(self, xi: float, eta: float) -> np.ndarray:
        """
        Matriz de transformacion.
        """
        J = self.J(xi, eta)
        gamma = np.linalg.inv(J)
        zero = np.zeros((2,2))
        Gamma = np.vstack((
                            np.hstack((gamma, zero)),
                            np.hstack((zero, gamma))
                        ))
        return Gamma

    def dN(self, xi: float, eta: float) -> np.ndarray:
        """
        Matriz de derivadas de forma.
        """
        dN = np.array([
            [eta/4 - 1/4, 0, 1/4 - eta/4, 0, eta/4 + 1/4, 0, -eta/4 - 1/4, 0, 2*xi, 0, 0, 0],
            [xi/4 - 1/4, 0, -xi/4 - 1/4, 0, xi/4 + 1/4, 0, 1/4 - xi/4, 0, 0, 0, 2*eta, 0],
            [0, eta/4 - 1/4, 0, 1/4 - eta/4, 0, eta/4 + 1/4, 0, -eta/4 - 1/4, 0, 2*xi, 0, 0],
            [0, xi/4 - 1/4, 0, -xi/4 - 1/4, 0, xi/4 + 1/4, 0, 1/4 - xi/4, 0, 0, 0, 2*eta]
        ])

        return dN


    def B(self,xi: float, eta: float) -> np.ndarray:
        """
        Matriz de rigidez local.
        """
        dN = self.dN(xi, eta)
        Gamma = self.Gamma(xi, eta)
        Ia = self.Ia()
        B = Ia @ Gamma @ dN
        return B

    def D(self):
        """
        Matriz constitutiva de esfuerzo plano.
        """
        return (self.E/(1-self.v**2))*np.array([
            [1, self.v, 0],
            [self.v, 1, 0],
            [0, 0, (1-self.v)/2],
        ])

    def phiKi(self, xi: float, eta: float) -> np.ndarray:
        """
        Matriz de rigidez local.
        """
        detJ = np.linalg.det(self.J(xi, eta))
        B = self.B(xi, eta)
        D = self.D()
        t = self.t
        return B.T @ D @ B * detJ * t

    def Ki(self) -> np.ndarray:
        """
        Matriz de rigidez global.
        """
        Ki = np.zeros((12,12))
        for i in range(len(self.xi)):
            Ki += self.phiKi(self.xi[i], self.eta[i]) * self.w[i]

        # CONDENSACION ESTATICA:
        kcc = Ki[:8,:8]
        kcs = Ki[:8,8:]
        ksc = Ki[8:,:8]
        kss = Ki[8:,8:]
        try:
            Ki = kcc - kcs @ np.linalg.inv(kss) @ ksc
        except np.linalg.LinAlgError:
            Ki = kcc
        return Ki

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Matriz de transformacion 12x12 de local a global.
        """
        angle = np.arctan2(self.node2.vertex.y - self.node1.vertex.y, self.node2.vertex.x - self.node1.vertex.x)

        T = np.array([
            [np.cos(angle), -np.sin(angle),       0,              0,             0,              0,              0,              0,      ],
            [np.sin(angle),  np.cos(angle),       0,              0,             0,              0,              0,              0,      ],
            [0,                   0,        np.cos(angle), -np.sin(angle),       0,              0,              0,              0,      ],
            [0,                   0,        np.sin(angle),  np.cos(angle),       0,              0,              0,              0,      ],
            [0,                   0,              0,              0,        np.cos(angle), -np.sin(angle),       0,              0,      ],
            [0,                   0,              0,              0,        np.sin(angle),  np.cos(angle),       0,              0,      ],
            [0,                   0,              0,              0,             0,              0,         np.cos(angle), -np.sin(angle)],
            [0,                   0,              0,              0,             0,              0,         np.sin(angle),  np.cos(angle)],

        ])
        return T

    def global_stiffness_matrix(self) -> np.ndarray:
        """
        Matriz de rigidez global.
        """
        T = self.get_transformation_matrix()
        Ki = self.Ki()
        return T @ Ki @ T.T


if __name__ == "__main__":
    from milcapy.utils.geometry import Vertex
    import pandas as pd
    import numpy as np
    pd.set_option('display.float_format', '{:.3f}'.format)
    E = 2.53456e6
    v = 0.2
    t = 0.25
    # nd1 = Node(1, Vertex(0.0, 0.0))
    # nd2 = Node(2, Vertex(0.6, 0.0))
    # nd3 = Node(3, Vertex(0.6, 3.0))
    # nd4 = Node(4, Vertex(0.0, 3.0))
    nd1 = Node(1, Vertex(0.0, 0.0))
    nd2 = Node(2, Vertex(3.0, 0.0))
    nd3 = Node(3, Vertex(3.0, 0.6))
    nd4 = Node(4, Vertex(0.0, 0.6))
    el = MembraneQuad6I(1, nd1, nd2, nd3, nd4, E, v, t)
    p = 6
    # Q = np.array([-p/2, 0, -p/2, 0])
    Q = np.array([0, -p/2, 0, -p/2])
    # dofs = np.array([1, 2, 3, 4]) - 1
    dofs = np.array([3, 4, 5, 6]) - 1
    U = np.linalg.solve(el.global_stiffness_matrix()[np.ix_(dofs,dofs)], Q)
    # print(el.K())
    # print(pd.DataFrame(el.K()))
    print(pd.DataFrame(U*1000))