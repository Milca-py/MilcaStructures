"""
==============================================
ELEMENTO CUADRILATERO MEMBRANA (Q8)
==============================================
|| Nodos                 || 8               ||
|| DOF por nodo          || 2               ||
|| Integracion           || Reducida 4P     ||
|| Shape Function        || Bi Cuadrada     ||
|| Formulacion           || Isoparametrica  ||
|| clase de elemento     || Serendipitos    ||
|| Estado de programacion|| Listo           ||
==============================================
"""


from milcapy.core.node import Node, NodeArbitrary
from milcapy.section.section import ShellSection
import numpy as np
import uuid
from milcapy.utils.types import ConstitutiveModelType, IntegrationType
from milcapy.utils.geometry import Vertex

class MembraneQuad8:
    def __init__(
        self,
        id: int,
        node1: Node,
        node2: Node,
        node3: Node,
        node4: Node,
        section: ShellSection,
        state: ConstitutiveModelType,
        integration: IntegrationType
    ) -> None:
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4
        self.node5 = NodeArbitrary(uuid.uuid4().int, Vertex(*((node1.vertex.coordinates + node2.vertex.coordinates)/2)))
        self.node6 = NodeArbitrary(uuid.uuid4().int, Vertex(*((node2.vertex.coordinates + node3.vertex.coordinates)/2)))
        self.node7 = NodeArbitrary(uuid.uuid4().int, Vertex(*((node3.vertex.coordinates + node4.vertex.coordinates)/2)))
        self.node8 = NodeArbitrary(uuid.uuid4().int, Vertex(*((node4.vertex.coordinates + node1.vertex.coordinates)/2)))
        self.state = state
        self.integration = integration
        self.section = section
        self.E = section.E()
        self.v = section.v()
        self.t = section.t
        self.dofs = np.concatenate(
            (node1.dofs[:2], node2.dofs[:2], node3.dofs[:2], node4.dofs[:2]))

        if self.integration == IntegrationType.REDUCED:
            r3 = np.sqrt(3)
            self.xi = [-1/r3, -1/r3, 1/r3, 1/r3]
            self.eta = [-1/r3, 1/r3, -1/r3, 1/r3]
            self.w = [1, 1, 1, 1]
        elif self.integration == IntegrationType.COMPLETE:
            ra = np.sqrt(0.6)
            self.xi = [-ra, ra, ra, -ra, 0, ra, 0, -ra, 0]
            self.eta = [-ra, -ra, ra, ra, -ra, 0, ra, 0, 0]
            self.w = [25/81, 25/81, 25/81, 25/81, 40/81, 40/81, 40/81, 40/81, 64/81]

    def shape_function(self, xi: float, eta: float) -> tuple[float, float, float, float, float, float, float, float]:
        """
        Calcula las funciones de forma en un punto (xi, eta).
        """
        N1 = -1/4 * (1-xi) * (1-eta) * (1+xi+eta)
        N2 = -1/4 * (1+xi) * (1-eta) * (1-xi+eta)
        N3 = -1/4 * (1+xi) * (1+eta) * (1-xi-eta)
        N4 = -1/4 * (1-xi) * (1+eta) * (1+xi-eta)
        N5 = 1/2 * (1-xi) * (1-eta) * (1+xi)
        N6 = 1/2 * (1+xi) * (1-eta) * (1+eta)
        N7 = 1/2 * (1-xi) * (1+eta) * (1+xi)
        N8 = 1/2 * (1-xi) * (1-eta) * (1+eta)
        return N1, N2, N3, N4, N5, N6, N7, N8

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna las coordenadas de los nodos del elemento.
        """
        x = np.array([self.node1.vertex.x, self.node2.vertex.x, self.node3.vertex.x, self.node4.vertex.x])
        y = np.array([self.node1.vertex.y, self.node2.vertex.y, self.node3.vertex.y, self.node4.vertex.y])
        return x, y


    def coordinates(self, xi: float, eta: float) -> tuple[float, float]:
        """
        Calcula las coordenadas en un punto (xi, eta).
        """
        N1, N2, N3, N4, N5, N6, N7, N8 = self.shape_function(xi, eta)
        x1, y1 = self.node1.vertex.coordinates
        x2, y2 = self.node2.vertex.coordinates
        x3, y3 = self.node3.vertex.coordinates
        x4, y4 = self.node4.vertex.coordinates
        x5, y5 = self.node5.vertex.coordinates
        x6, y6 = self.node6.vertex.coordinates
        x7, y7 = self.node7.vertex.coordinates
        x8, y8 = self.node8.vertex.coordinates
        x = N1*x1 + N2*x2 + N3*x3 + N4*x4 + N5*x5 + N6*x6 + N7*x7 + N8*x8
        y = N1*y1 + N2*y2 + N3*y3 + N4*y4 + N5*y5 + N6*y6 + N7*y7 + N8*y8
        return (x, y)

    def Ia(self) -> np.ndarray:
        """
        Calcula la matriz unitaria ampliada.
        """
        Ia = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        return Ia

    def J(self, xi: float, eta: float) -> np.ndarray:
        """
        Calcula la matriz Jacobiana.
        """
        x1, y1 = self.node1.vertex.coordinates
        x2, y2 = self.node2.vertex.coordinates
        x3, y3 = self.node3.vertex.coordinates
        x4, y4 = self.node4.vertex.coordinates
        x5, y5 = self.node5.vertex.coordinates
        x6, y6 = self.node6.vertex.coordinates
        x7, y7 = self.node7.vertex.coordinates
        x8, y8 = self.node8.vertex.coordinates

        dx_dxi = -0.25*eta**2*x1 + 0.25*eta**2*x2 + 0.25*eta**2*x3 - 0.25*eta**2*x4 - 0.5*eta**2*x6 + 0.5*eta**2*x8 - 0.5*eta*x1*xi + 0.25*eta*x1 - 0.5*eta*x2*xi - 0.25*eta*x2 + 0.5 * \
            eta*x3*xi + 0.25*eta*x3 + 0.5*eta*x4*xi - 0.25*eta*x4 + 1.0*eta*x5*xi - 1.0*eta*x7*xi + \
            0.5*x1*xi + 0.5*x2*xi + 0.5*x3*xi + 0.5*x4 * \
            xi - 1.0*x5*xi + 0.5*x6 - 1.0*x7*xi - 0.5*x8

        dy_dxi = -0.25*eta**2*y1 + 0.25*eta**2*y2 + 0.25*eta**2*y3 - 0.25*eta**2*y4 - 0.5*eta**2*y6 + 0.5*eta**2*y8 - 0.5*eta*xi*y1 - 0.5*eta*xi*y2 + 0.5*eta*xi*y3 + 0.5*eta*xi*y4 + \
            1.0*eta*xi*y5 - 1.0*eta*xi*y7 + 0.25*eta*y1 - 0.25*eta*y2 + 0.25*eta*y3 - 0.25*eta*y4 + \
            0.5*xi*y1 + 0.5*xi*y2 + 0.5*xi*y3 + 0.5*xi * \
            y4 - 1.0*xi*y5 - 1.0*xi*y7 + 0.5*y6 - 0.5*y8

        dx_deta = -0.5*eta*x1*xi + 0.5*eta*x1 + 0.5*eta*x2*xi + 0.5*eta*x2 + 0.5*eta*x3*xi + 0.5*eta*x3 - 0.5*eta*x4*xi + 0.5*eta*x4 - 1.0*eta*x6*xi - 1.0*eta*x6 + 1.0*eta*x8*xi - \
            1.0*eta*x8 - 0.25*x1*xi**2 + 0.25*x1*xi - 0.25*x2*xi**2 - 0.25*x2*xi + 0.25*x3*xi**2 + \
            0.25*x3*xi + 0.25*x4*xi**2 - 0.25*x4*xi + 0.5 * \
            x5*xi**2 - 0.5*x5 - 0.5*x7*xi**2 + 0.5*x7

        dy_deta = -0.5*eta*xi*y1 + 0.5*eta*xi*y2 + 0.5*eta*xi*y3 - 0.5*eta*xi*y4 - 1.0*eta*xi*y6 + 1.0*eta*xi*y8 + 0.5*eta*y1 + 0.5*eta*y2 + 0.5*eta*y3 + 0.5*eta*y4 - 1.0*eta*y6 - \
            1.0*eta*y8 - 0.25*xi**2*y1 - 0.25*xi**2*y2 + 0.25*xi**2*y3 + 0.25*xi**2*y4 + 0.5*xi**2 * \
            y5 - 0.5*xi**2*y7 + 0.25*xi*y1 - 0.25*xi*y2 + \
            0.25*xi*y3 - 0.25*xi*y4 - 0.5*y5 + 0.5*y7

        return np.array([[dx_dxi,  dy_dxi],
                        [dx_deta,  dy_deta]])

    def Gamma(self, xi: float, eta: float) -> np.ndarray:
        """
        Calcula la matriz Jacobiana invertida.
        """
        J = self.J(xi, eta)
        gamma = np.linalg.inv(J)
        zero = np.zeros((2, 2))
        Gamma = np.concatenate((
            np.concatenate((gamma, zero), axis=1),
            np.concatenate((zero, gamma), axis=1)
        ),
            axis=0)
        return Gamma

    def dN(self, xi: float, eta: float) -> np.ndarray:
        """
        Calcula la matriz de derivadas de las funciones de forma.
        """
        dN1_dxi = 0.25*(-eta - 2*xi)*(eta - 1)
        dN2_dxi = 0.25*(eta - 1)*(eta - 2*xi)
        dN3_dxi = 0.25*(eta + 1)*(eta + 2*xi)
        dN4_dxi = 0.25*(-eta + 2*xi)*(eta + 1)
        dN5_dxi = 1.0*xi*(eta - 1)
        dN6_dxi = 0.5 - 0.5*eta**2
        dN7_dxi = 1.0*xi*(-eta - 1)
        dN8_dxi = 0.5*eta**2 - 0.5
        dN1_deta = 0.25*(-2*eta - xi)*(xi - 1)
        dN2_deta = 0.25*(2*eta - xi)*(xi + 1)
        dN3_deta = 0.25*(2*eta + xi)*(xi + 1)
        dN4_deta = 0.25*(-2*eta + xi)*(xi - 1)
        dN5_deta = 0.5*xi**2 - 0.5
        dN6_deta = 1.0*eta*(-xi - 1)
        dN7_deta = 0.5 - 0.5*xi**2
        dN8_deta = 1.0*eta*(xi - 1)

        return np.array([
            [dN1_dxi,   0,        dN2_dxi,     0,           dN3_dxi,     0,           dN4_dxi,     0,
                dN5_dxi,     0,           dN6_dxi,     0,           dN7_dxi,     0,           dN8_dxi,     0],
            [dN1_deta,  0,        dN2_deta,    0,           dN3_deta,    0,           dN4_deta,    0,
                dN5_deta,    0,           dN6_deta,    0,           dN7_deta,    0,           dN8_deta,    0],
            [0,         dN1_dxi,  0,           dN2_dxi,     0,           dN3_dxi,     0,           dN4_dxi,
                0,           dN5_dxi,     0,           dN6_dxi,     0,           dN7_dxi,     0,           dN8_dxi],
            [0,         dN1_deta, 0,           dN2_deta,    0,           dN3_deta,    0,           dN4_deta,
                0,           dN5_deta,    0,           dN6_deta,    0,           dN7_deta,    0,           dN8_deta],
        ])

    def B(self, xi: float, eta: float) -> np.ndarray:
        """
        Calcula la matriz de deformacion.
        """
        dN = self.dN(xi, eta)
        Gamma = self.Gamma(xi, eta)
        Ia = self.Ia()
        B = Ia @ Gamma @ dN
        return B

    def D(self) -> np.ndarray:
        """
        Matriz constitutiva de esfuerzo plano.
        """
        v = self.section.v()
        E = self.section.E()
        # ! T E N S I O N    C O N S T A N T E
        if self.state == ConstitutiveModelType.PLANE_STRESS:
            k = E/(1-v**2)
            return k*np.array([
                            [1, v, 0],
                            [v, 1, 0],
                            [0, 0, (1-v)/2],
                            ])

        # ! D E F O R M A C I O N    C O N S T A N T E
        if self.state == ConstitutiveModelType.PLANE_STRAIN:
            k = (E*(1-v))/((1 + v)*(1 - 2*v))
            return k * np.array([
                [1,                 v/(1-v), 0],
                [v/(1-v),           1,       0],
                [0,                 0,       (1-2*v)/(2*(1-v))],
            ])

    def phiKi(self, xi: float, eta: float) -> np.ndarray:
        """
        Matriz funcional de rigidez.
        """
        detJ = np.linalg.det(self.J(xi, eta))
        B = self.B(xi, eta)
        D = self.D()
        t = self.t
        return B.T @ D @ B * detJ * t

    @staticmethod
    def P() -> np.ndarray:
        """
        Matriz de Permutacion.
        """
        P = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0]
            ])
        return P

    def global_stiffness_matrix(self) -> np.ndarray:
        """
        Calcula la matriz rigidez global atravez de la intergracion numerica cuadratura de gauss.
        y condensa la matriz rigidez en los nodos de vertices.
        """
        Ki = np.zeros((16, 16))
        for i in range(len(self.xi)):
            Ki += self.phiKi(self.xi[i], self.eta[i]) * self.w[i]
        # Condensacion estatica:
        # kc =kcc-kci*kii^-1*kic
        c = [0, 1, 2, 3, 4, 5, 6, 7]
        i = [8, 9, 10, 11, 12, 13, 14, 15]
        Kcc = Ki[np.ix_(c, c)]
        Kci = Ki[np.ix_(c, i)]
        Kic = Ki[np.ix_(i, c)]
        Kii = Ki[np.ix_(i, i)]
        kc = Kcc - Kci @ np.linalg.inv(Kii) @ Kic
        # kc = self.P() @ kc @ self.P().T
        return kc
        # return Ki


if __name__ == "__main__":
    import pandas as pd
    from milcapy.utils.geometry import Vertex
    pd.set_option('display.float_format', '{:.7f}'.format)
    from milcapy.material.material import Material
    from milcapy.section.section import ShellSection
    # Units: tonf, m
    l = 1.5  # m
    h = 0.6  # m
    E = 2534.56*10**3  # tonf/m2
    v = 0.2
    t = 0.25  # m
    material = Material("material", E, v, 0)
    section = ShellSection("section", material, t)
    nd1 = Node(1, Vertex(0, 0))
    nd2 = Node(2, Vertex(h, 0))
    nd3 = Node(3, Vertex(h, l))
    nd4 = Node(4, Vertex(0, l))
    ele = MembraneQuad8(1, nd1, nd2, nd3, nd4, section, ConstitutiveModel.PLANE_STRESS, IntegrationType.REDUCED)
    # K = ele.Ki()/10**5#[4:16, 4:16] #*0.018
    # F = np.array([3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # u = np.linalg.solve(K, F)
    K = ele.global_stiffness_matrix()[4:8, 4:8]
    F = np.array([3, 0, 3, 0])
    u = np.linalg.solve(K, F)
    # print(pd.DataFrame(K))
    print(pd.DataFrame(u)*1000)
    # print(ele.D())
