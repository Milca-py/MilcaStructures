"""
==============================================
ELEMENTO CUADRILATERO MEMBRANA (Q4)
==============================================
|| Nodos                 || 4               ||
|| DOF por nodo          || 2               ||
|| Integracion           || Reducida 4P     ||
|| Shape Function        || bilineal        ||
|| Formulacion           || Isoparametrica  ||
|| clase de elemento     || Serendipitos    ||
|| Estado de programacion|| Listo           ||
==============================================
"""

from milcapy.core.node import Node
from milcapy.section.section import ShellSection
from milcapy.utils.types import ConstitutiveModel
import numpy as np

class MembraneQuad4:
    def __init__(
        self,
        id: int,
        node1: Node,
        node2: Node,
        node3: Node,
        node4: Node,
        section: ShellSection,
        state: ConstitutiveModel,
    ) -> None:
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4
        self.state = state
        self.section = section
        self.E = section.E()
        self.v = section.v()
        self.t = section.t
        self.dofs = np.concatenate((node1.dofs[:2], node2.dofs[:2], node3.dofs[:2], node4.dofs[:2]))

        r3 = np.sqrt(3)
        self.xi = [-1/r3, -1/r3, 1/r3, 1/r3]
        self.eta = [-1/r3, 1/r3, -1/r3, 1/r3]
        self.w = [1, 1, 1, 1]

    def shape_function(self, xi: float, eta: float) -> tuple[float, float, float, float]:
        """
        Calcula las funciones de forma en un punto (xi, eta).
        """
        N1 = 1/4 * (1-xi) * (1-eta)
        N2 = 1/4 * (1+xi) * (1-eta)
        N3 = 1/4 * (1+xi) * (1+eta)
        N4 = 1/4 * (1-xi) * (1+eta)
        return N1, N2, N3, N4

    def coordinates(self, xi: float, eta: float) -> tuple[float, float]:
        """
        Calcula las coordenadas en un punto (xi, eta).
        """
        N1,N2,N3,N4 = self.shape_function(xi, eta)
        x1, y1 = self.node1.vertex.coordinates
        x2, y2 = self.node2.vertex.coordinates
        x3, y3 = self.node3.vertex.coordinates
        x4, y4 = self.node4.vertex.coordinates
        x = N1*x1 + N2*x2 + N3*x3 + N4*x4
        y = N1*y1 + N2*y2 + N3*y3 + N4*y4
        return x, y

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

    def J(self,xi: float, eta: float) -> np.ndarray:
        """
        Calcula la matriz Jacobiana.
        """
        x1, y1 = self.node1.vertex.coordinates
        x2, y2 = self.node2.vertex.coordinates
        x3, y3 = self.node3.vertex.coordinates
        x4, y4 = self.node4.vertex.coordinates
        J11 = -x1*(1-eta) + x2*(1-eta) + x3*(1+eta) - x4*(1+eta)
        J12 = -y1*(1-eta) + y2*(1-eta) + y3*(1+eta) - y4*(1+eta)
        J21 = -x1*(1-xi) - x2*(1+xi) + x3*(1+xi) + x4*(1-xi)
        J22 = -y1*(1-xi) - y2*(1+xi) + y3*(1+xi) + y4*(1-xi)
        return np.array([[J11, J12], [J21, J22]]) * (1/4)

    def Gamma(self, xi: float, eta: float) -> np.ndarray:
        """
        Calcula la matriz Jacobiana invertida.
        """
        J = self.J(xi, eta)
        gamma = np.linalg.inv(J)
        zero = np.zeros((2,2))
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
        dN = (1/4)*np.array([
            [eta-1, 0, 1-eta, 0, 1+eta, 0, -1-eta, 0],
            [xi-1,  0, -1-xi, 0, 1+xi,   0,  1-xi, 0],
            [0,     eta-1, 0, 1-eta, 0, 1+eta, 0, -1-eta],
            [0,     xi-1,  0, -1-xi, 0, 1+xi,   0,  1-xi]
        ])
        return dN


    def B(self,xi: float, eta: float) -> np.ndarray:
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
        if self.state == ConstitutiveModel.PLANE_STRESS:
            k = E/(1-v**2)
            return k*np.array([
                            [1, v, 0],
                            [v, 1, 0],
                            [0, 0, (1-v)/2],
                            ])

        # ! D E F O R M A C I O N    C O N S T A N T E
        if self.state == ConstitutiveModel.PLANE_STRAIN:
            k = (E*(1-v))/((1 + v)*(1 - 2*v))
            return k * np.array([
                [1,                 v/(1-v), 0],
                [v/(1-v),           1,       0],
                [0,                 0,       (1-2*v)/(2*(1-v))],
            ])

    def phiKi(self, xi: float, eta: float):
        """
        Calcula la matriz funcional de rigidez.
        """
        detJ = np.linalg.det(self.J(xi, eta))
        B = self.B(xi, eta)
        D = self.D()
        t = self.t
        return B.T @ D @ B * detJ * t

    def global_stiffness_matrix(self):
        """
        Calcula la matriz rigidez global.
        """
        Ki = np.zeros((8,8))
        for i in range(len(self.xi)):
            Ki += self.phiKi(self.xi[i], self.eta[i]) * self.w[i]
        return Ki



if __name__ == "__main__":
    from milcapy.utils.geometry import Vertex
    E = 30*10**6 # psi
    v = 0.25
    t = 1 # in
    nd1 = Node(1, Vertex(3, 2))
    nd2 = Node(2, Vertex(5, 2))
    nd3 = Node(3, Vertex(5, 4))
    nd4 = Node(4, Vertex(3, 4))
    el = MembraneQuad4(1, nd1, nd2, nd3, nd4, E, v, t, ConstitutiveModel.PLANE_STRESS)
    k = 10**4
    print(el.global_stiffness_matrix()/k)








