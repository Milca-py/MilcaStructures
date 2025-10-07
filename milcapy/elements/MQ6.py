import numpy as np
from milcapy.core.node import Node
from milcapy.section.section import ShellSection
from milcapy.utils.types import ConstitutiveModelType

class MembraneQuad6:
    """Elemento de membrana cuadrilátero de 4 nodos + grados de libertad adicionales de perforacion (para compatibilidad de rigidez e evitar la inestabilidad numerica en el ensamblaje)
    Nota: Este elemento converge a la solucion excata de la viga de Euler-Bernoulli (sin considerar deformaciones por cortante)
    
    Args:
        id (int): Identificador del elemento.
        node1 (Node): Primer nodo.
        node2 (Node): Segundo nodo.
        node3 (Node): Tercer nodo.
        node4 (Node): Cuarto nodo.
        section (ShellSection): Sección del elemento tipo area (shell).
        state (ConstitutiveModelType): Modelo constitutivo del elemento.
    
    Notas:
        Se recomienda usar este elemento por encima de los elementos MembraneQuad6I y MembraneQuad6IModificado.
    """
    def __init__(
        self,
        id: int,
        node1: Node,
        node2: Node,
        node3: Node,
        node4: Node,
        section: ShellSection,
        state: ConstitutiveModelType,
    ) -> None:
        """
        Inicializa el elemento de membrana cuadrilátero de 4 nodos y 3 dof por nodo.
        """
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4
        self.section = section
        self.state = state
        self.E = self.section.E()
        self.v = self.section.v()
        self.t = self.section.t
        self.dofs = np.concatenate((node1.dofs, node2.dofs, node3.dofs, node4.dofs))
        r3 = np.sqrt(3)
        self.xi = [-1/r3, -1/r3, 1/r3, 1/r3]
        self.eta = [-1/r3, 1/r3, -1/r3, 1/r3]
        self.w = [1, 1, 1, 1]

    def set_gauss_points(self, xi: list[float], eta: list[float], w: list[float]):
        """
        Setea los puntos de gauss.
        """
        self.xi = xi
        self.eta = eta
        self.w = w

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna las coordenadas de los nodos del elemento.
        """
        x = np.array([self.node1.vertex.x, self.node2.vertex.x, self.node3.vertex.x, self.node4.vertex.x])
        y = np.array([self.node1.vertex.y, self.node2.vertex.y, self.node3.vertex.y, self.node4.vertex.y])
        return x, y

    def shape_function(self, xi: float, eta: float):
        """
        Funciones de forma.
        """
        N1 = (1-xi)*(1-eta)/4
        N2 = (1+xi)*(1-eta)/4
        N3 = (1+xi)*(1+eta)/4
        N4 = (1-xi)*(1+eta)/4
        N5 = (1-xi**2)*(1-eta)/2
        N6 = (1-eta**2)*(1+xi)/2
        N7 = (1-xi**2)*(1+eta)/2
        N8 = (1-eta**2)*(1-xi)/2
        return N1, N2, N3, N4, N5, N6, N7, N8

    def coordinates(self, xi: float, eta: float):
        """
        Interpolacion de coordenadas del elemento.
        """
        N1, N2, N3, N4, *_ = self.shape_function(xi, eta)
        x1, y1 = self.node1.coords
        x2, y2 = self.node2.coords
        x3, y3 = self.node3.coords
        x4, y4 = self.node4.coords
        x = N1*x1 + N2*x2 + N3*x3 + N4*x4
        y = N1*y1 + N2*y2 + N3*y3 + N4*y4
        return x, y

    def Ia(self) -> np.ndarray:
        """
        Matriz de unitaria ampliada.
        """
        Ia = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        return Ia

    def J(self, xi: float, eta: float) -> np.ndarray:
        """
        Matriz de jacobiano.
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
        Matriz de jacobiano invertida.
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

    def dNdisp(self, xi: float, eta: float) -> np.ndarray:
        """
        Derivada de las funciones de forma (dof: desplazamientos).
        """
        dN = (1/4)*np.array([
            [eta-1, 0, 1-eta, 0, 1+eta, 0, -1-eta, 0],
            [xi-1,  0, -1-xi, 0, 1+xi,   0,  1-xi, 0],
            [0,     eta-1, 0, 1-eta, 0, 1+eta, 0, -1-eta],
            [0,     xi-1,  0, -1-xi, 0, 1+xi,   0,  1-xi]
        ])
        return dN

    def Bdisp(self, xi: float, eta: float) -> np.ndarray:
        """
        Matriz de deformacion de desplazamientos.
        """
        dN = self.dNdisp(xi, eta)
        Gamma = self.Gamma(xi, eta)
        Ia = self.Ia()
        B = Ia @ Gamma @ dN
        return B

    def dNrot(self, xi: float, eta: float) -> np.ndarray:
        """
        Derivada de las funciones de forma modificadas (dof: rotaciones).
        """
        x1, y1 = self.node1.vertex.coordinates
        x2, y2 = self.node2.vertex.coordinates
        x3, y3 = self.node3.vertex.coordinates
        x4, y4 = self.node4.vertex.coordinates

        dN = (1/16)*np.array([
            [
                2 * xi * (eta - 1) * (y1 - y2)
                + (eta**2 - 1) * (y1 - y4),

                -2 * xi * (eta - 1) * (y1 - y2)
                - (eta**2 - 1) * (y2 - y3),

                -2 * xi * (eta + 1) * (y3 - y4)
                + (eta**2 - 1) * (y2 - y3),

                2 * xi * (eta + 1) * (y3 - y4)
                - (eta**2 - 1) * (y1 - y4)
            ],
            [
                2 * eta * (xi - 1) * (y1 - y4)
                + (xi**2 - 1) * (y1 - y2),

                -2 * eta * (xi + 1) * (y2 - y3)
                - (xi**2 - 1) * (y1 - y2),

                2 * eta * (xi + 1) * (y2 - y3)
                - (xi**2 - 1) * (y3 - y4),

                -2 * eta * (xi - 1) * (y1 - y4)
                + (xi**2 - 1) * (y3 - y4)
            ],
            [
                -2 * xi * (eta - 1) * (x1 - x2)
                - (eta**2 - 1) * (x1 - x4),

                2 * xi * (eta - 1) * (x1 - x2)
                + (eta**2 - 1) * (x2 - x3),

                2 * xi * (eta + 1) * (x3 - x4)
                - (eta**2 - 1) * (x2 - x3),

                -2 * xi * (eta + 1) * (x3 - x4)
                + (eta**2 - 1) * (x1 - x4)
            ],
            [
                -2 * eta * (x1 - x4) * (xi - 1)
                - (x1 - x2) * (xi**2 - 1),

                2 * eta * (x2 - x3) * (xi + 1)
                + (x1 - x2) * (xi**2 - 1),

                -2 * eta * (x2 - x3) * (xi + 1)
                + (x3 - x4) * (xi**2 - 1),

                2 * eta * (x1 - x4) * (xi - 1)
                - (x3 - x4) * (xi**2 - 1)
            ]
        ])
        return dN

    def Brot(self, xi: float, eta: float) -> np.ndarray:
        """
        Matriz de deformacion de rotaciones.
        """
        dN = self.dNrot(xi, eta)
        Gamma = self.Gamma(xi, eta)
        Ia = self.Ia()
        B = Ia @ Gamma @ dN
        return B

    def B(self, xi: float, eta: float) -> np.ndarray:
        """
        Matriz de deformacion.
        """
        B = np.hstack((self.Bdisp(xi, eta), self.Brot(xi, eta)))

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

    def phiKi(self, xi: float, eta: float):
        """
        Matriz funcional de rigidez.
        """
        detJ = np.linalg.det(self.J(xi, eta))
        B = self.B(xi, eta)
        D = self.D()
        t = self.t
        return B.T @ D @ B * detJ * t

    def Ki(self):
        """
        Calcula la matriz rigidez atravez de la intergracion numerica cuadratura de gauss.
        """
        Ki = np.zeros((12, 12))
        for i in range(len(self.xi)):
            Ki += self.phiKi(self.xi[i], self.eta[i]) * self.w[i]


        # orden actual: [u1x, u1y, u2x, u2y, u3x, u3y, u4x, u4y, r1z, r2z, r3z, r4z]
        # orden deseado: [u1x, u1y, r1z, u2x, u2y, r2z, u3x, u3y, r3z, u4x, u4y, r4z]

        perm = [0, 1, 8, 2, 3, 9, 4, 5, 10, 6, 7, 11]

        # reordenar filas y columnas
        Ki = Ki[np.ix_(perm, perm)]


        return Ki #[U1X, U1Y, r1Z,    U2X, U2Y, r2Z,    U3X, U3Y, r3Z,    U4X, U4Y, r4Z]

    def get_transformation_matrix(self):
        """
        Matriz de transformacion 12x12 de local a global.
        """
        angle = np.arctan2(self.node2.vertex.y - self.node1.vertex.y, self.node2.vertex.x - self.node1.vertex.x)

        T = np.array([
            [np.cos(angle), -np.sin(angle), 0,      0,              0,        0,      0,              0,        0,      0,              0,        0],
            [np.sin(angle),  np.cos(angle), 0,      0,              0,        0,      0,              0,        0,      0,              0,        0],
            [0,                   0,        1,      0,              0,        0,      0,              0,        0,      0,              0,        0],
            [0,                   0,        0, np.cos(angle), -np.sin(angle), 0,      0,              0,        0,      0,              0,        0],
            [0,                   0,        0, np.sin(angle),  np.cos(angle), 0,      0,              0,        0,      0,              0,        0],
            [0,                   0,        0,      0,              0,        1,      0,              0,        0,      0,              0,        0],
            [0,                   0,        0,      0,              0,        0, np.cos(angle), -np.sin(angle), 0,      0,              0,        0],
            [0,                   0,        0,      0,              0,        0, np.sin(angle),  np.cos(angle), 0,      0,              0,        0],
            [0,                   0,        0,      0,              0,        0,      0,              0,        1,      0,              0,        0],
            [0,                   0,        0,      0,              0,        0,      0,              0,        0, np.cos(angle), -np.sin(angle), 0],
            [0,                   0,        0,      0,              0,        0,      0,              0,        0, np.sin(angle),  np.cos(angle), 0],
            [0,                   0,        0,      0,              0,        0,      0,              0,        0,      0,              0,        1],

        ])
        return T


    def global_stiffness_matrix(self):
        """
        Calcula la matriz rigidez global.
        """
        # T = self.get_transformation_matrix()
        Ki = self.Ki()
        # K_GLOBAL = T @ Ki @ T.T
        K_GLOBAL = Ki
        return K_GLOBAL




if __name__ == "__main__":
    import pandas as pd
    pd.set_option('display.float_format', '{:.5f}'.format)
    E = 25  # MPa
    v = 0.20
    t = 400  # mm
    l = 2000  # mm
    h = 600  # mm
    state = ConstitutiveModel.PLANE_STRESS
    from milcapy.utils.geometry import Vertex
    from milcapy.section.section import ShellSection
    from milcapy.material.material import Material

    material = Material("mat",E, v, 0)
    section = ShellSection("sec",material, t)
    nd1 = Node(1, Vertex(0, 0))
    nd2 = Node(2, Vertex(l, 0))
    nd3 = Node(3, Vertex(l, h))
    nd4 = Node(4, Vertex(0, h))
    el = MembraneQuad6(1, nd1, nd2, nd3, nd4, section, state)
    k = el.global_stiffness_matrix()
    f = 20
    # q = np.array([0, -0.5*f, 0,-0.5*f, 0, 0, 0, 0])
    q = np.array([0,     0, -0.5*f, 0,    0, -0.5*f, 0,    0])
    idx =        [3,     4, 5, 6,         7, 8, 9,         12]
    idx = np.array(idx) - 1
    k =k[np.ix_(idx, idx)]

    u = np.linalg.solve(k, q)
    # print(pd.DataFrame(k))
    print(pd.DataFrame(u))