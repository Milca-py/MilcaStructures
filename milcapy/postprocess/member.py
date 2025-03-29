from milcapy.elements.member import Member
from typing import Dict
import numpy as np


class BeamSeg():
    """
    Un segmento de viga matemáticamente continuo
    """

    def __init__(self):
        self.xi: float | None = None  # Ubicación inicial del segmento de la viga (relativa al inicio de la viga)
        self.xj: float | None = None  # Ubicación final del segmento de la viga (relativa al inicio de la viga)
        self.qi: float | None = None  # Carga distribuida transversal lineal en el inicio del segmento
        self.qj: float | None = None  # Carga distribuida transversal lineal en el final del segmento
        self.pi: float | None = None  # Carga distribuida axial lineal en el inicio del segmento
        self.pj: float | None = None  # Carga distribuida axial lineal en el final del segmento
        self.Pi: float | None = None  # Fuerza axial interna en el inicio del segmento
        self.Pj: float | None = None  # Fuerza axial interna en el final del segmento
        self.Vi: float | None = None  # Fuerza cortante interna en el inicio del segmento
        self.Vj: float | None = None  # Fuerza cortante interna en el final del segmento
        self.Mi: float | None = None  # Momento interno en el inicio del segmento
        self.Mj: float | None = None  # Momento interno en el final del segmento
        self.ui: float | None = None  # Desplazamiento axial en el inicio del segmento de la viga
        self.uj: float | None = None  # Desplazamiento axial en el final del segmento de la viga
        self.vi: float | None = None  # Desplazamiento transversal en el inicio del segmento de la viga
        self.vj: float | None = None  # Desplazamiento transversal en el final del segmento de la viga
        self.thetai: float | None = None  # Pendiente en el inicio del segmento de la viga
        self.thetaj: float | None = None  # Pendiente en el final del segmento de la viga
        self.E: float | None = None  # Módulo de elasticidad
        self.I: float | None = None  # Inercia
        self.A: float | None = None  # Área
        self.phi: float | None = None  # Aporte por cortante
        # Coeficientes de integración
        self.C1: float | None = None
        self.C2: float | None = None
        self.C3: float | None = None
        self.C4: float | None = None

    def coefficients(self):
        """
        Retorna los coeficientes de integración
        """
        L = self.Length()
        qi = self.qi
        qj = self.qj
        E = self.E
        I = self.I
        phi = self.phi
        vi = self.vi
        vj = self.vj
        thetai = self.thetai
        thetaj = self.thetaj

        A = -(qj - qi) / L
        B = -qi

        M = np.array([
            [0,                     0,          0, 1],
            [0,                     0,          1, 0],
            [L**3 * (2 - phi) / 12, L**2 / 2,   L, 1],
            [L**2 / 2,              L,          1, 0],
        ])

        N = np.array([
            E * I * vi,
            E * I * thetai,
            E * I * vj + A * L**5 * (0.6 - phi) / 72 + B * L**4 * (1 - phi) / 24,
            E * I * thetaj + A * L**4 / 24 + B * L**3 / 6,
        ])

        C = np.linalg.solve(M, N)

        self.C1 = C[0]
        self.C2 = C[1]
        self.C3 = C[2]
        self.C4 = C[3]

        return tuple(map(float, C))

    def Length(self):
        """
        Retorna la longitud del segmento
        """

        return self.xj - self.xi

    def shear(self, x):
        """
        Retorna la fuerza cortante en un punto 'x' del segmento
        """
        Vi = self.Vi
        qi = self.qi
        qj = self.qj
        L = self.Length()
        A = (qj - qi)/L
        B = qi

        return Vi + B*x + A*x**2/2

    def moment(self, x):
        """
        Retorna el momento en un punto 'x' del segmento
        """
        Vi = self.Vi
        Mi = self.Mi
        qi = self.qi
        qj = self.qj
        L = self.Length()
        A = (qj - qi)/L
        B = qi

        M = Mi - Vi*x - B*x**2/2 - A*x**3/6

        return M

    def axial(self, x):
        """
        Retorna la fuerza axial en un punto 'x' del segmento
        """
        Pi = self.Pi
        Pj = self.Pj
        # pi = self.pi
        # pj = self.pj
        L = self.Length()
        # A = (pj - pi)/L
        # B = pi

        # P = Pi + B*x + A*x**2/2

        A = (Pj + Pi) / L
        B = Pi

        # * REFACTORIZAR
        # TODO: P = - A*x + B
        P = - A*x + B

        return P

    def slope(self, x):
        """
        Retorna la pendiente de la curva elástica en cualquier punto `x` a lo largo del segmento.
        """

        # Vi = self.Vi
        # Mi = self.Mi
        qi = self.qi
        qj = self.qj
        # thetai = self.thetai
        L = self.Length()
        EI = self.E * self.I
        A = (qj - qi)/L
        B = qi
        # * sin tener encueta las deformaciones por corte:
        # ! theta_x = 1/EI * (thetai*EI - Mi*x + Vi*x**2/2 + B*x**3/6 + A*x**4/24)
        # ? theta_x = 1/EI * (thetai*EI - Mi*x + Vi*x**2/2 + B*x**3/6 + A*x**4/24)
        # TODO: theta_x = 1/EI * (thetai*EI - Mi*x + Vi*x**2/2 + B*x**3/6 + A*x**4/24)

        C1 = self.C1
        C2 = self.C2
        C3 = self.C3

        theta_x = 1/EI * (A*x**4/24 + B*x**3/6 + C1*x**2/2 + C2*x + C3)

        return theta_x

    def deflection(self, x):

        # Vi = self.Vi
        # Mi = self.Mi
        qi = self.qi
        qj = self.qj
        # thetai = self.thetai
        # vi = self.vi
        L = self.Length()
        EI = self.E * self.I
        A = (qj - qi)/L
        B = qi

        # * sin tener encueta las deformaciones por corte:
        # TODO: 1/EI * (vi + thetai*x - x**2*(Mi)/2 + Vi*x**3/6 + B*x**4/24 + A*x**5/120)

        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        C4 = self.C4

        phi = self.phi  # aporte por cortante (phi)

        term1 = A * L**2 * x**3 * (0.6 * (x / L)**2 - phi) / 72
        term2 = B * L**2 * x**2 * ((x / L)**2 - phi) / 24
        term3 = C1 * x * L**2 * (2 * (x / L)**2 - phi) / 12
        term4 = C2 * x**2 / 2
        term5 = C3 * x
        term6 = C4

        return (term1 + term2 + term3 + term4 + term5 + term6) / (EI)


    def process_builder(self, member: "Member", results: "Dict[str, np.ndarray]", pattern_name: str) -> None:
        self.xi =0
        self.xj = member.length()
        self.qi = member.get_distributed_load(pattern_name).q_i
        self.qj = member.get_distributed_load(pattern_name).q_j
        self.pi = member.get_distributed_load(pattern_name).p_i
        self.pj = member.get_distributed_load(pattern_name).p_j
        self.Pi = results["internal_forces"][0]
        self.Pj = results["internal_forces"][3]
        self.Vi = results["internal_forces"][1]
        self.Vj = results["internal_forces"][4]
        self.Mi = results["internal_forces"][2]
        self.Mj = results["internal_forces"][5]
        self.ui = results["displacements"][0]
        self.uj = results["displacements"][3]
        self.vi = results["displacements"][1]
        self.vj = results["displacements"][4]
        self.thetai = results["displacements"][2]
        self.thetaj = results["displacements"][5]
        self.E = member.section.E()
        self.I = member.section.I()
        self.A = member.section.A()
        self.phi = member.phi()