from milcapy.elements.member import Member
from typing import Dict, Tuple
import numpy as np
from milcapy.utils.element import q_phi

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

        # TOPICOS
        # Brazos rigidos
        self.la: float | None = None
        self.lb: float | None = None
        self.qla: bool | None = None
        self.qlb: bool | None = None

    def coefficients(self):
        """
        Retorna los coeficientes de integración
        """
        L = self.le()
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

    def length(self):
        """
        retorna la longitud total del elemento
        """
        return self.xj - self.xi

    def le(self):
        """
        Retorna la longitud del segmento flexible
        """

        return self.xj - self.xi - (self.la or 0) - (self.lb or 0)


    def shear(self, x):
        """
        Retorna la fuerza cortante en un punto 'x' del segmento
        """

        la, lb, le = self.la or 0, self.lb or 0, self.le()
        if (self.la or self.lb) and (0 <= x < la or le+la < x <= la+le+lb):
            return 0

        Vi = self.Vi
        qi = self.qi
        qj = self.qj
        L = self.le()
        A = (qj - qi)/L
        B = qi
        x = x - la

        return Vi + B*x + A*x**2/2

    def moment(self, x):
        """
        Retorna el momento en un punto 'x' del segmento
        """

        la, lb, le = self.la or 0, self.lb or 0, self.le()
        if (self.la or self.lb) and (0 <= x < la or le+la < x <= la+le+lb):
            return 0

        Vi = self.Vi
        Mi = self.Mi
        qi = self.qi
        qj = self.qj
        L = self.le()
        A = (qj - qi)/L
        B = qi
        x = x - la

        M = Mi - Vi*x - B*x**2/2 - A*x**3/6

        return M

    def axial(self, x):
        """
        Retorna la fuerza axial en un punto 'x' del segmento
        """

        la, lb, le = self.la or 0, self.lb or 0, self.le()
        if (self.la or self.lb) and (0 <= x < la or le+la < x <= la+le+lb):
            return 0

        C1 = self.Pi
        A = -(self.pj - self.pi)/le
        B = -self.pi
        x = x - la

        N = - A*x**2/2 - B*x + C1

        return N

    def axial_displacement(self, x):
        """
        Retorna el desplazamiento axial en un punto 'x' del segmento
        """

        la, lb, le = self.la or 0, self.lb or 0, self.le()
        if (self.la or self.lb) and (0 <= x < la):
            return self.ui
        if (self.la or self.lb) and (le+la < x <= la+le+lb):
            return self.uj

        L = self.le()
        EA = self.E * self.A
        EAoverL = EA / L
        C2 = self.ui * EA
        A = (self.pi - self.pj)/L
        B = self.pi
        C1 = EAoverL * (self.uj - self.ui) + L * (B - A * L/3)
        x = x - la

        u = 1/EA * (- A*x**3/6 - B*x**2/2 + C1*x + C2)

        return u

    def slope(self, x):
        """
        Retorna la pendiente de la curva elástica en cualquier punto `x` a lo largo del segmento.
        """

        la, lb, le = self.la or 0, self.lb or 0, self.le()
        if (self.la or self.lb) and (0 <= x < la):
            return self.thetai
        if (self.la or self.lb) and (le+la < x <= la+le+lb):
            return self.thetaj

        qi = self.qi
        qj = self.qj
        L = self.le()
        EI = self.E * self.I
        A = (qj - qi)/L
        B = qi

        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        x = x - la

        theta_x = 1/EI * (A*x**4/24 + B*x**3/6 + C1*x**2/2 + C2*x + C3)

        return theta_x

    def deflection(self, x):
        import math
        la, lb, le = self.la or 0, self.lb or 0, self.le()
        if (self.la or self.lb) and (0 <= x < la):
            vi = self.vi - la * self.thetai
            return vi + x * self.thetai
        if (self.la or self.lb) and (le+la < x <= la+le+lb):
            return self.vj + (x - le - la) * self.thetaj

        qi = self.qi
        qj = self.qj
        L = self.le()
        EI = self.E * self.I
        A = (qj - qi)/L
        B = qi
        x = x - la

        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        C4 = self.C4

        phi = self.phi

        term1 = A * L**2 * x**3 * (0.6 * (x / L)**2 - phi) / 72
        term2 = B * L**2 * x**2 * ((x / L)**2 - phi) / 24
        term3 = C1 * x * L**2 * (2 * (x / L)**2 - phi) / 12
        term4 = C2 * x**2 / 2
        term5 = C3 * x
        term6 = C4

        return (term1 + term2 + term3 + term4 + term5 + term6) / (EI)


    def process_builder(self, member: "Member", results: "Dict[str, np.ndarray]", pattern_name: str) -> None:

        self.la = member.la
        self.lb = member.lb
        self.qla = member.qla
        self.qlb = member.qlb

        self.xi =0
        self.xj = member.length()

        if self.la or self.lb:
            local_disp = results["displacements"]
            local_displacements_flex = member.length_offset_transformation_matrix() @ local_disp

            load = member.get_distributed_load(pattern_name)
            L = member.length()
            le = L - (self.la or 0) - (self.lb or 0)
            la, lb = self.la or 0, self.lb or 0
            qi, qj, pi, pj = load.q_i, load.q_j, load.p_i, load.p_j


            a = (qj - qi) / L
            b = qi

            c = (pj - pi) / L
            d = pi

            qa = a*la + b
            qb = a*(L -lb) + b

            pa = c*la + d
            pb = c*(L -lb) + d

            load_vector = q_phi(le, member.phi(), qa, qb, pa, pb)
            stiffness_matrix = member.flexible_stiffness_matrix()
            internal_forces_flex = stiffness_matrix @ local_displacements_flex - load_vector
            fixed_end_forces = [qa, qb, pa, pb]

        else:
            load = member.get_distributed_load(pattern_name)
            qi, qj, pi, pj = load.q_i, load.q_j, load.p_i, load.p_j

            local_displacements_flex = results["displacements"]
            internal_forces_flex = results["internal_forces"]
            fixed_end_forces = [qi, qj, pi, pj]

        # TODAS LAS ACCIONES Y REACCIONES EN LOS EXTREMOS DE LA PARTE FLEXIBLE
        self.qi = fixed_end_forces[0]
        self.qj = fixed_end_forces[1]
        self.pi = fixed_end_forces[2]
        self.pj = fixed_end_forces[3]
        self.Pi = internal_forces_flex[0]
        self.Pj = internal_forces_flex[3]
        self.Vi = internal_forces_flex[1]
        self.Vj = internal_forces_flex[4]
        self.Mi = internal_forces_flex[2]
        self.Mj = internal_forces_flex[5]
        self.ui = local_displacements_flex[0]
        self.uj = local_displacements_flex[3]
        self.vi = local_displacements_flex[1]
        self.vj = local_displacements_flex[4]
        self.thetai = local_displacements_flex[2]
        self.thetaj = local_displacements_flex[5]
        self.E = member.section.E()
        self.I = member.section.I()
        self.A = member.section.A()
        self.phi = member.phi()


def deformed_shape(member: "Member", results: dict, escale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula las coordenada de la deformada en sistema local"""
    L = member.length()

    flecha = results["deflections"] * escale

    x = results["x_val"]
    x_val = results["axial_displacements"]*escale + x

    return x_val, flecha