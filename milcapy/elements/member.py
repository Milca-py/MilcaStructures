import numpy as np
from typing import TYPE_CHECKING, Optional, Dict, Tuple
from milcapy.utils.geometry import angle_x_axis
from milcapy.loads.load import DistributedLoad
from milcapy.utils.types import BeamTheoriesType
from milcapy.utils.element import (
    local_stiffness_matrix,
    transformation_matrix,
    length_offset_transformation_matrix,
    length_offset_q,
    q_phi
)

from milcapy.loads.load import FrameRelease

if TYPE_CHECKING:
    from milcapy.core.node import Node
    from milcapy.utils.types import MemberType
    from milcapy.section.section import Section


class Member:
    """
    Clase que representa un miembro estructural.
    """

    def __init__(
        self,
        id:           int,
        node_i:      "Node",
        node_j:      "Node",
        section:     "Section",
        member_type: "MemberType",
        beam_theory: "BeamTheoriesType",
    ) -> None:
        """
        Inicializa un objeto que representa un marco estructural.

        Args:
            id (int): Identificador del miembro.
            node_i (Node): Nodo inicial.
            node_j (Node): Nodo final.
            section (Section): Sección del miembro.
            member_type (MemberType): Tipo de miembro.
            beam_theory (BeamTheoriesType): Teoría de vigas.
        """

        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.section = section
        self.member_type = member_type
        self.beam_theory = beam_theory

        # Mapa de grados de libertad
        self.dofs: np.ndarray = np.concatenate([node_i.dofs, node_j.dofs]) # [dofs node_i, dofs node_j]

        # Cargas distribuidas
        self.distributed_load: Dict[str, DistributedLoad] = {}  # {pattern_name: DistributedLoad}

        # Patrón de carga actual
        self.current_load_pattern: Optional[str] = None

        # Topicos
        self.la:  Optional[float] = None    # Longitud del brazo rigido inicial
        self.lb:  Optional[float] = None    # Longitud del brazo rigido final
        self.qla: Optional[bool]  = None    # Indica si hay cargas en el brazo rigido inicial
        self.qlb: Optional[bool]  = None    # Indica si hay cargas en el brazo rigido final
        self.fla: Optional[float] = None    # Factor de longitud del brazo rigido inicial
        self.flb: Optional[float] = None    # Factor de longitud del brazo rigido final

        # Liberaciones
        self.release: Optional[FrameRelease] = None

    def length(self) -> float:
        """
        Longitud del miembro.
        """
        return (self.node_i.vertex.distance_to(self.node_j.vertex))

    def angle_x(self) -> float:
        """
        Ángulo del miembro respecto al eje X del sistema global.
        """
        return angle_x_axis(
            self.node_j.vertex.x - self.node_i.vertex.x,
            self.node_j.vertex.y - self.node_i.vertex.y
        )

    def phi(self) -> float:
        """
        Ángulo de corte para efectos de deformación por cortante (parámetro phi).

        phi = (12 * E * I) / (L**2 * A * k * G)
        """
        if self.beam_theory == BeamTheoriesType.EULER_BERNOULLI:
            return 0

        elif self.beam_theory == BeamTheoriesType.TIMOSHENKO:

            laef, lbef = self.la_lb_efect()
            E = self.section.E()
            I = self.section.I()
            L = self.length() - laef - lbef
            A = self.section.A()
            k = self.section.k()
            G = self.section.G()

            return (12 * E * I) / (L**2 * A * k * G) if k != 0 else 0

    def la_lb_efect(self) -> Tuple[float, float]:
        """
        Longitudes efectivas de los brazos rigidos.
        """
        la = self.la or 0
        fla = self.fla or 1
        laef = la * fla

        lb = self.lb or 0
        flb = self.flb or 1
        lbef = lb * flb

        return laef, lbef

    def set_current_load_pattern(self, load_pattern_name: str) -> None:
        """
        Establece el patrón de carga actual del miembro.
        """
        self.current_load_pattern = load_pattern_name

    def set_distributed_load(self, load: DistributedLoad) -> None:
        """
        Asigna una carga distribuida al miembro para el patrón de carga actual.

        Args:
            load (DistributedLoad): Carga distribuida a aplicar.
        """
        if self.current_load_pattern is None:
            raise ValueError("Debe establecer un patrón de carga actual antes de asignar una carga distribuida.")
        self.distributed_load[self.current_load_pattern] = load

    def get_distributed_load(self, load_pattern_name: str) -> Optional["DistributedLoad"]:
        """
        Obtiene la carga distribuida para el patrón de carga actual.

        Args:
            load_pattern_name (str): Nombre del patrón de carga.

        Returns:
            Optional["DistributedLoad"]: Carga distribuida.
        """
        load = self.distributed_load.get(load_pattern_name, None)
        if load is None:
            return DistributedLoad()
        return load

    def transformation_matrix(self) -> np.ndarray:
        """
        Calcula la matriz de transformación del miembro.

        Returns:
            np.ndarray: Matriz de transformación.
        """
        return transformation_matrix(
            angle=self.angle_x()
        )

    def length_offset_transformation_matrix(self) -> np.ndarray:
        """
        Matriz de transformación para miembros con brazos rígidos (identidad si no hay brazos).
        """
        if self.la is None and self.lb is None:
            return np.eye(6)  # Matriz identidad 6x6
        laef, lbef = self.la_lb_efect()
        H = length_offset_transformation_matrix(laef, lbef)
        return H

    def k_local(self) -> np.ndarray:
        """
        Calcula la matriz de rigidez local del miembro.

        Returns:
            np.ndarray: Matriz de rigidez en local.
        """
        if self.la is None and self.lb is None:
            k = local_stiffness_matrix(
                E=self.section.E(),
                I=self.section.I(),
                A=self.section.A(),
                L=self.length(),
                le=self.length(),
                phi=self.phi(),
            )
            return k
        else:
            laef, lbef = self.la_lb_efect()
            H = self.length_offset_transformation_matrix()
            k = local_stiffness_matrix(
                E=self.section.E(),
                I=self.section.I(),
                A=self.section.A(),
                L=self.length(),
                le=self.length() - laef - lbef,
                phi=self.phi(),
            )
            return H.T @ k @ H

    def local_stiffness_matrix(self) -> np.ndarray:
        """
        Calcula la matriz de rigidez local del miembro.

        Returns:
            np.ndarray: Matriz de rigidez en local.
        """
        k = self.k_local()
        if self.release is not None:
            freeDof = self.release.get_dof_release()
            for dof in freeDof:
                k = self.apply_releases_for_stiffness_matrix(dof=dof, k=k)
        return k

    def apply_releases_for_stiffness_matrix(self, dof: int, k: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de rigidez local del miembro con liberaciones.

        Args:
            dof (int): Grado de libertad a liberar.
            k (np.ndarray): Matriz de rigidez local del miembro.

        Returns:
            np.ndarray: Matriz de rigidez local del miembro con liberaciones.
        """

        # Todos los DOF posibles en 6x6
        allDof = np.arange(6)

        # Submatrices
        k1 = k[np.ix_(allDof, [dof])]
        k2 = k[np.ix_([dof], allDof)]
        k3 = k[np.ix_([dof], [dof])]

        # Reducción de rigidez con liberaciones
        kp = k - k1 @ (1/k3) @ k2
        return kp


    def flexible_stiffness_matrix(self) -> np.ndarray:
        """
        Calcula la matriz de rigidez de la parte flexible del miembro.

        Returns:
            np.ndarray: Matriz de rigidez de la parte flexible del miembro.
        """
        laef, lbef = self.la_lb_efect()
        k = local_stiffness_matrix(
            E=self.section.E(),
            I=self.section.I(),
            A=self.section.A(),
            L=self.length(),
            le=self.length() - laef - lbef,
            phi=self.phi(),
        )
        return k

    def q_local(self) -> np.ndarray:
        """
        Calcula el vector de cargas distribuidas equivalentes en sistema local.

        Returns:
            np.ndarray: Vector de cargas distribuidas equivalentes en sistema local.
        """
        load = self.get_distributed_load(self.current_load_pattern)
        if self.la is None and self.lb is None:
            q = q_phi(
                L=self.length(),
                phi=self.phi(),
                qi=load.q_i,
                qj=load.q_j,
                pi=load.p_i,
                pj=load.p_j
            )
            return q
        else:
            laef, lbef = self.la_lb_efect()
            q = length_offset_q(
                self.length(),
                self.phi(),
                load.q_i,
                load.q_j,
                load.p_i,
                load.p_j,
                laef,
                lbef,
                self.qla,
                self.qlb,
            )
            return q

    def local_load_vector(self) -> np.ndarray:
        """
        Calcula el vector de cargas distribuidas equivalentes en sistema local.

        Returns:
            np.ndarray: Vector de cargas distribuidas equivalentes en sistema local.
        """
        q = self.q_local()
        k = self.k_local()
        if self.release is not None:
            freeDof = self.release.get_dof_release()
            for dof in freeDof:
                q = self.apply_releases_for_load_vector(dof=dof, q=q, k=k)
                k = self.apply_releases_for_stiffness_matrix(dof=dof, k=k)
        return q

    def apply_releases_for_load_vector(
        self,
        dof: int,
        q: np.ndarray,
        k: np.ndarray
    ) -> np.ndarray:
        """
        Aplica liberaciones al vector de cargas.

        Args:
            dof (int): Grado de libertad a liberar.
            q (np.ndarray): Vector de cargas.
            k (np.ndarray): Matriz de rigidez local del miembro.

        Returns:
            np.ndarray: Vector de cargas con liberaciones.
        """
        vecudof = np.zeros_like(q)
        if k[dof, dof] != 0.0:
            vecudof[dof] = q[dof] / k[dof, dof]
        else:
            raise ZeroDivisionError(f"El término k[{dof},{dof}] es cero, no se puede aplicar release.")

        # Nuevo vector de cargas con liberación
        qp = q - k @ vecudof
        return qp



    def global_stiffness_matrix(self) -> np.ndarray:
        """
        Calcula la matriz de rigidez global del miembro.

        Returns:
            np.ndarray: Matriz de rigidez global.
        """
        T = self.transformation_matrix()
        ke = self.local_stiffness_matrix()

        return T.T @ ke @ T

    def global_load_vector(self) -> np.ndarray:
        """
        Calcula el vector de fuerzas global del miembro para el patron de carga actual.

        Returns:
            np.ndarray: Vector de fuerzas global.
        """
        if self.current_load_pattern is None:
            raise ValueError("Debe establecer un patrón de carga actual antes de calcular el vector de fuerzas global.")
        T = self.transformation_matrix()
        q = self.local_load_vector()

        return T.T @ q

    def add_releases(
        self,
        uxi: bool=False,
        uyi: bool=False,
        rzi: bool=False,
        uxj: bool=False,
        uyj: bool=False,
        rzj: bool=False
    ) -> None:
        """
        Asigna liberaciones a un miembro.

        Args:
            uxi (bool, opcional): Liberación de U en el nodo inicial en la dirección x. Default es False.
            uyi (bool, opcional): Liberación de U en el nodo inicial en la dirección y. Default es False.
            rzi (bool, opcional): Liberación de R en el nodo inicial en la dirección z. Default es False.
            uxj (bool, opcional): Liberación de U en el nodo final en la dirección x. Default es False.
            uyj (bool, opcional): Liberación de U en el nodo final en la dirección y. Default es False.
            rzj (bool, opcional): Liberación de R en el nodo final en la dirección z. Default es False.
        """
        self.release = FrameRelease(uxi=uxi, uyi=uyi, rzi=rzi, uxj=uxj, uyj=uyj, rzj=rzj)