from milcapy.postprocess.member import BeamSeg
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from milcapy.utils.element import q_phi



if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from milcapy.core.results import Results


@dataclass
class PostProcessingOptions:
    """Opciones para el post-procesamiento de resultados estructurales."""

    factor: float  # Factor de escala para la visualización de resultados
    n: int         # Número de puntos para discretizar los elementos


class PostProcessing:   # para un solo load pattern
    """Clase para el post-procesamiento de resultados estructurales."""

    def __init__(
        self,
        model: "SystemMilcaModel",
        results: "Results",                 # ya viene con U, R del modelo para el LP activo
        options: "PostProcessingOptions",
        load_pattern_name: str
    ) -> None:
        """
        Inicializa el post-procesamiento para un sistema estructural.

        Args:
            model: Sistema estructural analizado
            results: Resultados del análisis
            options: Opciones de post-procesamiento
            load_pattern_name: Nombre del patrón de carga
        """
        self.model = model
        self.results = results
        self.options = options
        self.load_pattern_name = load_pattern_name

        self.reactions = self.results.model["reactions"]
        self.displacements = self.results.model["displacements"]

    def process_displacements_for_nodes(self) -> None:
        """Almacena las desplazamientos de los nodos en el objeto Results."""
        for id, node in self.model.nodes.items():
            array_displacements = self.displacements[node.dofs-1]
            self.results.set_node_displacements(id, array_displacements)

    def process_reactions_for_nodes(self) -> None:
        """Almacena las reacciones de los nodos en el objeto Results."""
        for id, node in self.model.nodes.items():
            if np.all(self.reactions[node.dofs - 1] == 0):
                pass
            else:
                array_reactions = self.reactions[node.dofs-1]
                self.results.set_node_reactions(id, array_reactions)

    def process_displacements_for_members(self) -> None:
        """Almacena las desplazamientos de los miembros en sistema local en el objeto Results."""
        for id, member in self.model.members.items():
            global_displacements = self.displacements[member.dofs-1]
            local_disp = np.dot(member.transformation_matrix(), global_displacements)
            local_disp_flex = member.H() @ local_disp
            self.results.set_member_displacements(id, local_disp_flex)

    def process_internal_forces_for_members(self) -> None:
        """Almacena las fuerzas internas de los miembros en sistema local en el objeto Results."""
        for id, member in self.model.members.items():
            local_displacements = self.results.get_member_displacements(id)
            load = member.get_distributed_load(self.load_pattern_name)
            L = member.length()
            la, lb = member.la or 0, member.lb or 0
            qi, qj, pi, pj = load.q_i, load.q_j, load.p_i, load.p_j


            a = (qj - qi) / L
            b = qi

            c = (pj - pi) / L
            d = pi

            qa = a*la + b
            qb = a*(L -lb) + b

            pa = c*la + d
            pb = c*(L -lb) + d


            load_vector = q_phi(member.le(), member.phi(), qa, qb, pa, pb)
            H = member.H()
            H_inv = np.linalg.pinv(H)
            HT_pinv = np.linalg.pinv(H.T)
            stiffness_matrix = HT_pinv @ member.local_stiffness_matrix() @ H_inv
            array_internal_forces = np.dot(stiffness_matrix, local_displacements) - load_vector
            self.results.set_member_internal_forces(id, array_internal_forces)

    def post_process_for_members(self) -> None:
        """Almacena todos los resultados para cada miembro en el objeto Results."""
        n = self.options.n
        calculator = BeamSeg()

        for id, member in self.model.members.items():
            result = self.results.get_results_member(id)
            calculator.process_builder(member, result, self.load_pattern_name)
            calculator.coefficients()

            x_val = np.linspace(0, member.le(), n)

            array_axial_force = np.zeros(n)
            array_shear_force = np.zeros(n)
            array_bending_moment = np.zeros(n)
            array_slope = np.zeros(n)
            array_deflection = np.zeros(n)

            for i, x in enumerate(x_val):
                # Calcular fuerzas axiales
                array_axial_force[i] = calculator.axial(x)

                # Calcular fuerzas cortantes
                array_shear_force[i] = calculator.shear(x)

                # Calcular momentos de flexión
                array_bending_moment[i] = calculator.moment(x)

                # Calcular pendientes
                array_slope[i] = calculator.slope(x)

                # Calcular deflexiones
                array_deflection[i] = calculator.deflection(x)

            self.results.set_member_axial_force(id, array_axial_force)
            self.results.set_member_shear_force(id, array_shear_force)
            self.results.set_member_bending_moment(id, array_bending_moment)
            self.results.set_member_slope(id, array_slope)
            self.results.set_member_deflection(id, array_deflection)
