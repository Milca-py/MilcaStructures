from __future__ import annotations
from typing import Dict
import numpy as np

class Results:
    """Clase que almacena los resultados de un análisis de un Load Pattern.
    La estructura de los resultados es la siguiente:

    self.model: {
        "displacements": np.ndarray,
        "reactions": np.ndarray
            }

    self.nodes: {
        node_id: {
            "displacements": np.ndarray,
            "reactions": np.ndarray
                }
            }

    self.members: {
        "x_val": np.ndarray,
        member_id: {
            "displacements": np.ndarray,
            "internal_forces": np.ndarray,
            "axial_forces": np.ndarray,
            "axial_displacements": np.ndarray,
            "shear_forces": np.ndarray,
            "bending_moments": np.ndarray,
            "slopes": np.ndarray,
            "deflections": np.ndarray
                    }
            }

    self.CST: {
        "displacements": np.ndarray,
        "strains": np.ndarray,
        "stresses": np.ndarray,
        }

    self.membrane_q6: {
        "displacements": np.ndarray,
        "strains": np.ndarray,
        "stresses": np.ndarray,
        }

    self.membrane_q6i: {
        "displacements": np.ndarray,
        "strains": np.ndarray,
        "stresses": np.ndarray,
        }

    self.trusses: {
        "displacements": np.ndarray,
        "internal_forces": np.ndarray,
        "axial_forces": np.ndarray,
        "axial_displacements": np.ndarray,
            }
    """

    def __init__(self):
        self.nodes: Dict[int, Dict[str, np.ndarray]] = {}
        self.members: Dict[int, Dict[str, np.ndarray]] = {}
        self.CST: Dict[int, np.ndarray] = {}
        self.membrane_q6: Dict[int, np.ndarray] = {}
        self.membrane_q6i: Dict[int, np.ndarray] = {}
        self.trusses: Dict[int, Dict[str, np.ndarray]] = {}
        self.model: Dict[str, np.ndarray] = {}

    def set_model_displacements(self, displacements: np.ndarray) -> None:
        self.model["displacements"] = displacements

    def set_model_reactions(self, reactions: np.ndarray) -> None:
        self.model["reactions"] = reactions

    def set_node_displacements(self, node_id: int, displacement: np.ndarray) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"displacements": np.zeros(3), "reactions": np.zeros(3)}
        self.nodes[node_id]["displacements"] = displacement

    def set_node_reactions(self, node_id: int, reaction: np.ndarray) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"displacements": np.zeros(3), "reactions": np.zeros(3)}
        self.nodes[node_id]["reactions"] = reaction

    def set_member_displacements(self, member_id: int, displacement: np.ndarray) -> None:
        if member_id not in self.members:
            self.members[member_id] = {"displacements": np.zeros(6), "internal_forces": np.zeros(6)}
        self.members[member_id]["displacements"] = displacement

    def set_member_internal_forces(self, member_id: int, internal_forces: np.ndarray) -> None:
        if member_id not in self.members:
            self.members[member_id] = {"displacements": np.zeros(6), "internal_forces": np.zeros(6)}
        self.members[member_id]["internal_forces"] = internal_forces

    def set_x_val(self, member_id: int, x_val: np.ndarray) -> None:
        self.members[member_id]["x_val"] = x_val

    def set_member_axial_force(self, member_id: int, axial_force: np.ndarray) -> None:
        self.members[member_id]["axial_forces"] = axial_force

    def set_member_shear_force(self, member_id: int, shear_force: np.ndarray) -> None:
        self.members[member_id]["shear_forces"] = shear_force

    def set_member_bending_moment(self, member_id: int, bending_moment: np.ndarray) -> None:
        self.members[member_id]["bending_moments"] = bending_moment

    def set_member_deflection(self, member_id: int, deflection: np.ndarray) -> None:
        self.members[member_id]["deflections"] = deflection

    def set_member_slope(self, member_id: int, slope: np.ndarray) -> None:
        self.members[member_id]["slopes"] = slope

    def set_member_axial_displacement(self, member_id: int, axial_displacement: np.ndarray) -> None:
        self.members[member_id]["axial_displacements"] = axial_displacement

    def get_model_displacements(self) -> np.ndarray:
        return self.model["displacements"]

    def get_model_reactions(self) -> np.ndarray:
        return self.model["reactions"]

    def get_node_displacements(self, node_id: int) -> np.ndarray:
        return self.nodes[node_id]["displacements"]

    def get_node_reactions(self, node_id: int) -> np.ndarray:
        return self.nodes[node_id]["reactions"]

    def get_member_displacements(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["displacements"]

    def get_member_internal_forces(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["internal_forces"]

    def get_member_x_val(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["x_val"]

    def get_member_axial_force(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["axial_forces"]

    def get_member_shear_force(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["shear_forces"]

    def get_member_bending_moment(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["bending_moments"]

    def get_member_deflection(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["deflections"]

    def get_member_slope(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["slopes"]

    def get_member_axial_displacement(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["axial_displacements"]

    def get_results_node(self, node_id: int) -> Dict[str, np.ndarray]:
        return self.nodes[node_id]

    def get_results_member(self, member_id: int) -> Dict[str, np.ndarray]:
        return self.members[member_id]

    def get_results_model(self) -> Dict[str, np.ndarray]:
        return self.model

    def get_results(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self.results

    def set_cst_displacements(self, cst_id: int, displacement: np.ndarray) -> None:
        if cst_id not in self.CST:
            self.CST[cst_id] = {"displacements": np.zeros(6)}
        self.CST[cst_id]["displacements"] = displacement

    def get_cst_displacements(self, cst_id: int) -> np.ndarray:
        return self.CST[cst_id]["displacements"]

    def set_cst_strains(self, cst_id: int, strains: np.ndarray) -> None:
        self.CST[cst_id]["strains"] = strains

    def set_cst_stresses(self,  cst_id: int, stresses: np.ndarray) -> None:
            self.CST[cst_id]["stresses"] = stresses

    def get_cst_strains(self, cst_id: int) -> None:
        return self.CST[cst_id]["strains"]

    def get_cst_stresses(self, cst_id: int) -> None:
        return self.CST[cst_id]["stresses"]

    def set_membrane_q6_displacements(self, membrane_q6_id: int, displacement: np.ndarray) -> None:
        if membrane_q6_id not in self.membrane_q6:
            self.membrane_q6[membrane_q6_id] = {"displacements": np.zeros(6)}
        self.membrane_q6[membrane_q6_id]["displacements"] = displacement

    def get_membrane_q6_displacements(self, membrane_q6_id: int) -> np.ndarray:
        return self.membrane_q6[membrane_q6_id]["displacements"]

    def set_membrane_q6_strains(self, membrane_q6_id: int, strains: np.ndarray) -> None:
        pass

    def set_membrane_q6_stresses(self, membrane_q6_id: int, stresses: np.ndarray) -> None:
        pass

    def get_membrane_q6_strains(self, membrane_q6_id: int) -> None:
        pass

    def get_membrane_q6_stresses(self, membrane_q6_id: int) -> None:
        pass

    def set_membrane_q6i_displacements(self, membrane_q6i_id: int, displacement: np.ndarray) -> None:
        if membrane_q6i_id not in self.membrane_q6i:
            self.membrane_q6i[membrane_q6i_id] = {"displacements": np.zeros(6)}
        self.membrane_q6i[membrane_q6i_id]["displacements"] = displacement

    def get_membrane_q6i_displacements(self, membrane_q6i_id: int) -> np.ndarray:
        return self.membrane_q6i[membrane_q6i_id]["displacements"]

    def set_membrane_q6i_strains(self, membrane_q6i_id: int, strains: np.ndarray) -> None:
        pass

    def set_membrane_q6i_stresses(self, membrane_q6i_id: int, stresses: np.ndarray) -> None:
        pass

    def get_membrane_q6i_strains(self, membrane_q6i_id: int) -> None:
        pass

    def get_membrane_q6i_stresses(self, membrane_q6i_id: int) -> None:
        pass

    def set_truss_displacements(self, truss_id: int, displacement: np.ndarray) -> None:
        if truss_id not in self.trusses:
            self.trusses[truss_id] = {"displacements": np.zeros(6)}
        self.trusses[truss_id]["displacements"] = displacement

    def get_truss_displacements(self, truss_id: int) -> np.ndarray:
        return self.trusses[truss_id]["displacements"]

    def set_truss_axial_force(self, truss_id: int, axial_force: np.ndarray) -> None:
        self.trusses[truss_id]["axial_forces"] = axial_force

    def get_truss_axial_force(self, truss_id: int) -> np.ndarray:
        return self.trusses[truss_id]["axial_forces"]

    def set_truss_internal_forces(self, truss_id: int, internal_forces: np.ndarray) -> None:
        self.trusses[truss_id]["internal_forces"] = internal_forces

    def get_truss_internal_forces(self, truss_id: int) -> np.ndarray:
        return self.trusses[truss_id]["internal_forces"]

    def get_results_truss(self, truss_id: int) -> Dict[str, np.ndarray]:
        return self.trusses[truss_id]

    def set_truss_axial_displacement(self, truss_id: int, axial_displacement: np.ndarray) -> None:
        self.trusses[truss_id]["axial_displacements"] = axial_displacement

    def get_truss_axial_displacement(self, truss_id: int) -> np.ndarray:
        return self.trusses[truss_id]["axial_displacements"]

