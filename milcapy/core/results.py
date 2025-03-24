from __future__ import annotations
from typing import Dict
import numpy as np

class Results:
    """Clase que almacena los resultados de un análisis de un Load Pattern.
    La estructura de los resultados es la siguiente:

    model: {
        "displacements": np.ndarray,
        "reactions": np.ndarray
    }

    nodes: {
        node_id: {
            "displacements": np.ndarray,
            "reactions": np.ndarray
        }
    }

    members: {
        member_id: {
            "displacements": np.ndarray,
            "internal_forces": np.ndarray,
            "axial_forces": np.ndarray,
            "shear_forces": np.ndarray,
            "bending_moments": np.ndarray,
            "slopes": np.ndarray,
            "deflections": np.ndarray
        }
    }"""
    def __init__(self):
        self.nodes: Dict[int, Dict[str, np.ndarray]] = {}
        self.members: Dict[int, Dict[str, np.ndarray]] = {}
        self.model: Dict[str, np.ndarray] = {}

    def set_model_displacements(self, displacements: np.ndarray) -> None:
        self.model["displacements"] = displacements

    def set_model_reactions(self, reactions: np.ndarray) -> None:
        self.model["reactions"] = reactions

    def set_node_displacement(self, node_id: int, displacement: np.ndarray) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"displacements": np.zeros(3), "reactions": np.zeros(3)}
        self.nodes[node_id]["displacements"] = displacement

    def set_node_reaction(self, node_id: int, reaction: np.ndarray) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"displacements": np.zeros(3), "reactions": np.zeros(3)}
        self.nodes[node_id]["reactions"] = reaction

    def set_member_displacement(self, member_id: int, displacement: np.ndarray) -> None:
        if member_id not in self.members:
            self.members[member_id] = {"displacements": np.zeros(6), "internal_forces": np.zeros(6)}
        self.members[member_id]["displacements"] = displacement

    def set_member_internal_forces(self, member_id: int, internal_forces: np.ndarray) -> None:
        if member_id not in self.members:
            self.members[member_id] = {"displacements": np.zeros(6), "internal_forces": np.zeros(6)}
        self.members[member_id]["internal_forces"] = internal_forces

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

    def get_model_displacements(self) -> np.ndarray:
        return self.model["displacements"]

    def get_model_reactions(self) -> np.ndarray:
        return self.model["reactions"]

    def get_node_displacement(self, node_id: int) -> np.ndarray:
        return self.nodes[node_id]["displacements"]

    def get_node_reaction(self, node_id: int) -> np.ndarray:
        return self.nodes[node_id]["reactions"]

    def get_member_displacement(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["displacements"]

    def get_member_internal_forces(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["internal_forces"]

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

    def get_results_node(self, node_id: int) -> Dict[str, np.ndarray]:
        return self.nodes[node_id]

    def get_results_member(self, member_id: int) -> Dict[str, np.ndarray]:
        return self.members[member_id]

    def get_results_model(self) -> Dict[str, np.ndarray]:
        return self.model

    def get_results(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self.results
