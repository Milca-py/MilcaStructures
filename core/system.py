from typing import TYPE_CHECKING, Dict
import numpy as np

from core.material import Material, GenericMaterial
from core.section import Section, RectangularSection
from core.node import Node
from core.element import Element

from loads import LoadPattern, PointLoad, DistributedLoad

from utils.custom_types import LoadPatternType, CoordinateSystemType, State, ElementType, to_enum
from utils.vertex import Vertex

if TYPE_CHECKING:
    from utils.custom_types import Restraints, VertexLike

from components.system_components import calculate_load_vector, assemble_global_stiffness_matrix, solve, process_conditions



class SystemMilcaModel:
    def __init__(self):
        # properties of the elements
        self.material_map: Dict[str, Material] = {}
        self.section_map: Dict[str, Section] = {}
        
        # elements of the model
        self.node_map: Dict[int, Node] = {}
        self.element_map: Dict[int, Element] = {}
        
        # collections of loads
        self.load_pattern_map: Dict[str, LoadPattern] = {}
        
        # calculated matrix's
        self.global_force_vector: np.ndarray = None
        self.global_stiffness_matrix: np.ndarray = None

        # results
        self.displacements: np.ndarray = None
        self.reactions: np.ndarray = None

    def add_material(
        self,
        name: str,
        E: float,
        v: float,
        g: float = 0.0
        ):
        
        self.material_map[name] = GenericMaterial(name, E, v, g)
    
    def add_rectangular_section(
        self,
        name: str,
        material_name: str,
        b: float,
        h: float
        ):
        
        self.section_map[name] = RectangularSection(name, self.material_map[material_name], b, h)

    def add_node(
        self,
        id: int,
        vertex: VertexLike,
    ):
        vertex = Vertex(vertex)
        self.node_map[id] = Node(id, vertex)
    
    def add_element(
        self,
        id: int,
        type: str,
        node_i_id: int,
        node_j_id: int,
        section_name: str,
    ):
        
        type = to_enum(type, ElementType)
        self.element_map[id] = Element(
            id,
            type,
            self.node_map[node_i_id],
            self.node_map[node_j_id],
            self.section_map[section_name])

    def add_restraint(
        self,
        node_id: int,
        restraints: 'Restraints'
    ):
        self.node_map[node_id].add_restraints(restraints)
    
    def add_load_pattern(
        self,
        name: str,
        load_type: str,
        self_weight_multiplier: float,
        auto_load_pattern: bool = False,
        create_load_case: bool = False,
        state: str = "ACTIVE"
        ):
        
        load_type = to_enum(load_type, LoadPatternType)
        state = to_enum(state, State)
        
        self.load_pattern_map[name] = LoadPattern(
            name,
            load_type,
            self_weight_multiplier,
            auto_load_pattern,
            create_load_case,
            state
        )
    
    def add_point_load(
        self,
        node_id: int,
        load_pattern_name: str,
        CSys: str = "GLOBAL",
        fx: float = 0.0,
        fy: float = 0.0,
        mz: float = 0.0,
        angle: float = 0.0,
        replace: bool = False
        ):
        
        CSys = to_enum(CSys, CoordinateSystemType)
        
        self.load_pattern_map[load_pattern_name].add_point_load(
            node_id,
            PointLoad(fx=fx, fy=fy, mz=mz),
            CSys,
            angle,
            replace)
    
    def solve(self):
        
        lp = list(self.load_pattern_map.values())[0]

        lp.assign_loads_to_nodes(self)
        lp.assign_loads_to_elements(self)
        
        self.global_force_vector = calculate_load_vector(self)
        self.global_stiffness_matrix = assemble_global_stiffness_matrix(self)

        self.displacements = solve(self)