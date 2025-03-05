from typing import TYPE_CHECKING, Dict, Optional
import numpy as np

from core.material import Material, GenericMaterial
from core.section import Section, RectangularSection
from core.node import Node
from core.element import Element
from utils.vertex import Vertex

from loads import LoadPattern, PointLoad, DistributedLoad

from utils.custom_types import (
    LoadPatternType,
    CoordinateSystemType,
    State,
    ElementType,
    DirectionType,
    LoadType,
    to_enum,
)

if TYPE_CHECKING:
    from utils.custom_types import Restraints, VertexLike

from components.system_components import (
    assemble_global_load_vector,
    assemble_global_stiffness_matrix,
    solve,
)

from display.plotter import Plotter, PlotterValues

from core.results import Results

from core.analysis import Analysis

class SystemMilcaModel:
    """
    Clase que representa el modelo estructural completo para análisis mediante el método de rigidez.
    """

    def __init__(self) -> None:
        # Propiedades de los elementos
        self.material_map: Dict[str, Material] = {}
        self.section_map: Dict[str, Section] = {}
        
        # Elementos del modelo
        self.node_map: Dict[int, Node] = {}
        self.element_map: Dict[int, Element] = {}
        
        # Colecciones de cargas
        self.load_pattern_map: Dict[str, LoadPattern] = {}
        
        # Matrices calculadas
        self.global_force_vector: Optional[np.ndarray] = None
        self.global_stiffness_matrix: Optional[np.ndarray] = None

        # Análisis
        self.analysis: Optional[Analysis] = Analysis(self)

        # Resultados
        self.displacements: Optional[np.ndarray] = None
        self.reactions: Optional[np.ndarray] = None
        
        self.results: Optional[Results] = None
        
        # ploter
        self.plotter: Optional[Plotter] = None
        
        # plotter values
        self.plotter_values: Optional[PlotterValues] = None 

        
    def add_material(
        self,
        name: str,
        modulus_elasticity: float,
        poisson_ratio: float,
        specific_weight: float = 0.0
    ) -> None:
        """
        Agrega un material al modelo.

        Args:
            name (str): Nombre del material.
            modulus_elasticity (float): Módulo de elasticidad (E).
            poisson_ratio (float): Coeficiente de Poisson.
            specific_weight (float, opcional): Peso específico o densidad. Default es 0.0.
        """
        self.material_map[name] = GenericMaterial(name, modulus_elasticity, poisson_ratio, specific_weight)
    
    def add_rectangular_section(
        self,
        name: str,
        material_name: str,
        base: float,
        height: float
    ) -> None:
        """
        Agrega una sección rectangular al modelo.

        Args:
            name (str): Nombre de la sección.
            material_name (str): Nombre del material asociado (ya agregado).
            base (float): Base de la sección.
            height (float): Altura de la sección.
        """
        self.section_map[name] = RectangularSection(name, self.material_map[material_name], base, height)

    def add_node(
        self,
        id: int,
        vertex: "VertexLike"
    ) -> None:
        """
        Agrega un nodo al modelo.

        Args:
            id (int): Identificador del nodo.
            vertex (VertexLike): Coordenadas del nodo (convertibles a Vertex).
        """
        self.node_map[id] = Node(id, Vertex(vertex))
    
    def add_element(
        self,
        id: int,
        # type: str,
        node_i_id: int,
        node_j_id: int,
        section_name: str
    ) -> None:
        """
        Agrega un elemento estructural al modelo.

        Args:
            id (int): Identificador del elemento.
            type (str): Tipo de elemento (se convertirá a ElementType).
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.
        """
        # element_type = to_enum(type, ElementType)
        self.element_map[id] = Element(
            id=id,
            type=ElementType.FRAME,
            node_i=self.node_map[node_i_id],
            node_j=self.node_map[node_j_id],
            section=self.section_map[section_name]
        )

    def add_restraint(
        self,
        node_id: int,
        restraints: "Restraints"
    ) -> None:
        """
        Asigna restricciones (condiciones de frontera) a un nodo.

        Args:
            node_id (int): Identificador del nodo.
            restraints (Restraints): Tupla booleana con las restricciones.
        """
        self.node_map[node_id].add_restraints(restraints)
    
    def add_load_pattern(
        self,
        name: str,
        # pattern_type: str = "DEAD",
        # self_weight_multiplier: float = 0.0,
        # auto_load_pattern: bool = False,
        # create_load_case: bool = False,
        # state: str = "ACTIVE"
    ) -> None:
        """
        Agrega un patrón de carga al modelo.

        Args:
            name (str): Nombre del patrón de carga.
            load_type (str): Tipo de carga (se convertirá a LoadPatternType).
            self_weight_multiplier (float): Multiplicador del peso propio.
            auto_load_pattern (bool, opcional): Si se genera automáticamente. Default es False.
            create_load_case (bool, opcional): Si se crea un caso de carga asociado. Default es False.
            state (str, opcional): Estado del patrón (se convertirá a State). Default es "ACTIVE".
        """
        # lpattern_type = to_enum(pattern_type, LoadPatternType)
        # lp_state = to_enum(state, State)
        self.load_pattern_map[name] = LoadPattern(
            name,
            # lpattern_type,
            # self_weight_multiplier,
            # auto_load_pattern,
            # create_load_case,
            # lp_state,
            system=self
        )
    
    def add_point_load(
        self,
        node_id: int,
        load_pattern_name: str,
        CSys: str = "GLOBAL",
        fx: float = 0.0,
        fy: float = 0.0,
        mz: float = 0.0,
        angle_rot: Optional[float] = None,
        replace: bool = False
    ) -> None:
        """
        Asigna una carga puntual a un nodo dentro de un patrón de carga.

        Args:
            node_id (int): Identificador del nodo.
            load_pattern_name (str): Nombre del patrón de carga.
            CSys (str, opcional): Sistema de coordenadas ("GLOBAL" o "LOCAL"). Default es "GLOBAL".
            fx (float, opcional): Fuerza en X.
            fy (float, opcional): Fuerza en Y.
            mz (float, opcional): Momento en Z.
            angle (float, opcional): Ángulo de rotación en radianes.
            replace (bool, opcional): Si se reemplaza la carga existente. Default es False.
        """
        csys_enum = to_enum(CSys, CoordinateSystemType)
        self.load_pattern_map[load_pattern_name].add_point_load(
            node_id=node_id,
            forces=PointLoad(fx=fx, fy=fy, mz=mz),
            csys=csys_enum,
            angle_rot=angle_rot,
            replace=replace
        )
    
    def add_distributed_load(
        self,
        element_id: int,
        load_pattern_name: str,
        CSys: str = "GLOBAL",
        load_start: float = 0.0,
        load_end: float = 0.0,
        replace: bool = False,
        direction: str = "LOCAL_2",
        load_type: str = "FORCE"
        
    ) -> None:
        """
        Asigna una carga distribuida a un elemento dentro de un patrón de carga.

        Args:
            element_id (int): Identificador del elemento.
            load_pattern_name (str): Nombre del patrón de carga.
            CSys (str, opcional): Sistema de coordenadas ("GLOBAL" o "LOCAL"). Default es "GLOBAL".
            q_i (float, opcional): Magnitud de la carga en el nodo inicial.
            q_j (float, opcional): Magnitud de la carga en el nodo final.
            replace (bool, opcional): Si se reemplaza la carga existente. Default es False.
            
        """
        csys_enum = to_enum(CSys, CoordinateSystemType)
        direction_enum = to_enum(direction, DirectionType)
        load_type_enum = to_enum(load_type, LoadType)
        
        self.load_pattern_map[load_pattern_name].add_distributed_load(
            element_id=element_id,
            load_start=load_start,
            load_end=load_end,
            load_type=load_type_enum,
            csys=csys_enum,
            replace=replace,
            direction=direction_enum
        )
    
    def solve(self) -> None:
        """
        Resuelve el sistema estructural aplicando el método de rigidez:
        - Asigna las cargas a nodos y elementos.
        - Calcula el vector de fuerzas y la matriz de rigidez global.
        - Resuelve el sistema de ecuaciones para obtener los desplazamientos y reacciones.
        """
        # Se considera el primer patrón de carga agregado
        lp = list(self.load_pattern_map.values())[0]

        # Asignar las cargas a los nodos y elementos almacenados en el patrón de carga
        lp.assign_loads_to_nodes(self)
        lp.assign_loads_to_elements(self)
        
        # Calcular el vector de fuerzas global y la matriz de rigidez global
        self.global_force_vector = assemble_global_load_vector(self)
        self.global_stiffness_matrix = assemble_global_stiffness_matrix(self)

        # Se puede aplicar el procesamiento de condiciones de frontera si se requiere:
        # process_conditions(self)
        
        # Resolver el sistema de ecuaciones
        self.displacements, self.reactions = solve(self)
        
        # acutalizar estado de análisis
        self.analysis.options.status = True
        self.results = Results(self)
        self.plotter = Plotter(self)
        self.plotter_values = PlotterValues(self)

    def show_structure(self, show: bool = True) -> None:
        """
        Muestra la estructura del modelo.
        """
        if self.plotter is None:
            self.plotter = Plotter(self)  # Inicializar solo cuando se use
        self.plotter.plot_structure(show=show)







