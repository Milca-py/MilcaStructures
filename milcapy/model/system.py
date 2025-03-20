from typing import TYPE_CHECKING, Dict, Optional, Union, Tuple
import numpy as np

from milcapy.material.material import Material, GenericMaterial
from milcapy.section.section import Section, RectangularSection
from milcapy.elements.node import Node
from milcapy.elements.element import Element
from milcapy.utils.vertex import Vertex

from milcapy.loads import LoadPattern, PointLoad

from milcapy.utils.custom_types import (
    LoadPatternType,
    CoordinateSystemType,
    State,
    ElementType,
    DirectionType,
    LoadType,
    to_enum,
)

if TYPE_CHECKING:
    from milcapy.utils.custom_types import Restraints, VertexLike

from milcapy.plotter.plotter import Plotter, PlotterOptions
from milcapy.plotter.plotter_values import PlotterValuesFactory, PlotterValues
from milcapy.results.results import Results, ResultOptions
from milcapy.analysis.analysis import AnalysisOptions, AnalysisManager, LinearStaticOptions
from milcapy.postprocess.post_processing import PostProcessingOptions
from milcapy.solver.solver import DirectStiffnessSolverrOptions
from milcapy.plotter.options import GraphicOptionCalculator

class SystemMilcaModel:
    """
    Clase que representa el modelo estructural completo para análisis mediante el método de rigidez.
    
    Esta clase permite definir materiales, secciones, nodos, elementos y cargas para realizar
    un análisis estructural completo utilizando el método de rigidez directa.
    """

    def __init__(self) -> None:
        """Inicializa un nuevo modelo estructural vacío."""
        # Propiedades de los elementos [UNIQUE]
        self.material_map: Dict[str, Material] = {}
        self.section_map: Dict[str, Section] = {}
        
        # Elementos del modelo [UNIQUE / MOD]
        self.node_map: Dict[int, Node] = {}
        self.element_map: Dict[int, Element] = {}
        
        # Colecciones de frontera [UNIQUE]
        self.load_pattern_map: Dict[str, LoadPattern] = {}
        # self.restraint_map: Dict[int, Restraints] = {}
        
        # Matrices calculadas [MOD]
        self.global_load_vector: Optional[np.ndarray] = None
        self.global_stiffness_matrix: Optional[np.ndarray] = None

        # Resultados de analisis matricial [MOD]
        self.displacements: Optional[np.ndarray] = None
        self.reactions: Optional[np.ndarray] = None
        
        # coleccion de resultados incluyendo postprocesamiento [ADD]
        self.loadpattern_results: Dict[str, Results] = {}

        # Análisis [UNIQUE]
        self.analysis: AnalysisManager = AnalysisManager(self)

        # Visualización [UNIQUE]
        self.plotter: Optional[Plotter] = None
        self.plotter_values_factory: Optional[PlotterValuesFactory] = None
        self.plotter_options_calculator: Optional[GraphicOptionCalculator] = None

        # Opciones del modelo [UNIQUE]
        self.analysis_options: "AnalysisOptions" = LinearStaticOptions() # IMPLEMETAR PARA CADA TIPO DE ANALISIS
        self.results_options: "ResultOptions" = ResultOptions()
        self.plotter_options: "PlotterOptions" = None
        self.postprocessing_options: "PostProcessingOptions" = PostProcessingOptions(factor=1, n=5)
        self.solver_options: "DirectStiffnessSolverrOptions" = DirectStiffnessSolverrOptions()

    def inicialize_plotter(self) -> None:
        self.plotter_options_calculator = GraphicOptionCalculator(self)
        self.plotter_options = PlotterOptions(self.plotter_options_calculator)
        self.plotter_values_factory = PlotterValuesFactory(self)
        self.plotter = Plotter(self)

    def add_material(
        self,
        name: str,
        modulus_elasticity: float,
        poisson_ratio: float,
        specific_weight: float = 0.0
    ) -> Material:
        """
        Agrega un material al modelo.

        Args:
            name (str): Nombre del material.
            modulus_elasticity (float): Módulo de elasticidad (E).
            poisson_ratio (float): Coeficiente de Poisson.
            specific_weight (float, opcional): Peso específico o densidad. Default es 0.0.
            
        Returns:
            Material: El material creado.
            
        Raises:
            ValueError: Si ya existe un material con el mismo nombre.
        """
        if name in self.material_map:
            raise ValueError(f"Ya existe un material con el nombre '{name}'")
            
        material = GenericMaterial(name, modulus_elasticity, poisson_ratio, specific_weight)
        self.material_map[name] = material
        return material
    
    def add_rectangular_section(
        self,
        name: str,
        material_name: str,
        base: float,
        height: float
    ) -> Section:
        """
        Agrega una sección rectangular al modelo.

        Args:
            name (str): Nombre de la sección.
            material_name (str): Nombre del material asociado (ya agregado).
            base (float): Base de la sección.
            height (float): Altura de la sección.
            
        Returns:
            Section: La sección creada.
            
        Raises:
            ValueError: Si ya existe una sección con el mismo nombre o si no existe el material.
        """
        if name in self.section_map:
            raise ValueError(f"Ya existe una sección con el nombre '{name}'")
            
        if material_name not in self.material_map:
            raise ValueError(f"No existe un material con el nombre '{material_name}'")
            
        section = RectangularSection(name, self.material_map[material_name], base, height)
        self.section_map[name] = section
        return section

    def add_node(
        self,
        id: int,
        vertex: Union["VertexLike", Tuple[float, float]]
    ) -> Node:
        """
        Agrega un nodo al modelo.

        Args:
            id (int): Identificador del nodo.
            vertex (VertexLike o Tuple[float, float]): Coordenadas del nodo.
            
        Returns:
            Node: El nodo creado.
            
        Raises:
            ValueError: Si ya existe un nodo con el mismo ID.
        """
        if id in self.node_map:
            raise ValueError(f"Ya existe un nodo con el ID {id}")
            
        node = Node(id, Vertex(vertex))
        self.node_map[id] = node
        return node
    
    def add_element(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        section_name: str,
        element_type: Union[str, ElementType] = ElementType.FRAME
    ) -> Element:
        """
        Agrega un elemento estructural al modelo.

        Args:
            id (int): Identificador del elemento.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.
            element_type (str o ElementType, opcional): Tipo de elemento. Por defecto es FRAME.
            
        Returns:
            Element: El elemento creado.
            
        Raises:
            ValueError: Si ya existe un elemento con el mismo ID, o si no existen los nodos o la sección.
        """
        if id in self.element_map:
            raise ValueError(f"Ya existe un elemento con el ID {id}")
            
        if node_i_id not in self.node_map:
            raise ValueError(f"No existe un nodo con el ID {node_i_id}")
            
        if node_j_id not in self.node_map:
            raise ValueError(f"No existe un nodo con el ID {node_j_id}")
            
        if section_name not in self.section_map:
            raise ValueError(f"No existe una sección con el nombre '{section_name}'")
        
        # Convertir a enum si es string
        if isinstance(element_type, str):
            element_type = to_enum(element_type, ElementType)
            
        element = Element(
            id=id,
            type=element_type,
            node_i=self.node_map[node_i_id],
            node_j=self.node_map[node_j_id],
            section=self.section_map[section_name]
        )
        self.element_map[id] = element
        return element

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
            
        Raises:
            ValueError: Si no existe el nodo.
        """
        if node_id not in self.node_map:
            raise ValueError(f"No existe un nodo con el ID {node_id}")
            
        self.node_map[node_id].add_restraints(restraints)
    
    def add_load_pattern(
        self,
        name: str,
        pattern_type: Union[str, LoadPatternType] = LoadPatternType.DEAD,
        self_weight_multiplier: float = 0.0,
        auto_load_pattern: bool = False,
        create_load_case: bool = False,
        state: Union[str, State] = State.ACTIVE
    ) -> LoadPattern:
        """
        Agrega un patrón de carga al modelo.

        Args:
            name (str): Nombre del patrón de carga.
            pattern_type (str o LoadPatternType, opcional): Tipo de carga. Default es DEAD.
            self_weight_multiplier (float, opcional): Multiplicador del peso propio. Default es 0.0.
            auto_load_pattern (bool, opcional): Si se genera automáticamente. Default es False.
            create_load_case (bool, opcional): Si se crea un caso de carga asociado. Default es False.
            state (str o State, opcional): Estado del patrón. Default es ACTIVE.
            
        Returns:
            LoadPattern: El patrón de carga creado.
            
        Raises:
            ValueError: Si ya existe un patrón con el mismo nombre.
        """
        if name in self.load_pattern_map:
            raise ValueError(f"Ya existe un patrón de carga con el nombre '{name}'")
        
        # Convertir a enum si son strings
        if isinstance(pattern_type, str):
            pattern_type = to_enum(pattern_type, LoadPatternType)
        
        if isinstance(state, str):
            state = to_enum(state, State)
            
        load_pattern = LoadPattern(
            name=name,
            pattern_type=pattern_type,
            self_weight_multiplier=self_weight_multiplier,
            auto_load_pattern=auto_load_pattern,
            create_load_case=create_load_case,
            state=state,
            system=self,
        )
        self.load_pattern_map[name] = load_pattern
        return load_pattern
    
    def add_point_load(
        self,
        node_id: int,
        load_pattern_name: str,
        CSys: Union[str, CoordinateSystemType] = "GLOBAL",
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
            CSys (str o CoordinateSystemType, opcional): Sistema de coordenadas. Default es "GLOBAL".
            fx (float, opcional): Fuerza en X. Default es 0.0.
            fy (float, opcional): Fuerza en Y. Default es 0.0.
            mz (float, opcional): Momento en Z. Default es 0.0.
            angle_rot (float, opcional): Ángulo de rotación en radianes. Default es None.
            replace (bool, opcional): Si se reemplaza la carga existente. Default es False.
            
        Raises:
            ValueError: Si no existe el nodo o el patrón de carga.
        """
        if node_id not in self.node_map:
            raise ValueError(f"No existe un nodo con el ID {node_id}")
            
        if load_pattern_name not in self.load_pattern_map:
            raise ValueError(f"No existe un patrón de carga con el nombre '{load_pattern_name}'")
        
        # Convertir a enum si es string
        if isinstance(CSys, str):
            csys_enum = to_enum(CSys, CoordinateSystemType)
        else:
            csys_enum = CSys
            
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
        CSys: Union[str, CoordinateSystemType] = "LOCAL",
        load_start: float = 0.0,
        load_end: float = 0.0,
        replace: bool = False,
        direction: Union[str, DirectionType] = "LOCAL_2",
        load_type: Union[str, LoadType] = "FORCE"
    ) -> None:
        """
        Asigna una carga distribuida a un elemento dentro de un patrón de carga.

        Args:
            element_id (int): Identificador del elemento.
            load_pattern_name (str): Nombre del patrón de carga.
            CSys (str o CoordinateSystemType, opcional): Sistema de coordenadas. Default es "LOCAL".
            load_start (float, opcional): Magnitud de la carga en el inicio. Default es 0.0.
            load_end (float, opcional): Magnitud de la carga en el final. Default es 0.0.
            replace (bool, opcional): Si se reemplaza la carga existente. Default es False.
            direction (str o DirectionType, opcional): Dirección de la carga. Default es "LOCAL_2".
            load_type (str o LoadType, opcional): Tipo de carga. Default es "FORCE".
            
        Raises:
            ValueError: Si no existe el elemento o el patrón de carga.
        """
        if element_id not in self.element_map:
            raise ValueError(f"No existe un elemento con el ID {element_id}")
            
        if load_pattern_name not in self.load_pattern_map:
            raise ValueError(f"No existe un patrón de carga con el nombre '{load_pattern_name}'")
        
        # Convertir a enum si son strings
        if isinstance(CSys, str):
            csys_enum = to_enum(CSys, CoordinateSystemType)
        else:
            csys_enum = CSys
            
        if isinstance(direction, str):
            direction_enum = to_enum(direction, DirectionType)
        else:
            direction_enum = direction
            
        if isinstance(load_type, str):
            load_type_enum = to_enum(load_type, LoadType)
        else:
            load_type_enum = load_type
        
        self.load_pattern_map[load_pattern_name].add_distributed_load(
            element_id=element_id,
            load_start=load_start,
            load_end=load_end,
            load_type=load_type_enum,
            csys=csys_enum,
            replace=replace,
            direction=direction_enum
        )
    
    def solve(self) -> Dict[str, Results]:
        """
        Resuelve el sistema estructural aplicando el método de rigidez:
        - Asigna las cargas a nodos y elementos.
        - Calcula el vector de fuerzas y la matriz de rigidez global.
        - Resuelve el sistema de ecuaciones para obtener los desplazamientos y reacciones.
        
        Returns:
            Dict[str, Results]: Objeto con los resultados del análisis.
            
        Raises:
            ValueError: Si no hay patrones de carga definidos.
        """
        # Verificar que exista al menos un patrón de carga
        if not self.load_pattern_map:
            raise ValueError("No hay patrones de carga definidos. Agregue al menos uno para resolver el sistema.")
            
        # Inicializar análisis
        self.analysis.run()
        
        return self.loadpattern_results

    def show_structure(
        self,
        axes_i: int = 0,
        labels_nodes: bool = False,
        labels_elements: bool = False,
        color_nodes: str = "red",
        color_elements: str = "blue",
        color_labels_node: str = "red",
        color_labels_element: str = "red",
        labels_point_loads: bool = True,
        labels_distributed_loads: bool = True,
        color_point_loads: str = "green",
        color_distributed_loads: str = "purple",
        show: bool = True
    ) -> None:
        """
        Muestra la estructura del modelo con sus nodos, elementos y cargas.
        
        Args:
            axes_i (int, opcional): Índice del eje para graficar. Default es 0.
            labels_nodes (bool, opcional): Mostrar etiquetas de nodos. Default es False.
            labels_elements (bool, opcional): Mostrar etiquetas de elementos. Default es False.
            color_nodes (str, opcional): Color de los nodos. Default es "red".
            color_elements (str, opcional): Color de los elementos. Default es "blue".
            color_labels_node (str, opcional): Color de las etiquetas de nodos. Default es "red".
            color_labels_element (str, opcional): Color de las etiquetas de elementos. Default es "red".
            labels_point_loads (bool, opcional): Mostrar etiquetas de cargas puntuales. Default es True.
            labels_distributed_loads (bool, opcional): Mostrar etiquetas de cargas distribuidas. Default es True.
            color_point_loads (str, opcional): Color de las cargas puntuales. Default es "green".
            color_distributed_loads (str, opcional): Color de las cargas distribuidas. Default es "purple".
            show (bool, opcional): Mostrar el gráfico inmediatamente. Default es True.
        """
        if self.plotter is None:
            self.plotter = Plotter(self)  # Inicializar solo cuando se use
            
        self.plotter.plot_structure(
            axes_i=axes_i,
            labels_nodes=labels_nodes,
            labels_elements=labels_elements,
            color_nodes=color_nodes,
            color_elements=color_elements,
            color_labels_node=color_labels_node,
            color_labels_element=color_labels_element,
            labels_point_loads=labels_point_loads,
            labels_distributed_loads=labels_distributed_loads,
            color_point_loads=color_point_loads,
            color_distributed_loads=color_distributed_loads,
            show=show
        )
    
    def reset(self) -> None:
        """
        Reinicia el modelo a su estado base sin eliminar los elementos estructurales.
        Mantiene los materiales, secciones, nodos y elementos, pero limpia las cargas,
        condiciones de frontera y resultados previos.
        """
        # Reiniciar las cargas en los nodos
        for node in self.node_map.values():
            node.reset()
        
        # Reiniciar las cargas distribuidas y resultados en los elementos
        for element in self.element_map.values():
            element.reset()
        
        # Reiniciar matrices y resultados globales
        self.global_load_vector = None
        self.global_stiffness_matrix = None
        self.displacements = None
        self.reactions = None
