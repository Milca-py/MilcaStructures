from typing import TYPE_CHECKING, Dict, Optional, Union, Sequence

from milcapy.material.material import Material, GenericMaterial
from milcapy.section.section import Section, RectangularSection
from milcapy.core.node import Node
from milcapy.core.results import Results
from milcapy.elements.member import Member
from milcapy.plotter.plotter import Plotter, PlotterOptions
from milcapy.analysis.manager import AnalysisManager
from milcapy.postprocess.post_processing import PostProcessingOptions
from milcapy.loads import LoadPattern, PointLoad
from milcapy.utils.geometry import Vertex
from milcapy.utils.types import (
    LoadPatternType,
    CoordinateSystemType,
    StateType,
    MemberType,
    BeamTheoriesType,
    DirectionType,
    ShearCoefficientMethodType,
    LoadType,
    to_enum,
)

if TYPE_CHECKING:
    from milcapy.utils.types import Restraints

class SystemMilcaModel:
    """Clase que representa el modelo estructural, colecciona materiales, secciones, nodos, miembros, load patterns, resultados"""

    def __init__(self) -> None:
        """Inicializa un nuevo modelo estructural vacío."""
        # Propiedades de los elementos [UNIQUE]
        self.materials: Dict[str, Material] = {}
        self.sections: Dict[str, Section] = {}

        # Elementos del modelo [UNIQUE / ADD]
        self.nodes: Dict[int, Node] = {}
        self.members: Dict[int, Member] = {}

        # patrones de carga con las asiganciones de carga en los miembros y nodos
        self.load_patterns: Dict[str, LoadPattern] = {} # {pattern_name: load pattern}

        # coleccion de resultados incluyendo postprocesamiento [ADD]
        self.results: Dict[str, Results] = {} # {pattern_name: results}

        # Análisis [UNIQUE]
        self.analysis: Optional[AnalysisManager] = None

        # Visualización [UNIQUE]
        self.plotter: Optional[Plotter] = None

        # Opciones del modelo [UNIQUE]
        self.plotter_options: "PlotterOptions" = PlotterOptions(self)
        self.postprocessing_options: "PostProcessingOptions" = PostProcessingOptions(factor=1, n=17)


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
        if name in self.materials:
            raise ValueError(f"Ya existe un material con el nombre '{name}'")

        material = GenericMaterial(name, modulus_elasticity, poisson_ratio, specific_weight)
        self.materials[name] = material
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
        if name in self.sections:
            raise ValueError(f"Ya existe una sección con el nombre '{name}'")

        if material_name not in self.materials:
            raise ValueError(f"No existe un material con el nombre '{material_name}'")

        section = RectangularSection(
            name=name,
            material=self.materials[material_name],
            base=base,
            height=height,
            shear_method=ShearCoefficientMethodType.TIMOSHENKO
        )
        self.sections[name] = section
        return section

    def add_node(
        self,
        id: int,
        x: float,
        y: float
    ) -> Node:
        """
        Agrega un nodo al modelo.

        Args:
            id (int): Identificador del nodo.
            x (float): Coordenada x del nodo.
            y (float): Coordenada y del nodo.

        Returns:
            Node: El nodo creado.

        Raises:
            ValueError: Si ya existe un nodo con el mismo ID.
        """
        if id in self.nodes:
            raise ValueError(f"Ya existe un nodo con el ID {id}")

        node = Node(id, Vertex(x, y))
        self.nodes[id] = node
        return node

    def add_member(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        section_name: str,
        member_type: Union[str, MemberType] = MemberType.FRAME
    ) -> Member:
        """
        Agrega un miembro estructural al modelo.

        Args:
            id (int): Identificador del miembro.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.
            member_type (str, opcional): Tipo de miembro. Por defecto es FRAME.

        Returns:
            Member: El miembro creado.

        Raises:
            ValueError: Si ya existe un miembro con el mismo ID, o si no existen los nodos o la sección.
        """
        if id in self.members:
            raise ValueError(f"Ya existe un miembro con el ID {id}")

        if node_i_id not in self.nodes:
            raise ValueError(f"No existe un nodo con el ID {node_i_id}")

        if node_j_id not in self.nodes:
            raise ValueError(f"No existe un nodo con el ID {node_j_id}")

        if section_name not in self.sections:
            raise ValueError(f"No existe una sección con el nombre '{section_name}'")

        # Convertir a enum si es string
        if isinstance(member_type, str):
            member_type = to_enum(member_type, MemberType)

        element = Member(
            id=id,
            node_i=self.nodes[node_i_id],
            node_j=self.nodes[node_j_id],
            section=self.sections[section_name],
            member_type=member_type,
            beam_theory=BeamTheoriesType.TIMOSHENKO,
        )
        self.members[id] = element
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
        if node_id not in self.nodes:
            raise ValueError(f"No existe un nodo con el ID {node_id}")

        self.nodes[node_id].set_restraints(restraints)

    def add_load_pattern(
        self,
        name: str,
        pattern_type: Union[str, LoadPatternType] = LoadPatternType.DEAD,
        self_weight_multiplier: float = 0.0,
        state: Union[str, StateType] = StateType.ACTIVE
    ) -> LoadPattern:
        """
        Agrega un patrón de carga al modelo.

        Args:
            name (str): Nombre del patrón de carga.
            pattern_type (str o LoadPatternType, opcional): Tipo de carga. Default es DEAD.
            self_weight_multiplier (float, opcional): Multiplicador del peso propio. Default es 0.0.
            state (str o State, opcional): Estado del patrón. Default es ACTIVE.

        Returns:
            LoadPattern: El patrón de carga creado.

        Raises:
            ValueError: Si ya existe un patrón con el mismo nombre.
        """
        if name in self.load_patterns:
            raise ValueError(f"Ya existe un patrón de carga con el nombre '{name}'")

        # Convertir a enum si son strings
        if isinstance(pattern_type, str):
            pattern_type = to_enum(pattern_type, LoadPatternType)

        if isinstance(state, str):
            state = to_enum(state, StateType)

        load_pattern = LoadPattern(
            name=name,
            pattern_type=pattern_type,
            self_weight_multiplier=self_weight_multiplier,
            state=state,
            system=self,
        )
        self.load_patterns[name] = load_pattern
        return load_pattern

    def add_point_load(
        self,
        node_id: int,
        load_pattern_name: str,
        fx: float = 0.0,
        fy: float = 0.0,
        mz: float = 0.0,
        CSys: Union[str, CoordinateSystemType] = "GLOBAL",
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
        if node_id not in self.nodes:
            raise ValueError(f"No existe un nodo con el ID {node_id}")

        if load_pattern_name not in self.load_patterns:
            raise ValueError(f"No existe un patrón de carga con el nombre '{load_pattern_name}'")

        # Convertir a enum si es string
        if isinstance(CSys, str):
            csys_enum = to_enum(CSys, CoordinateSystemType)
        else:
            csys_enum = CSys

        self.load_patterns[load_pattern_name].add_point_load(
            node_id=node_id,
            forces=PointLoad(fx=fx, fy=fy, mz=mz),
            csys=csys_enum,
            angle_rot=angle_rot,
            replace=replace
        )

    def add_distributed_load(
        self,
        member_id: int,
        load_pattern_name: str,
        load_start: float = 0.0,
        load_end: float = 0.0,
        CSys: Union[str, CoordinateSystemType] = "LOCAL",
        direction: Union[str, DirectionType] = "LOCAL_2",
        load_type: Union[str, LoadType] = "FORCE",
        replace: bool = False,
    ) -> None:
        """
        Asigna una carga distribuida a un miembro dentro de un patrón de carga.

        Args:
            member_id (int): Identificador del miembro.
            load_pattern_name (str): Nombre del patrón de carga.
            load_start (float, opcional): Magnitud de la carga en el inicio. Default es 0.0.
            load_end (float, opcional): Magnitud de la carga en el final. Default es 0.0.
            CSys (str o CoordinateSystemType, opcional): Sistema de coordenadas. Default es "LOCAL".
            direction (str o DirectionType, opcional): Dirección de la carga. Default es "LOCAL_2".
            load_type (str o LoadType, opcional): Tipo de carga. Default es "FORCE".
            replace (bool, opcional): Si se reemplaza la carga existente. Default es False.

        Raises:
            ValueError: Si no existe el miembro o el patrón de carga.
        """
        if member_id not in self.members:
            raise ValueError(f"No existe un miembro con el ID {member_id}")

        if load_pattern_name not in self.load_patterns:
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

        self.load_patterns[load_pattern_name].add_distributed_load(
            member_id=member_id,
            load_start=load_start,
            load_end=load_end,
            csys=csys_enum,
            direction=direction_enum,
            load_type=load_type_enum,
            replace=replace,
        )

    def add_end_length_offset(self, member_id: int, la: float, lb: float, qla: bool = True, qlb: bool = True, fla: float = 1, flb: float = 1) -> None:
        if member_id not in self.members:
            raise ValueError(f"No existe un miembro con el ID {member_id}")
        self.members[member_id].la = la*fla
        self.members[member_id].lb = lb*flb
        self.members[member_id].qla = qla
        self.members[member_id].qlb = qlb




    def solve(self, load_pattern_name: list[str] | None = None) -> Dict[str, Results]:
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
        if not self.load_patterns:
            raise ValueError("No hay patrones de carga definidos. Agregue al menos uno para resolver el sistema.")

        # Inicializar análisis
        self.analysis = AnalysisManager(self)
        self.analysis.analyze(load_pattern_name)

        return self.results

    def show(self):
        from milcapy.plotter.UIdisplay import main_window
        self.plotter = Plotter(self)
        self.plotter.initialize_plot()
        main_window(self)