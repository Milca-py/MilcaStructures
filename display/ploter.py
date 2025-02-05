import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.system import SystemMilcaModel

class Plotter:
    """Clase para graficar la estructura del sistema."""

    def __init__(self, system: "SystemMilcaModel") -> None:
        """
        Inicializa el objeto Plotter.

        Args:
            system (SystemMilcaModel): Instancia del sistema estructural a graficar.
        """
        self.system = system

    def plot_structure(self) -> None:
        """Grafica la estructura del sistema."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Dibujar elementos y nodos
        self._plot_elements(ax)
        self._plot_nodes(ax)
        self._plot_loads(ax)

        # Configuración de la gráfica
        ax.set_title("Estructura del Sistema", fontsize=14, fontweight='bold')
        # ax.set_xlabel("Coordenada X", fontsize=12)
        # ax.set_ylabel("Coordenada Y", fontsize=12)
        # ax.set_aspect("equal")
        # ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.axis('equal')
        plt.show()

    def _plot_elements(self, ax: plt.Axes) -> None:
        """Dibuja los elementos estructurales."""
        for element in self.system.element_map.values():
            node_i, node_j = element.node_i, element.node_j
            x_vals = [node_i.vertex.x, node_j.vertex.x]
            y_vals = [node_i.vertex.y, node_j.vertex.y]
            ax.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.8, label="Elemento" if element.id == 1 else "")
            ax.text((x_vals[0] + x_vals[1]) / 2, 
                    (y_vals[0] + y_vals[1]) / 2, 
                    f"{element.id}", 
                    fontsize=10, 
                    color="blue", 
                    # ha="center", 
                    # va="center"
                    )

    def _plot_nodes(self, ax: plt.Axes) -> None:
        """Dibuja los nodos estructurales."""
        for node in self.system.node_map.values():
            # ax.plot(node.vertex.x, node.vertex.y, 'ko', markersize=5, label="Nodo" if node.id == 1 else "")
            ax.text(node.vertex.x, node.vertex.y, f"{node.id}", 
                    fontsize=10, 
                    color="black", 
                    ha="right", 
                    va="bottom")

    def _plot_supports(self, ax: plt.Axes) -> None:
        """Dibuja los apoyos."""
        pass
    
    def _plot_loads(self, ax: plt.Axes) -> None:
        """Dibuja las cargas."""
        from display.graphics_point import carga_distribuida
        #     carga_distribuida(ax, phi=3.1416/5, direccion="X",
        #                         longitud=10, w=-100, k=0.01, color="blue")
        for load_pattern in self.system.load_pattern_map.values():
            id_el, dist_load = load_pattern.distributed_loads_map.items()
            for ide, dload in zip(id_el, dist_load):
                carga_distribuida(
                    xi=self.system.element_map[2].node_i.vertex.x,
                    yi=self.system.element_map[2].node_i.vertex.y,
                    ax=ax,
                    phi=3.1416/4,
                    direccion="Y",
                    longitud=200,
                    w=-1000,
                    k=0.01,
                    color="blue",
                )











