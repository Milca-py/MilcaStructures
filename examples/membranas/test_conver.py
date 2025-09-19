import milcapy as mp
from enum import Enum, auto
import matplotlib.pyplot as plt

class ElementType(Enum):
    CST = auto()
    MQ4 = auto()
    MQ6 = auto()
    MQ8Reduced = auto()
    MQ8Complete = auto()
    MQ6I = auto()

def test(element_type, L, h, b, E, v, Fy, nx=1, show=False):
    model = mp.SystemModel()
    model.add_material("concreto", E, v)
    model.add_shell_section("viga", "concreto", b)

    # === crear nodos ===
    dx = L / nx
    node_id = 1
    node_map = {}  # (i,j) -> id
    for i in range(nx+1):   # columnas
        x = i * dx
        for j, y in enumerate([0, h]):  # solo dos niveles: base y peralte
            model.add_node(node_id, x, y)
            node_map[(i,j)] = node_id
            # restringir apoyo en el extremo izquierdo (x=0)
            if i == 0:
                model.add_restraint(node_id, True, True, False)
            node_id += 1

    # === cargas: media mitad repartida en los nodos extremos derechos ===
    model.add_load_pattern("carga")
    right_nodes = [node_map[(nx,0)], node_map[(nx,1)]]
    for nid in right_nodes:
        model.add_point_load(nid, "carga", fy=0.5*Fy)

    # === elementos rectangulares ===
    elem_id = 1
    for i in range(nx):
        n1 = node_map[(i,1)]
        n2 = node_map[(i,0)]
        n3 = node_map[(i+1,0)]
        n4 = node_map[(i+1,1)]

        if element_type == ElementType.CST:
            model.add_cst(elem_id, n1, n2, n3, "viga")
            elem_id += 1
            model.add_cst(elem_id, n3, n4, n1, "viga")
        elif element_type == ElementType.MQ6:
            model.add_membrane_q6(elem_id, n1, n2, n3, n4, "viga")
        elif element_type == ElementType.MQ6I:
            model.add_membrane_q6i(elem_id, n1, n2, n3, n4, "viga")
        elif element_type == ElementType.MQ4:
            model.add_membrane_q4(elem_id, n1, n2, n3, n4, "viga")
        elif element_type == ElementType.MQ8Reduced:
            model.add_membrane_q8(elem_id, n1, n2, n3, n4, "viga", integration="REDUCED")
        elif element_type == ElementType.MQ8Complete:
            model.add_membrane_q8(elem_id, n2, n3, n4, n1, "viga", integration="COMPLETE")
        elem_id += 1

    # === resolver ===
    model.solve()

    # desplazamiento en nodo superior derecho
    top_right = node_map[(nx,1)]
    if show:
        if element_type == ElementType.CST:
            print(list((
                model.csts[elem_id-1].node1.id,
                model.csts[elem_id-1].node2.id,
                model.csts[elem_id-1].node3.id,
            )))
            print(list((
                model.csts[elem_id-2].node1.id,
                model.csts[elem_id-2].node2.id,
                model.csts[elem_id-2].node3.id,
            )))
        else:
            print(list((
                model.membrane_q2dof[elem_id-1].node1.id,
                model.membrane_q2dof[elem_id-1].node2.id,
                model.membrane_q2dof[elem_id-1].node3.id,
                model.membrane_q2dof[elem_id-1].node4.id,
            )))
        model.show()
    return float(model.results["carga"].get_node_displacements(top_right)[1])


# if __name__ == "__main__":
#     test(ElementType.CST, 1.5, 0.6, 0.25, 2.53456e6, 0.2, -6, 5, True)



if __name__ == "__main__":
    # Datos del problema
    L, h, b, E, v, Fy = 2.0, 0.6, 0.4, 2e6, 0.2, -20
    n_div = list(range(1, 15))

    # Soluciones exactas
    I = b*h**3/12
    G = E/(2*(1+v))
    A = b*h
    k = 1.2
    SOL_FLEXION = L**3/(3*E*I)*Fy #-0.00185185*2   # m (sin cortante)
    SOL_TIMOSHENKO = (L/(G*A*k) + L**3/(3*E*I))*Fy #-0.00193519*2  # m (con cortante)

    fig, ax = plt.subplots(figsize=(8, 5))

    for element_type in ElementType:
        sol = []
        for nx in n_div:
            val = test(element_type, L, h, b, E, v, Fy, nx)
            sol.append(val)
            # Si quieres ver cada resultado, descomenta:
            # print(f"Elemento: {element_type.name}, nx={nx}, sol={val:.6e}")
        ax.plot(n_div, sol, label=element_type.name, marker="o", markersize=2, linewidth=1)

    # Líneas horizontales de solución exacta
    ax.axhline(SOL_FLEXION, color="k", linestyle="--", linewidth=1, label="Exacta flexión (Euler-Bernoulli)")
    ax.axhline(SOL_TIMOSHENKO, color="r", linestyle=":", linewidth=1, label="Exacta flexión+cortante (Timoshenko)")

    # Mejoras visuales
    ax.set_title("Convergencia de deflexión en viga en voladizo")
    ax.set_xlabel("Número de divisiones (nx)")
    ax.set_ylabel("Deflexión en extremo libre [m]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
