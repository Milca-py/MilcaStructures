from milcapy.core.node import Node
from milcapy.elements.member import Member
from milcapy.utils.geometry import Vertex
from milcapy.loads.load import PointLoad
from milcapy.section.section import Section
from milcapy.material.material import Material
from milcapy.utils.types import ElementType

def test_node():
    node = Node(1, Vertex(1, 2))
    node.set_load(PointLoad(1, 2, 3))
    print(node)


def test_member():
    node_i = Node(1, Vertex(1, 2))
    node_j = Node(2, Vertex(3, 4))
    material = Material("mat", 2e6, 0.15, 2400)
    section = Section("sec1", material)
    member = Member(1, node_i, node_j, section, ElementType.FRAME)
    print(member.length)


if __name__ == "__main__":
    test_member()
