from milcapy import *

# patch test de Pian y Sumihara

def main():
    E = 1500
    v = 0.25
    t = 1
    patch_test = SystemModel()
    patch_test.add_material('material', E, v)
    patch_test.add_shell_section('section', 'material', t)
    patch_test.add_node(1, 0, 0)
    patch_test.add_node(2, 1, 0)
    patch_test.add_node(3, 2, 0)
    patch_test.add_node(4, 4, 0)
    patch_test.add_node(5, 7, 0)
    patch_test.add_node(6, 10, 0)
    patch_test.add_node(7, 0, 2)
    patch_test.add_node(8, 2, 2)
    patch_test.add_node(9, 4, 2)
    patch_test.add_node(10, 5, 2)
    patch_test.add_node(11, 6, 2)
    patch_test.add_node(12, 10, 2)
    patch_test.add_membrane_q6i(1, 1, 2, 8, 7, 'section')
    patch_test.add_membrane_q6i(2, 2, 3, 9, 8, 'section')
    patch_test.add_membrane_q6i(3, 3, 4, 10, 9, 'section')
    patch_test.add_membrane_q6i(4, 4, 5, 11, 10, 'section')
    patch_test.add_membrane_q6i(5, 5, 6, 12, 11, 'section')
    patch_test.add_restraint(1, True, True, False)
    patch_test.add_restraint(7, True, False, False)
    patch_test.add_load_pattern('CASE A')
    patch_test.add_point_load(12, 'CASE A', -1000, 0, 0)
    patch_test.add_point_load(6,  'CASE A', 1000, 0, 0)
    patch_test.add_load_pattern('CASE B')
    patch_test.add_point_load(12, 'CASE B', 0, 150, 0)
    patch_test.add_point_load(6,  'CASE B', 0, 150, 0)
    patch_test.solve()
    patch_test.show()


if __name__ == "__main__":
    main()
