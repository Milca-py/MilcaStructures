

from core.system import SytemMilcaModel
# from main import SytemMilcaModel

# crear el modelo estructural
model = SytemMilcaModel()


E = 210e9
v = 0.3
g = 0.0
b = 0.3
h = 0.6


# agregar materiales
model.add_material("concreto", E, v, g)

# agregar secciones
model.add_rectangular_section("seccion1", "concreto", b, h)

# agregar nodos
model.add_node(1, (0, 0))
model.add_node(2, (0, 5))
model.add_node(3, (5, 5))
model.add_node(4, (5, 0))

# agregar elementos
model.add_element(1, "FRAME", 1, 2, "seccion1")
model.add_element(2, "FRAME", 2, 3, "seccion1")
model.add_element(3, "FRAME", 3, 4, "seccion1")

# agregar restricciones
model.add_restraint(1, (True, True, True))
model.add_restraint(4, (True, True, True))

# agregar patrones de carga
model.add_load_pattern("muerta")

# agregar cargas puntuales
model.add_point_load(2, "muerta", fx=100)

# resolver el modelo
model.solve()

