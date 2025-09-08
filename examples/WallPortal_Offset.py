from milcapy import SystemModel

E = 2e6 # Modulo de Young
v = 0.3 # Coeficiente de Poisson

h = 2.4 # Altura de la pared
b = 1.2 # Ancho de la pared
l = 2.4 # Longitud de la vigas

t = [0.4, b] # Espesor de la pared
sec = [0.4, 0.4] # Sección de la vigas
F = 100 # Carga aplicada

wallPortal = SystemModel()
wallPortal.add_material("concreto", E, v)
wallPortal.add_rectangular_section("vig40x40", "concreto", *sec)
wallPortal.add_rectangular_section("muro", "concreto", *t)
wallPortal.add_node(1, 0, 0)
wallPortal.add_node(2, 0, h)
wallPortal.add_node(3, 0, 2*h)
wallPortal.add_node(4, 0, 3*h)
wallPortal.add_node(5, b+l, 0)
wallPortal.add_node(6, b+l, h)
wallPortal.add_node(7, b+l, 2*h)
wallPortal.add_node(8, b+l, 3*h)
wallPortal.add_elastic_timoshenko_beam(1, 1, 2, "muro")
wallPortal.add_elastic_timoshenko_beam(2, 2, 3, "muro")
wallPortal.add_elastic_timoshenko_beam(3, 3, 4, "muro")
wallPortal.add_elastic_timoshenko_beam(4, 5, 6, "muro")
wallPortal.add_elastic_timoshenko_beam(5, 6, 7, "muro")
wallPortal.add_elastic_timoshenko_beam(6, 7, 8, "muro")
wallPortal.add_elastic_euler_bernoulli_beam(7, 2, 6, "vig40x40")
wallPortal.add_elastic_euler_bernoulli_beam(8, 3, 7, "vig40x40")
wallPortal.add_elastic_euler_bernoulli_beam(9, 4, 8, "vig40x40")
wallPortal.add_end_length_offset(7, b/2, b/2)
wallPortal.add_end_length_offset(8, b/2, b/2)
wallPortal.add_end_length_offset(9, b/2, b/2)
wallPortal.add_restraint(1, (True, True, True))
wallPortal.add_restraint(5, (True, True, True))
wallPortal.add_load_pattern("carga")
wallPortal.add_point_load(2, "carga", fx=1*F)
wallPortal.add_point_load(3, "carga", fx=2*F)
wallPortal.add_point_load(4, "carga", fx=3*F)
wallPortal.solve()
wallPortal.show()



