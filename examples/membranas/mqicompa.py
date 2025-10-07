import milcapy as mp

E = 1500
v = 0.25
t = 1.00
p = 1000
coords = [(1, 0), (3, 2), (2, 3), (0, 1)] # RECTANGULO LOCAL


model = mp.SystemModel()
model.add_material("concreto", E, v)
model.add_shell_section("viga", "concreto", t)
for i, (x, y) in enumerate(coords):
    model.add_node(i+1, x, y)
model.add_restraint(1, *(True, True, False))
model.add_restraint(4, *(True, False, False))
model.add_load_pattern("carga")
model.add_point_load(2, "carga", fx= p)
model.add_point_load(3, "carga", fx=-p)

model.add_membrane_q6i(1, 1, 2, 3, 4, "viga")
model.solve()
# model.show()
model.plot_model('carga')

