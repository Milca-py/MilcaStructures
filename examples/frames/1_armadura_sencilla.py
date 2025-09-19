from milcatrusspy import Model

matK1 = [2.1e6, 0.3*0.4]
matK2 = [2.1e6, 0.4*0.4]

model = Model(ndm=2)

model.add_node(1, 0, 0)
model.add_node(2, 3, 0)
model.add_node(3, 5, 3)
model.add_node(4, 3, 3)
model.add_node(5, 0, 3)


model.add_element(1, 2, 1, *matK1)
model.add_element(2, 3, 2, *matK1)
model.add_element(3, 4, 3, *matK1)
model.add_element(4, 4, 5, *matK1)
model.add_element(5, 5, 1, *matK1)
model.add_element(6, 2, 4, *matK2)
model.add_element(7, 1, 4, *matK2)
model.add_element(8, 2, 5, *matK2)

model.set_restraints(1, True, True)
model.set_restraints(2, False, True)

model.set_load(3, 5, -6)
model.solve()
model.print_results()

model.plot_model(labels=True)
model.plot_deformed(scale=1000, labels=True)
model.plot_reactions()
model.plot_axial_forces(scale=0.03, labels=True)


