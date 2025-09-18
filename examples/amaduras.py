from milcapy import SystemModel

cercha = SystemModel()

dx = 1
dy = 1
nt = 70
cercha.add_material("concreto", 2.1e6, 0.2)
cercha.add_rectangular_section("truss", "concreto", 0.3, 0.5)
for i in range(nt):
    cercha.add_node(i+1, i*dx, 0)
    cercha.add_node(i+nt+1, i*dx, dy)

modelType = "truss"
disPath = {"truss": cercha.add_truss, "member": cercha.add_member}

for i in range(nt-1):
    disPath[modelType](i+1, i+1, i+2, "truss")
    disPath[modelType](i+nt+1, i+nt+1, i+nt+2, "truss")
    if nt-1 > i > 0:
        disPath[modelType](i+2*nt+1, i+1, i+nt+1, "truss")
    disPath[modelType](i+3*nt+1, i+1, i+nt+2, "truss")
    disPath[modelType](i+4*nt+1, i+2, i+nt+1, "truss")

cercha.add_restraint(1, *(True, True, False))
cercha.add_restraint(nt, *(True, True, False))
cercha.add_restraint(nt+1, *(True, True, False))
cercha.add_restraint(nt+nt, *(True, True, False))

cercha.add_load_pattern("Live Load")
cercha.add_point_load(int(nt/2)+1+nt, "Live Load", fy=-10*len(cercha.nodes))

for member in cercha.members.keys():
    cercha.add_releases(member, mi=True, mj=True)

cercha.solve()

cercha.get_results_excel("Live Load")
cercha.show()
