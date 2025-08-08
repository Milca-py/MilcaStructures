from milcapy import SystemModel

model = SystemModel()

model.add_material(name="concreto", modulus_elasticity=2100000, poisson_ratio=0.2)
model.add_rectangular_section(name="seccion1", material_name="concreto", base=0.3, height=0.5)
model.add_rectangular_section(name="seccion2", material_name="concreto", base=0.5, height=0.5)
model.add_rectangular_section(name="seccion3", material_name="concreto", base=0.6, height=0.6)

######################################################
########### MODELO SIN END LENGTH OFFSET #############
######################################################
model.add_node(1, 0, 0)
model.add_node(2, 0, 5)
model.add_node(3, 0, 8.5)
model.add_node(4, 0, 12)
model.add_node(5, 7, 0)
model.add_node(6, 7, 5)
model.add_node(7, 7, 8.5)
model.add_node(8, 7, 12)

model.add_member(1, 1, 2, "seccion3")
model.add_member(2, 2, 3, "seccion3")
model.add_member(3, 3, 4, "seccion3")
model.add_member(4, 5, 6, "seccion2")
model.add_member(5, 6, 7, "seccion2")
model.add_member(6, 7, 8, "seccion2")
model.add_member(7, 2, 6, "seccion1")
model.add_member(8, 3, 7, "seccion1")
model.add_member(9, 4, 8, "seccion1")

model.add_restraint(1, (True, True, True))
model.add_restraint(5, (True, True, True))

model.add_load_pattern("Live Load")
model.add_point_load(2, "Live Load", 5, 0, 0, "GLOBAL")
model.add_point_load(3, "Live Load", 10, 0, 0, "GLOBAL")
model.add_point_load(4, "Live Load", 20, 0, 0, "GLOBAL")

model.add_distributed_load(7, "Live Load", -5, -5) ###
model.add_distributed_load(8, "Live Load", -2, -6) ###
model.add_distributed_load(9, "Live Load", -4, -3) ###


######################################################
######### MODELO CON END LENGTH OFFSET F=1 ###########
######################################################
model.add_node(9, 14, 0)
model.add_node(10, 14, 5)
model.add_node(11, 14, 8.5)
model.add_node(12, 14, 12)
model.add_node(13, 21, 0)
model.add_node(14, 21, 5)
model.add_node(15, 21, 8.5)
model.add_node(16, 21, 12)

model.add_member(10, 9, 10, "seccion3")
model.add_member(11, 10, 11, "seccion3")
model.add_member(12, 11, 12, "seccion3")
model.add_member(13, 13, 14, "seccion2")
model.add_member(14, 14, 15, "seccion2")
model.add_member(15, 15, 16, "seccion2")
model.add_member(16, 10, 14, "seccion1")
model.add_member(17, 11, 15, "seccion1")
model.add_member(18, 12, 16, "seccion1")

model.add_restraint(9, (True, True, True))
model.add_restraint(13, (True, True, True))

model.add_point_load(10, "Live Load", 5, 0, 0, "GLOBAL")
model.add_point_load(11, "Live Load", 10, 0, 0, "GLOBAL")
model.add_point_load(12, "Live Load", 20, 0, 0, "GLOBAL")

model.add_distributed_load(16, "Live Load", -5, -5) ###
model.add_distributed_load(17, "Live Load", -2, -6) ###
model.add_distributed_load(18, "Live Load", -4, -3) ###

model.add_end_length_offset(16, 0.3, 0.25)
model.add_end_length_offset(17, 0.3, 0.25)
model.add_end_length_offset(18, 0.3, 0.25)


######################################################
######### MODELO CON END LENGTH OFFSET F=0.5 #########
######################################################
model.add_node(17, 28, 0)
model.add_node(18, 28, 5)
model.add_node(19, 28, 8.5)
model.add_node(20, 28, 12)
model.add_node(21, 35, 0)
model.add_node(22, 35, 5)
model.add_node(23, 35, 8.5)
model.add_node(24, 35, 12)

model.add_member(20, 17, 18, "seccion3")
model.add_member(21, 18, 19, "seccion3")
model.add_member(22, 19, 20, "seccion3")
model.add_member(23, 21, 22, "seccion2")
model.add_member(24, 22, 23, "seccion2")
model.add_member(25, 23, 24, "seccion2")
model.add_member(26, 18, 22, "seccion1")
model.add_member(27, 19, 23, "seccion1")
model.add_member(28, 20, 24, "seccion1")

model.add_restraint(17, (True, True, True))
model.add_restraint(21, (True, True, True))

model.add_point_load(18, "Live Load", 5, 0, 0, "GLOBAL")
model.add_point_load(19, "Live Load", 10, 0, 0, "GLOBAL")
model.add_point_load(20, "Live Load", 20, 0, 0, "GLOBAL")

model.add_distributed_load(26, "Live Load", -5, -5) ###
model.add_distributed_load(27, "Live Load", -2, -6) ###
model.add_distributed_load(28, "Live Load", -4, -3) ###

model.add_end_length_offset(26, 0.3, 0.25, fla=0.3, flb=0.3)
model.add_end_length_offset(27, 0.3, 0.25, fla=0.3, flb=0.3)
model.add_end_length_offset(28, 0.3, 0.25, fla=0.3, flb=0.3)

#####################################################
################# PORTICO 1 X 1 #####################
#####################################################
model.add_node(25, 42, 0)
model.add_node(26, 42, 5)
model.add_node(27, 49, 5)
model.add_node(28, 49, 0)
model.add_member(29, 25, 26, "seccion3")
model.add_member(30, 26, 27, "seccion1")
model.add_member(31, 27, 28, "seccion2")
model.add_restraint(25, (True, True, True))
model.add_restraint(28, (True, True, True))
model.add_point_load(26, "Live Load", 20, 0, 0, "GLOBAL")
model.add_end_length_offset(30, 0.5, 1.0)


model.add_node(29, 42, 8)
model.add_node(30, 42, 13)
model.add_node(31, 49, 13)
model.add_node(32, 49, 8)
model.add_member(32, 29, 30, "seccion3")
model.add_member(33, 30, 31, "seccion1")
model.add_member(34, 31, 32, "seccion2")
model.add_restraint(29, (True, True, True))
model.add_restraint(32, (True, True, True))
model.add_point_load(30, "Live Load", 20, -20, 120, "GLOBAL")
model.add_end_length_offset(33, 0.5, 1.0)

#####################################################
#### MODEL WITH END LENGTH OFFSET F=1 + P_LOAD ######
#####################################################
model.add_node(33, 56, 0)
model.add_node(34, 56, 5)
model.add_node(35, 56, 8.5)
model.add_node(36, 56, 12)
model.add_node(37, 63, 0)
model.add_node(38, 63, 5)
model.add_node(39, 63, 8.5)
model.add_node(40, 63, 12)

model.add_member(36, 33, 34, "seccion3")
model.add_member(37, 34, 35, "seccion3")
model.add_member(38, 35, 36, "seccion3")
model.add_member(39, 37, 38, "seccion2")
model.add_member(40, 38, 39, "seccion2")
model.add_member(41, 39, 40, "seccion2")
model.add_member(42, 34, 38, "seccion1")
model.add_member(43, 35, 39, "seccion1")
model.add_member(44, 36, 40, "seccion1")

model.add_restraint(33, (True, True, True))
model.add_restraint(37, (True, True, True))

model.add_point_load(34, "Live Load", 5, 0, 0, "GLOBAL")
model.add_point_load(35, "Live Load", 10, 0, 0, "GLOBAL")
model.add_point_load(36, "Live Load", 20, 0, 0, "GLOBAL")

model.add_distributed_load(42, "Live Load", -5, -5) ###
model.add_distributed_load(43, "Live Load", -2, -6) ###
model.add_distributed_load(44, "Live Load", -4, -3) ###
model.add_distributed_load(44, "Live Load", -4, -3, "LOCAL","LOCAL_1") ###

model.add_end_length_offset(42, 0.3, 0.25)
model.add_end_length_offset(43, 0.3, 0.25)
model.add_end_length_offset(44, 0.3, 0.25)

######################################################
################ MODEL WITH P_LOAD ###################
######################################################
model.add_node(41, 70, 0)
model.add_node(42, 70, 5)
model.add_node(43, 70, 8.5)
model.add_node(44, 70, 12)
model.add_node(45, 77, 0)
model.add_node(46, 77, 5)
model.add_node(47, 77, 8.5)
model.add_node(48, 77, 12)

model.add_member(45, 41, 42, "seccion3")
model.add_member(46, 42, 43, "seccion3")
model.add_member(47, 43, 44, "seccion3")
model.add_member(48, 45, 46, "seccion2")
model.add_member(49, 46, 47, "seccion2")
model.add_member(50, 47, 48, "seccion2")
model.add_member(51, 42, 46, "seccion1")
model.add_member(52, 43, 47, "seccion1")
model.add_member(53, 44, 48, "seccion1")

model.add_restraint(41, (True, True, True))
model.add_restraint(45, (True, True, True))

model.add_point_load(42, "Live Load", 5, 0, 0, "GLOBAL")
model.add_point_load(43, "Live Load", 10, 0, 0, "GLOBAL")
model.add_point_load(44, "Live Load", 20, 0, 0, "GLOBAL")

model.add_distributed_load(51, "Live Load", -5, -5) ###
model.add_distributed_load(52, "Live Load", -2, -6) ###
model.add_distributed_load(53, "Live Load", -4, -3) ###
model.add_distributed_load(53, "Live Load", -4, -3, "LOCAL","LOCAL_1") ###

#####################################################
## MODEL WITH END LENGTH OFFSET F=1 qla, qlb=False ##
#####################################################
model.add_node(49, 86, 0)
model.add_node(50, 86, 5)
model.add_node(51, 86, 8.5)
model.add_node(52, 86, 12)
model.add_node(53, 93, 0)
model.add_node(54, 93, 5)
model.add_node(55, 93, 8.5)
model.add_node(56, 93, 12)

model.add_member(54, 49, 50, "seccion3")
model.add_member(55, 50, 51, "seccion3")
model.add_member(56, 51, 52, "seccion3")
model.add_member(57, 53, 54, "seccion2")
model.add_member(58, 54, 55, "seccion2")
model.add_member(59, 55, 56, "seccion2")
model.add_member(60, 50, 54, "seccion1")
model.add_member(61, 51, 55, "seccion1")
model.add_member(62, 52, 56, "seccion1")

model.add_restraint(49, (True, True, True))
model.add_restraint(53, (True, True, True))

model.add_point_load(50, "Live Load", 5, 0, 0, "GLOBAL")
model.add_point_load(51, "Live Load", 10, 0, 0, "GLOBAL")
model.add_point_load(52, "Live Load", 20, 0, 0, "GLOBAL")

model.add_distributed_load(60, "Live Load", -5, -5) ###
model.add_distributed_load(61, "Live Load", -2, -6) ###
model.add_distributed_load(62, "Live Load", -4, -3) ###

model.add_end_length_offset(60, 0.3, 0.25, False, False)
model.add_end_length_offset(61, 0.3, 0.25, False, False)
model.add_end_length_offset(62, 0.3, 0.25, False, False)

######################################################
#################### SOLVE AND SHOW ##################
######################################################
model.solve()


import numpy as np
ele_id = 28
fi = model.results["Live Load"].get_member_internal_forces(ele_id)  # OK
mtt = model.members[ele_id].length_offset_transformation_matrix()   # OK
mtt = np.linalg.inv(mtt.T)
fi_flex = mtt @ fi


print(f"FUERZAS INTERNAS EN LOS EXTREMOS DEL ELEMENTO: \n {fi}")
print(f"MATRIZ DE TRANSFORMACION DE BRAZO RIGIDO: \n {mtt}")
print(f"FUERZAS INTERNAS EN LOS EXTREMOS DEL ELEMENTO FLEXIBLE: \n {fi_flex}")


model.show()




#######################################################
            # NOTAS PARA EL DESARROLLO
#######################################################
# CALIBRAR PARA PONER EL FACTOR DE BRAZO RIGIDO         NOT: falta el postprocesamiento correcto - SAP2000 Lf = L - r*a
# CALIBRAR POARA qla y qlb == False                     OK ingreso de dato qi, qj, pi, pj son ahora qa, qb, pa, pb
# QUE LAS ASIGNACIONES SE MUESTREN EN EL MODELO         OK
# SI NO HAY CARGAS EN LOS BRAZOS RIGIDOS NO SE MUESTRAN OK
#######################################################