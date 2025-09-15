
#######################################################################################################################
            # NOTAS PARA EL DESARROLLO
#######################################################################################################################

###########################################
        # NOTAS GENERALES
###########################################
# FALTA IMPLEMENTAR EL TIPOS DE ELEMENTOS
# FALTA IMPLEMENTAR EL TIPOS DE CARGAS
# FALTA IMPLEMENTAR EL ALGUNOS PARAMETROS Y ATRIBUTOS DE LAS CLASES Y QUE YA ESTAN DEFINIDOS EN EL CODIGO
###########################################

###########################################
    # CORRECCIONES EN ELASTIC SUPPORT
###########################################
# FALTA VERIFICAR CON SAP2000
###########################################

###########################################
    # CORRECCIONES EN PRESCRIBED SUPPORT
###########################################
# FALTA VERIFICAR CON SAP2000
# FALTA PONER DE MANERA BONOTA LAS ASIGNACIONES
###########################################

###########################################
    # CORRECCIONES EN END LENGTH OFFSET
###########################################
# EN EJERCICIO 4.2 SALE DISTORCIONADO EL GRAFICO        NOT: Falta revision a fondo
# CALIBRAR PARA PONER EL FACTOR DE BRAZO RIGIDO         NOT: falta el postprocesamiento correcto, SAP2000: Lf = L - r*a
# CALIBRAR POARA qla y qlb == False                     OK ingreso de dato qi, qj, pi, pj son ahora qa, qb, pa, pb
# QUE LAS ASIGNACIONES SE MUESTREN EN EL MODELO         OK
# SI NO HAY CARGAS EN LOS BRAZOS RIGIDOS NO SE MUESTRAN OK
###########################################

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


# TOPICOS
# - releases
# - induccion de Truss
# - induccion de seccion variable
# - tabiques
# - Diafragmas rigidos


# MAS IMPLEMENTACIONES
# - mas secciones, t, h, i, c
# - implementar deformada rigida para areas





#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################



#######################
# documentar los metodos hechos en model para agregar cargas
# ME QUESE EN PLOTER PARA PLOTEAR A LOS CST, CONFIGURAR Y ANDIR EN PLOTEROPTIONS, Y EN UIDISPLAY.PY

# CORREGIR LA DEFORMADA DE LOS MIEMBROS LINEALES
# FALTA CALCULAR LOS ESFUERSO EN LOS ELEMENTOS Q6



# entre dos nodos concetados al aplicar local axis sale mal



# pasos para incorporar nuevos elementos finitos:
# 1. crear la clase del elemento finito y obtener la matriz de rigidez y carga correctas
# 2. en model agregar el metodo para añadir el elemento finito
# 3. en linear estatic modificar el ensamblaje de la matriz de rigidez global y vector de cargas
# 4. modificar el postprocesamiento y los resultados (agregar el metodo en mananger analysis process_displacements_for_membrane_q6i)
# 5. modificar el plotter:
#   - crear parametros de opciones (modificar los calculadores para que se ajusten a los nuevos elementos, length_mean, length_max)
#   - crear values para plotter
#   - crear metodos para plotter: plot, update, reset
#   - darle uso a los metodos en plotter, UIdisplay (update_changer, diagrams_and_deformed)
