from core.element import Element
import numpy as np
from utils import rotation_matrix



def plot_deformed_element(element, ax, scale):
    
    # Obtener coordenadas iniciales de los nodos
    vertice_i = np.array(element.node_i.vertex.coordinates)  # (x_i, y_i)
    vertice_j = np.array(element.node_j.vertex.coordinates)  # (x_j, y_j)

    # Obtener desplazamientos de los nodos (se asume [ux, uy, θ])
    u_i = np.array(element.node_i.desplacement[:2]) # (u_xi, u_yi)
    u_j = np.array(element.node_j.desplacement[:2]) # (u_xj, u_yj)

    # Deflexión local en la dirección y local
    deflexiones = np.array(element.deflection) * scale

    # Longitud del elemento y su ángulo de inclinación
    L = element.length
    angle = element.angle_x  # Se asume que ya está en radianes

    # Discretización de la longitud del elemento
    n_points = len(deflexiones)
    s = np.linspace(0, L, n_points)  # Coordenada local x

    # Matriz de rotación para transformar coordenadas locales a globales
    c = np.cos(angle)
    s_ang = np.sin(angle)

    # Coordenadas locales iniciales (posición en el eje del elemento)
    x_local = s  # El eje x local va de 0 a L
    y_local = deflexiones  # Deflexión en la dirección y local

    # Convertir las coordenadas deformadas a globales
    x_global = x_local * c - y_local * s_ang
    y_global = x_local * s_ang + y_local * c

    # Aplicar desplazamiento rígido interpolado
    # weight_i = 1 - s / L
    # weight_j = s / L
    # rigid_disp_x = weight_i * u_i[0] + weight_j * u_j[0]
    # rigid_disp_y = weight_i * u_i[1] + weight_j * u_j[1]

    rigid_disp_x = np.linspace(u_i[0], u_j[0], n_points)
    rigid_disp_y = np.linspace(u_i[1], u_j[1], n_points)

    # Aplicar desplazamientos globales
    x_final = vertice_i[0] + x_global #+ rigid_disp_x
    y_final = vertice_i[1] + y_global #+ rigid_disp_y

    # dezplazamientos rigidos
    x_finall = np.array([vertice_i[0] + u_i[0] * scale, vertice_j[0] + u_j[0] * scale])
    y_finall = np.array([vertice_i[1] + u_i[1] * scale, vertice_j[1] + u_j[1] * scale])
    # Graficar la deformada del elemento en rojo
    ax.plot(x_final, y_final, '-', lw=1)
    ax.plot(x_finall, y_finall, lw=1, color="#444545", linestyle="--")

def deformed_element(element: Element, ax, scale):
    Lo = element.length
    # calculo de longitud deformada
    vertice_i = np.array(element.node_i.vertex.coordinates)  # (x_i, y_i)
    vertice_j = np.array(element.node_j.vertex.coordinates)  # (x_j, y_j)

    # Obtener desplazamientos de los nodos (se asume [ux, uy, θ])
    u_i = np.array(element.node_i.desplacement[:2]) # (u_xi, u_yi)
    u_j = np.array(element.node_j.desplacement[:2]) # (u_xj, u_yj)


    desp_i = vertice_i + u_i * scale
    desp_j = vertice_j + u_j * scale


    Lf = np.sqrt((desp_j[0] - desp_i[0])**2 + (desp_j[1] - desp_i[1])**2)
    # Lf = desp_i[0] - desp_j[0]

    # factor de escalamiento lontitudinal o axial
    s = Lf / Lo

    deflexion = element.deflection * scale
    x = np.linspace(0, Lo, len(deflexion))

    #  1. ESCALAR EL VECTOR VECTOR X
    x = x * s
    deformed = np.column_stack((x, deflexion))

    #  2. ROTAR EL VECTOR DE DEFLEXIONES
    # angulo de la coordenadas iniciales y finales de la deflexion con el eje x (betha)
    xid = 0
    ydi = deflexion[0]
    xfd = Lo
    ydf = deflexion[-1]
    betha = np.arctan((ydf - ydi) / (xfd - xid))

    # angulo de la coordenadas iniciales y finales de la deformada con el eje x (theta_p)
    theta_p = np.arctan((desp_j[1] - desp_i[1]) / (desp_j[0] - desp_i[0]))

    # deformada rotada
    theta = element.angle_x
    thetea = theta_p - betha
    print("theta", theta*180/np.pi)
    print("thetea", thetea*180/np.pi)
    
    
    deformed = np.dot(deformed, rotation_matrix(thetea).T)

    #  3. TRASLADAR EL VECTOR DE DEFLEXIONES
    deformed = deformed + desp_i

    #  4. DIBUJAR LA DEFORMADA
    ax.plot(deformed[:, 0], deformed[:, 1], color="#506e7f")
    
    # 5. DIBUJAR LA DEFORMADA RIGIDA
    x_finall = np.array([desp_i[0], desp_j[0]])
    y_finall = np.array([desp_i[1], desp_j[1]])
    # Graficar la deformada del elemento en rojo
    ax.plot(x_finall, y_finall, lw=1, color="#444545", linestyle="--")

def deformed_for_element(element: Element, ax, scale):
    Lo = element.length
    # calculo de longitud deformada
    vertice_i = np.array(element.node_i.vertex.coordinates)  # (x_i, y_i)
    vertice_j = np.array(element.node_j.vertex.coordinates)  # (x_j, y_j)
    
    vertices = np.array([vertice_i, vertice_j])

    # # Obtener desplazamientos de los nodos (se asume [ux, uy, θ])
    # u_i = np.array(element.node_i.desplacement[:2]) # (u_xi, u_yi)
    # u_j = np.array(element.node_j.desplacement[:2]) # (u_xj, u_yj)


    # desp_i = vertice_i + u_i * scale
    # desp_j = vertice_j + u_j * scale


    # Lf = desp_i[0] - desp_j[0]

    # # factor de escalamiento lontitudinal o axial
    # s = Lf / Lo

    # deflexion = element.deflection * scale
    # x = np.linspace(0, Lo, len(deflexion))

    # #  1. ESCALAR EL VECTOR VECTOR X
    # x = x * s
    # deformed = np.column_stack((x, deflexion))

    # #  2. ROTAR EL VECTOR DE DEFLEXIONES
    # # angulo de la coordenadas iniciales y finales de la deflexion con el eje x (betha)
    # xid = 0
    # ydi = deflexion[0]
    # xfd 