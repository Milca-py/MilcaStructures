from function.widgets import InternalForceDiagramWidget, DiagramConfig
import pandas as pd
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# # Configurar notación científica con 2 decimales
# np.set_printoptions(precision=4, suppress=True, formatter={
#                     'float_kind': '{:0.3e}'.format})
# posibles unidades (fuerza, longitud, temperatura)
# lb_in_F	1
# lb_ft_F	2
# kip_in_F	3
# kip_ft_F	4
# kN_mm_C	5
# kN_m_C	6
# kgf_mm_C	7
# kgf_m_C	8
# N_mm_C	9
# N_m_C	    10
# Tonf_mm_C	11
# Tonf_m_C	12
# kN_cm_C	13
# kgf_cm_C	14
# N_cm_C	15
# Tonf_cm_C	16

# Definir unidades
tn = 1
m = 1
# Definir unidades base
fuerza = tn
longitud = m

# Definir material
E = 2.1 * 10**6 * fuerza / (longitud**2)
v = 0.2
G = E / (2 * (1 + v))

# Condiciones de análisis
def_corte = 1  # colocar 1 si desea incluir deformaciones por cortadura

# Definir secciones
secciones = {
    "Seccion": list(np.array([0, 1, 2])),
    "b": list(np.array([0.3, 0.5, 0.6]) * m),
    "h": list(np.array([0.5, 0.5, 0.6]) * m),
    "D": list(np.array([0.0, 0.0, 0.0]) * m),
}

# Coordenadas de los nodos
nodos = {
    "Nudo": list(np.array([0, 1, 2, 3, 4, 5, 6, 7])),
    "X": list(np.array([0, 0, 0, 0, 7, 7, 7, 7]) * m),
    "Y": list(np.array([0, 5, 8.5, 12, 0, 5, 8.5, 12]) * m),
}

# Definir dirección de las barras y secciones (índices corregidos)
barras = {
    "inicio": list(np.array([0, 1, 2, 4, 5, 6, 1, 2, 3])),
    "fin": list(np.array([1, 2, 3, 5, 6, 7, 5, 6, 7])),
    "seccion": list(np.array([2, 2, 2, 1, 1, 1, 0, 0, 0])),
}

# # Ploteo inicial
# plt.figure(figsize=(5, 7))
# plt.plot(nodos["X"], nodos["Y"], 'ro')
# for i in range(len(barras["inicio"])):
#     plt.plot(
#         [nodos["X"][barras["inicio"][i]], nodos["X"][barras["fin"][i]]],
#         [nodos["Y"][barras["inicio"][i]], nodos["Y"][barras["fin"][i]]],
#         'b'
#     )
#     plt.text(
#         (nodos["X"][barras["inicio"][i]] + nodos["X"][barras["fin"][i]]) / 2,
#         (nodos["Y"][barras["inicio"][i]] + nodos["Y"][barras["fin"][i]]) / 2,
#         str(i)
#     )
# for i in nodos["Nudo"]:
#     plt.text(nodos["X"][i], nodos["Y"][i], str(i))
# plt.axis('equal')
# plt.tight_layout()
# plt.show()
##########################################################################################
# A) PRIMERA ESTRUCTURA DE PROGRAMACION PARA DIBUJO DE REFERENCIA
N_b = len(barras["inicio"])
X_c = np.zeros([2, N_b])
Y_c = np.zeros([2, N_b])
for j in range(N_b):
    X_c[0][j] = nodos["X"][barras["inicio"][j]]
    X_c[1][j] = nodos["X"][barras["fin"][j]]
    Y_c[0][j] = nodos["Y"][barras["inicio"][j]]
    Y_c[1][j] = nodos["Y"][barras["fin"][j]]
##########################################################################################
# paso 7 - reaciiones (1: restringido, 0: libre)
reacciones = {
    "Apoyo": list(np.array([0, 4])),
    "Rx": list(np.array([1, 1])),
    "Ry": list(np.array([1, 1])),
    "Rz": list(np.array([1, 1]))
}

# paso 8 - cargas distribuidas y brazos regidos

cargas_distribuidas = {
    "Barra": list(np.array([6, 7, 8])),
    "qa": list(np.array([5, 2, 4]) * fuerza / m),
    "qb": list(np.array([5, 6, 3]) * fuerza / m),
    "g": list(np.array([0, 0, 0])),
    "bri": list(np.array([0.3, 0.3, 0.3]) * m),
    "brf": list(np.array([0.25, 0.25, 0.25]) * m),
    "qbri": list(np.array([1, 1, 1])),
    "qbrf": list(np.array([1, 1, 1]))
}

# paso 9 - cargas concentradas
cargas_concentradas = {
    "Nudo": list(np.array([1, 2, 3])),
    "Fx": list(np.array([5, 10, 20]) * fuerza),
    "Fy": list(np.array([0, 0, 0]) * fuerza),
    "Mz": list(np.array([0, 0, 0]) * fuerza*m)
}

# B) ESTRUCTURA DE PROGRMACIÓN PARA OBTENER LA MATRIZ DE RIGIDEZ GLOBAL
# Numero de secciones
N_sec = len(secciones["Seccion"])
# Numero de nodos
N_n = len(nodos["Nudo"])
# Numero de barras
N_b = len(barras["inicio"])
# Numero de DOF
N_gl = 3 * N_n
# Numero de cargas concentradas
N_cp = len(cargas_concentradas["Nudo"])
# nuemro de cargas distribuidas
N_br = len(cargas_distribuidas["Barra"])
N_cd = len(cargas_distribuidas["Barra"])
# Numero de apooyos
N_a = len(reacciones["Apoyo"])
# nuemro de reacciones
N_r = sum(reacciones["Rx"][i] for i in range(N_a)) +\
    sum(reacciones["Ry"][i] for i in range(N_a)) +\
    sum(reacciones["Rz"][i] for i in range(N_a))
# contador de barras
i = np.arange(len(barras["inicio"]))
# contador de nudos
nn = np.arange(len(nodos["Nudo"]))
# dimensiones de cada elemento
Base = np.array([secciones["b"][i] for i in barras["seccion"]])
Peralte = np.array([secciones["h"][i] for i in barras["seccion"]])
Diametro = np.array([secciones["D"][i] for i in barras["seccion"]])
# area de cada elemento
A = np.zeros(N_b)
for i in range(N_b):
    if Diametro[i] != 0:
        A[i] = np.pi * Diametro[i]**2 / 4
    else:
        A[i] = Base[i] * Peralte[i]
# momento de inercia de cada elemento
I = np.zeros(N_b)
for i in range(N_b):
    if Diametro[i] != 0:
        I[i] = np.pi * Diametro[i]**4 / 64
    else:
        I[i] = Base[i] * Peralte[i]**3 / 12
# longitud de cada elemento
L = np.zeros(N_b)
for i in range(N_b):
    L[i] = np.sqrt((nodos["X"][barras["fin"][i]] - nodos["X"][barras["inicio"][i]])**2 +
                   (nodos["Y"][barras["fin"][i]] - nodos["Y"][barras["inicio"][i]])**2)
Lo = L.copy()
# angulo de rotacion de cada elemento
theta_x = np.zeros(N_b)
for j in range(N_b):
    xi = nodos["X"][barras["inicio"][j]]
    xf = nodos["X"][barras["fin"][j]]
    yi = nodos["Y"][barras["inicio"][j]]
    yf = nodos["Y"][barras["fin"][j]]
    if yf - yi == 0 and xf - xi < 0:
        theta_x[j] = -np.pi
    else:
        if xf - xi == 0:
            theta_x[j] = np.pi / 2*np.sign(yf - yi)
        elif xf - xi < 0:
            theta_x[j] = np.arctan((yf - yi) / (xf - xi)) + np.pi
        else:
            theta_x[j] = np.arctan((yf - yi) / (xf - xi))
# angulo de giro de cada elemento con respecto al eje y
theta_y = np.zeros(N_b)
for i in range(N_b):
    theta_y[i] = -theta_x[i] + np.pi / 2
# matriz de transformacion de coordenadas para cada elemento
T = np.zeros((N_b, 6, 6))
for i in range(N_b):
    T[i] = np.array([
        [np.cos(theta_x[i]), np.sin(theta_x[i]), 0, 0, 0, 0],
        [-np.sin(theta_x[i]), np.cos(theta_x[i]), 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, np.cos(theta_x[i]), np.sin(theta_x[i]), 0],
        [0, 0, 0, -np.sin(theta_x[i]), np.cos(theta_x[i]), 0],
        [0, 0, 0, 0, 0, 1]
    ])
# matriz que identifica las reacciones en los nodos
Mre = np.zeros((N_n, 3))
for i in range(N_a):
    Mre[reacciones["Apoyo"][i]][0] = reacciones["Rx"][i]
    Mre[reacciones["Apoyo"][i]][1] = reacciones["Ry"][i]
    Mre[reacciones["Apoyo"][i]][2] = reacciones["Rz"][i]
# matriz que asigna DOF a cada nodo
Mgl = np.zeros((N_n, 3))
i = N_gl
j = 0
for k in range(N_n):
    for l in range(3):
        if Mre[k][l] == 1:
            i = i - 1
            Mgl[k][l] = i
        else:
            Mgl[k][l] = j
            j = j + 1
# matriz que asigna DOF a cada barra
a = np.zeros((N_b, 6)).T
for i in range(N_b):
    for j in range(3):
        a[j][i] = Mgl.T[j][nodos["Nudo"].index(barras["inicio"][i])]
        a[j+3][i] = Mgl.T[j][nodos["Nudo"].index(barras["fin"][i])]
# operadores para el manejo de unidades compatibles
χ = 1/longitud
φ = χ**2
Υ = χ/(φ*fuerza)

# matriz de transformacion para elementos con brazos rigidos
H = np.zeros((N_b, 6, 6))
for i in range(N_b):
    H[i] = np.identity(6)
    for j in range(N_br):
        if cargas_distribuidas["Barra"][j] == i:
            H[i][1, 2] = cargas_distribuidas["bri"][j]*χ
            H[i][4, 5] = -cargas_distribuidas["brf"][j]*χ
# longitud final de cada elemento restando los brazos rigidos
for i in range(N_b):
    for j in range(N_br):
        if cargas_distribuidas["Barra"][j] == i:
            L[i] = L[i] - cargas_distribuidas["bri"][j] - \
                cargas_distribuidas["brf"][j]
# longitud de brazos rigidos de cada elemento
Br = np.zeros((N_b, 2))
for i in range(N_b):
    for j in range(N_br):
        if cargas_distribuidas["Barra"][j] == i:
            Br[i][0] = cargas_distribuidas["bri"][j]
            Br[i][1] = cargas_distribuidas["brf"][j]
# factor que mide el efecto de corte en la deformaciones
As = np.zeros(N_b)
for i in range(N_b):
    if Diametro[i] != 0:
        As[i] = (9/10)*A[i]
    else:
        As[i] = (5/6)*A[i]

β_y = 12*E*I/(G*As)
if def_corte == 1:
    β_y = β_y/L**2
else:
    β_y = (β_y/L**2)*def_corte
# matriz de rigidez de cada elemento en coordenadas locales
K_e = np.zeros((N_b, 6, 6))
for i in range(N_b):
    k11 = E*A[i]/Lo[i]
    k12 = 0
    k13 = 0
    k14 = -E*A[i]/Lo[i]
    k15 = 0
    k16 = 0
    k21 = 0
    k22 = 12*E*I[i]/((1+β_y[i])*L[i]**3)
    k23 = 6*E*I[i]*χ/((1+β_y[i])*L[i]**2)
    k24 = 0
    k25 = -12*E*I[i]/((1+β_y[i])*L[i]**3)
    k26 = 6*E*I[i]*χ/((1+β_y[i])*L[i]**2)
    k31 = 0
    k32 = 6*E*I[i]*χ/((1+β_y[i])*L[i]**2)
    k33 = (4+β_y[i])*E*I[i]*φ/((1+β_y[i])*L[i])
    k34 = 0
    k35 = -6*E*I[i]*χ/((1+β_y[i])*L[i]**2)
    k36 = (2-β_y[i])*E*I[i]*φ/((1+β_y[i])*L[i])
    k41 = -E*A[i]/Lo[i]
    k42 = 0
    k43 = 0
    k44 = E*A[i]/Lo[i]
    k45 = 0
    k46 = 0
    k51 = 0
    k52 = -12*E*I[i]/((1+β_y[i])*L[i]**3)
    k53 = -6*E*I[i]*χ/((1+β_y[i])*L[i]**2)
    k54 = 0
    k55 = 12*E*I[i]/((1+β_y[i])*L[i]**3)
    k56 = -6*E*I[i]*χ/((1+β_y[i])*L[i]**2)
    k61 = 0
    k62 = 6*E*I[i]*χ/((1+β_y[i])*L[i]**2)
    k63 = (2-β_y[i])*E*I[i]*φ/((1+β_y[i])*L[i])
    k64 = 0
    k65 = -6*E*I[i]*χ/((1+β_y[i])*L[i]**2)
    k66 = (4+β_y[i])*E*I[i]*φ/((1+β_y[i])*L[i])
    K_e[i] = np.array([
        [k11, k12, k13, k14, k15, k16],
        [k21, k22, k23, k24, k25, k26],
        [k31, k32, k33, k34, k35, k36],
        [k41, k42, k43, k44, k45, k46],
        [k51, k52, k53, k54, k55, k56],
        [k61, k62, k63, k64, k65, k66]
    ])

# matriz de rigidez de cada elemento de la parte flexible
K_et = K_e.copy()
# matriz de rigidez de cada elemento considenando los brazos rigidos
for i in range(N_b):
    K_e[i] = np.dot(np.dot(H[i].T, K_et[i]), H[i])
# matriz de rigidez de cada elemento en coordenadas globales
K_E = np.zeros((N_b, 6, 6))
for i in range(N_b):
    K_E[i] = np.dot(np.dot(T[i].T, K_e[i]), T[i])
# ENSAMBLAJE DE LA MATRIZ DE RIGIDEZ GLOBAL
K_g = np.zeros((N_b, N_gl, N_gl))
for i in range(N_b):
    for j in range(6):
        EG = a.T[i][j]
        for k in range(6):
            CG = a.T[i][k]
            K_g[i][int(EG), int(CG)] = K_E[i][j, k]
# Matriz de rigidez total
K_G = sum(K_g[i] for i in range(N_b))
# Grados de libertad libres
N_cond = N_gl - N_r
# divicion de la matriz regidez segun grados libertad libes y restringidos
K_xx = K_G[:N_cond, :N_cond]
K_xr = K_G[:N_cond, N_cond:N_gl]
K_rx = K_G[N_cond:N_gl, :N_cond]
K_rr = K_G[N_cond:N_gl, N_cond:N_gl]

# C) ESTRUCTURA DE PROGRAMACIÓN PARA OBTENER EL VECTOR DE CARGAS
# Matriz de compatibilidad de unidades
Unidades = np.array([

    [Υ*χ,   Υ*χ*0,   Υ*φ*0, Υ*χ*0, Υ*χ*0, Υ*φ*0],
    [Υ*χ*0, Υ*χ,     Υ*φ*0, Υ*χ*0, Υ*χ*0, Υ*φ*0],
    [Υ*χ*0, Υ*χ*0,   Υ*φ,   Υ*χ*0, Υ*χ*0, Υ*φ*0],
    [Υ*χ*0, Υ*χ*0,   Υ*φ*0, Υ*χ,   Υ*χ*0, Υ*φ*0],
    [Υ*χ*0, Υ*χ*0,   Υ*φ*0, Υ*χ*0, Υ*χ,   Υ*φ*0],
    [Υ*χ*0, Υ*χ*0,   Υ*φ*0, Υ*χ*0, Υ*χ*0, Υ*φ],

])
# vector de cargas en cada elemento sin considerar cargas sobre brazos rigidos
q = np.zeros((N_b, 6, 1))
for i in range(N_b):
    for j in range(N_cd):
        # reconoce la barra cargada conQlod
        if cargas_distribuidas["Barra"][j] == i:
            if cargas_distribuidas["g"][j] == 1:  # si es direccion gravitatoria
                γ = 1
                ξ = np.cos(theta_x[i])
            else:
                γ = 0  # error γ != Υ
                ξ = 1
            qa = cargas_distribuidas["qa"][j]*Υ*ξ
            qb = cargas_distribuidas["qb"][j]*Υ*ξ
            LL = L[i]*χ
            ιi = cargas_distribuidas["bri"][j]*χ
            ιf = cargas_distribuidas["brf"][j]*χ
            def qx(x): return ((qb - qa)/(LL + ιi + ιf))*x + qa
            if qa == qb:
                q11 = qa
                q22 = qa
            else:
                q11 = qx(ιi)
                q22 = qx(LL + ιi)
            if cargas_distribuidas["qbri"][j] == 0:
                q11 = qa
            else:
                q11 = q11
            if cargas_distribuidas["qbrf"][j] == 0:
                q22 = qb
            else:
                q22 = q22
            A_1 = (q22-q11)/LL
            B_1 = q11
            M_1 = np.array([

                [LL**2/2, LL],
                [LL**3/12*(2-β_y[i]), LL**2/2]

            ])
            N_1 = np.array([

                [A_1*LL**4/24 + B_1*LL**3/6],
                [A_1*LL**5*(0.6 - β_y[i])/72 + B_1*LL**4*(1 - β_y[i])/24]

            ])
            C = np.dot(np.linalg.inv(M_1), N_1)
            q[i][1, 0] = -float(C[0][0])
            q[i][2, 0] = float(C[1][0])
            q[i][4, 0] = -A_1*LL**2/2 - B_1*LL + float(C[0][0])
            q[i][5, 0] = -(-A_1*LL**3/6 - B_1*LL**2/2 +
                           float(C[0][0])*LL + float(C[1][0]))
            q[i][0, 0] = -(q22*0.5 + q11*1)/3 * \
                (1/np.cos(theta_x[i])) * (LL*np.sin(theta_x[i])*γ)
            q[i][3, 0] = -(q22*1 + q11*0.5)/3 * \
                (1/np.cos(theta_x[i])) * (LL*np.sin(theta_x[i])*γ)
# vector de cargas en debido a las cargas muesrtas sobre los brazos rigidos de cada elemento
q_p = np.zeros((N_b, 6, 2))
for i in range(N_b):
    for j in range(N_cd):
        # reconoce la barra cargada conQlod
        if cargas_distribuidas["Barra"][j] == i:
            ιi = cargas_distribuidas["bri"][j]*χ
            ιf = cargas_distribuidas["brf"][j]*χ
            if cargas_distribuidas["g"][j] == 1:  # si es direccion gravitatoria
                γ = 1
                ξ = np.cos(theta_x[i])
            else:
                γ = 0  # error γ != Υ
                ξ = 1
            qa = cargas_distribuidas["qa"][j]*Υ*ξ
            qb = cargas_distribuidas["qb"][j]*Υ*ξ
            Long = L[i]*χ
            def qx(x): return ((qb - qa)/(Long + ιi + ιf))*x + qa
            if qa == qb:
                q11 = qa*cargas_distribuidas["qbri"][j]
                q12 = qa*cargas_distribuidas["qbri"][j]
                q13 = qa*cargas_distribuidas["qbrf"][j]
                q14 = qa*cargas_distribuidas["qbrf"][j]
            else:
                q11 = qx(0)*cargas_distribuidas["qbri"][j]
                q12 = qx(ιi)*cargas_distribuidas["qbri"][j]
                q13 = qx(Long + ιi)*cargas_distribuidas["qbrf"][j]
                q14 = qx(Long + ιi + ιf)*cargas_distribuidas["qbrf"][j]
            if ιi != 0:
                def Ni_1(x): return (1 - 3*(x/ιi)**2 + 2*(x/ιi)**3)
                def Ni_2(x): return x*(1 - x/ιi)**2
                def Ni_3(x): return (3*(x/ιi)**2 - 2*(x/ιi)**3)
                def Ni_4(x): return x*((x/ιi)**2 - x/ιi)
                def qxi(x): return ((q12 - q11)/ιi)*x + q11
            else:
                def Ni_1(x): return 0
                def Ni_2(x): return 0
                def Ni_3(x): return 0
                def Ni_4(x): return 0
                def qxi(x): return 0
            if ιf != 0:
                def Nf_1(x): return (1 - 3*(x/ιf)**2 + 2*(x/ιf)**3)
                def Nf_2(x): return x*(1 - x/ιf)**2
                def Nf_3(x): return (3*(x/ιf)**2 - 2*(x/ιf)**3)
                def Nf_4(x): return x*((x/ιf)**2 - x/ιf)
                def qxf(x): return ((q14 - q13)/ιf)*x + q13
            else:
                def Nf_1(x): return 0
                def Nf_2(x): return 0
                def Nf_3(x): return 0
                def Nf_4(x): return 0
                def qxf(x): return 0
            q_p[i][0, 0] = -(q12*1 + q11*0.5)/3 * \
                (ιi*np.tan(theta_x[i])*γ)
            q_p[i][3, 0] = -(q14*0.5 + q13*1)/3 * \
                (ιf*np.tan(theta_x[i])*γ)
            q_p[i][0, 1] = -(q12*0.5 + q11*1)/3 * \
                (ιi*np.tan(theta_x[i])*γ)
            q_p[i][3, 1] = -(q14*1 + q13*0.5)/3 * \
                (ιf*np.tan(theta_x[i])*γ)
            resultado, _ = quad(lambda x: Ni_3(x) * qxi(x), 0, ιi)
            q_p[i][1, 0] = -resultado
            resultado, _ = quad(lambda x: Ni_4(x) * qxi(x), 0, ιi)
            q_p[i][2, 0] = -resultado
            resultado, _ = quad(lambda x: Nf_1(x) * qxf(x), 0, ιf)
            q_p[i][4, 0] = -resultado
            resultado, _ = quad(lambda x: Nf_2(x) * qxf(x), 0, ιf)
            q_p[i][5, 0] = -resultado
            resultado, _ = quad(lambda x: Ni_1(x) * qxi(x), 0, ιi)
            q_p[i][1, 1] = -resultado
            resultado, _ = quad(lambda x: Ni_2(x) * qxi(x), 0, ιi)
            q_p[i][2, 1] = -resultado
            resultado, _ = quad(lambda x: Nf_3(x) * qxf(x), 0, ιf)
            q_p[i][4, 1] = -resultado
            resultado, _ = quad(lambda x: Nf_4(x) * qxf(x), 0, ιf)
            q_p[i][5, 1] = -resultado
# vector de cargas en sistema global
Q_p_D = np.zeros((N_b, 6, 1))
for i in range(N_b):
    Q_p_D[i, :, 0] = np.dot(
        T[i].T, (np.dot(H[i].T, q[i, :, 0] + q_p[i, :, 0]) + q_p[i, :, 1]))

Q_pp_D = np.zeros([N_gl, 1])
for l in range(N_b):
    for k in range(6):
        m = int(a[k, l])
        f = Q_p_D[l, :, 0]
        Q_pp_D[m, 0] = f[k] + Q_pp_D[m, 0]

# vector de cargas nodales concentradas:
Q_L = np.zeros((N_gl, 1))
for i in range(N_cp):
    Q_L[int(Mgl[cargas_concentradas["Nudo"][i], 0])
        ] = cargas_concentradas["Fx"][i]*Υ*χ

Q_G = np.zeros((N_gl, 1))
for i in range(N_cp):
    Q_G[int(Mgl[cargas_concentradas["Nudo"][i], 1])
        ] = cargas_concentradas["Fy"][i]*Υ*χ
    Q_G[int(Mgl[cargas_concentradas["Nudo"][i], 2])
        ] = cargas_concentradas["Fy"][i]*Υ*χ*φ

# vector de cargas para todas las cargas existentes:
Q_TOTAL = Q_L + Q_G + Q_pp_D
# Vectores de cargas reducidos:
Qx_TOTAL = np.zeros((N_gl-N_r, 1))
for i in range(N_gl-N_r):
    Qx_TOTAL[i, 0] = Q_TOTAL[i, 0]

Qr_TOTAL = np.zeros((N_r, 1))
for i in range(N_r):
    Qr_TOTAL[i, 0] = Q_TOTAL[N_gl-N_r + i, 0]


# D) OBTENSION DE RESULTADOS, SOLUCIÓN DEL SISTEMA Q = K*D:
# Desplazaientos desconocidos
u_xTO = np.dot(np.linalg.inv(K_xx), Qx_TOTAL)

# E) Calculo de reacciones
f_rTO = np.dot(K_rx, u_xTO)
Reacciones_TO = f_rTO - Qr_TOTAL

Reato = np.zeros((N_gl, 1))
for k in range(N_gl-N_r, N_gl):
    Reato[k, 0] = Reacciones_TO[k - N_gl + N_r, 0]

# F) Calculo de fuerzas internas
# Matriz completa de desplazamientos
u_toTO = np.zeros((N_gl, 1))
for k in range(N_gl-N_r):
    u_toTO[k, 0] = u_xTO[k, 0]

U_TTO = u_toTO

# Desplazamientos en cada barra en sistema global
u_glTO = np.zeros((N_b, 6, 1))
for i in range(N_b):
    for j in range(6):
        k = int(a[j, i])
        u_glTO[i, j, 0] = U_TTO[k, 0]

# Desplazamientos en cada barra en sistema local
u_localTO = np.zeros((N_b, 6, 1))
for i in range(N_b):
    u_localTO[i] = np.dot(H[i], np.dot(T[i], u_glTO[i]))

# Vector de cargas en cada barra en sistema local
q_localTO = np.zeros((N_b, 6, 1))
for i in range(N_b):
    q_localTO[i] = np.dot(K_et[i], u_localTO[i]) - q[i]

q_p_localTO = np.zeros((N_b, 6, 1))
for i in range(N_b):
    q_p_localTO[i] = np.dot(np.linalg.inv(Unidades), q_localTO[i])

# Matriz de unidades para desplazamientos:
rad = 1
Unidades2 = np.array([

    [χ, χ*0, rad*0, χ*0, χ*0, rad*0],
    [χ*0, χ, rad*0, χ*0, χ*0, rad*0],
    [χ*0, χ*0, rad, χ*0, χ*0, rad*0],
    [χ*0, χ*0, rad*0, χ, χ*0, rad*0],
    [χ*0, χ*0, rad*0, χ*0, χ, rad*0],
    [χ*0, χ*0, rad*0, χ*0, χ*0, rad]

])

u_p_localTO = np.zeros((N_b, 6, 1))
for i in range(N_b):
    u_p_localTO[i] = np.dot(np.linalg.inv(Unidades2), u_localTO[i])

# G) OBTENSION DE ECUACIONES DE MOMENTO, CORTANTE, NORMAL, FLECHAS, GISROS, ETC.
# obtension de coeficientes de carga
B_p = np.zeros((N_b, 1))
for i in range(N_b):
    for j in range(N_cd):
        if cargas_distribuidas["Barra"][j] == i:
            if cargas_distribuidas["g"][j] == 1:
                γ = 1
                ξ = np.cos(theta_x[i])
            else:
                γ = 0
                ξ = 1
            long = L[i]*χ
            qa = cargas_distribuidas["qa"][j]*Υ*ξ
            qb = cargas_distribuidas["qb"][j]*Υ*ξ
            q11 = qa
            q22 = qb
            B_p[i] = q11

A_p = np.zeros((N_b, 1))
for i in range(N_b):
    for j in range(N_cd):
        if cargas_distribuidas["Barra"][j] == i:
            if cargas_distribuidas["g"][j] == 1:
                γ = 1
                ξ = np.cos(theta_x[i])
            else:
                γ = 0
                ξ = 1
            long = L[i]*χ
            qa = cargas_distribuidas["qa"][j]*Υ*ξ
            qb = cargas_distribuidas["qb"][j]*Υ*ξ
            q11 = qa
            q22 = qb
            A_p[i] = (q22 - q11)/long

EI = np.zeros(N_b)
for i in range(N_b):
    EI[i] = E*I[i]*Υ*φ*χ

L_n = np.zeros(N_b)
for i in range(N_b):
    L_n[i] = L[i]*χ

Unidades5 = np.array([

    [Υ*χ, Υ*χ**2*0, Υ*χ**3*0, Υ*χ**4*0],
    [Υ*χ**2*0, Υ*χ, Υ*χ**2*0, Υ*χ**3*0],
    [Υ*χ**3*0, Υ*χ**2*0, Υ*χ, Υ*χ**2*0],
    [Υ*χ**4*0, Υ*χ**3*0, Υ*χ**2*0, Υ*χ]

])

# Matrices para obtension de coeficientes de momento, cortante...
# Utilizando valores de frontera desplazamientos conocidos:
mx = np.zeros((N_b, 4, 4))
for i in range(N_b):
    mx[i] = np.array([

        [0,  0,  0,  1],
        [0,  0,  1,  0],
        [L_n[i]**3*(2-β_y[i])/12, L_n[i]**2/2, L_n[i], 1],
        [L_n[i]**2/2, L_n[i], 1, 0]

    ])

C_p_TO = np.zeros((N_b, 4, 1))
for i in range(N_b):
    C_p_TO[i] = np.array([

        [EI[i]*u_localTO[i][1][0]],
        [EI[i]*u_localTO[i][2][0]],
        [EI[i]*u_localTO[i][4][0] + A_p[i, 0]*L_n[i]**5 *
            (0.6 - β_y[i])/72 + B_p[i, 0]*L_n[i]**4*(1 - β_y[i])/24],
        [EI[i]*u_localTO[i][5][0] + A_p[i, 0]*L_n[i]**4/24 + B_p[i, 0]*L_n[i]**3/6]

    ])

C_TO = np.zeros((N_b, 4, 1))
for i in range(N_b):
    C_TO[i] = np.dot(np.linalg.inv(mx[i]), C_p_TO[i])

for i in range(N_b):
    C_TO[i] = np.dot(np.linalg.inv(Unidades5), C_TO[i])

# Asignacion de unidades a los cueficientes:
A_D = np.zeros((N_b, 1))
B_D = np.zeros((N_b, 1))
A_L = np.zeros((N_b, 1))
B_L = np.zeros((N_b, 1))
for i in range(N_b):
    A_D[i] = A_p[i]*χ/Υ
    B_D[i] = B_p[i]*1/Υ
    A_L[i] = A_p[i]*χ/Υ
    B_L[i] = B_p[i]*1/Υ

# Factor de escala de dibujo
Ψ_p = 0.03
# numero de puntos para graficar
nnp = 40

# H) FUERZAS AXIALES EN TODOS LOS ELEMENTOS # DUDA INDICES
X_pp_c = np.zeros((nnp+2, 2*N_b))
A = np.zeros(N_b)
B = np.zeros(N_b)
x_p = nnp - 1
xc = np.zeros(x_p+3)
x_p_c = np.zeros(x_p+3)
y_p_c = np.zeros(x_p+3)

for i in range(N_b):
    Ω = longitud
    Ω_p = fuerza
    n = L[i]/x_p
    A[i] = (q_p_localTO[i][3, 0] + q_p_localTO[i][0, 0])/L[i]
    B[i] = -q_p_localTO[i][0, 0]
    def N(x): return A[i]*x + B[i]
    for j in range(1, x_p+2):
        xc[j] = n*(j-1)
        x_p_c[j] = xc[j]*np.cos(theta_x[i])/Ω - \
            N(xc[j])*Ψ_p/Ω_p*np.sin(theta_x[i]) + \
            X_c[0][i]/Ω
        y_p_c[j] = xc[j]*np.sin(theta_x[i])/Ω + \
            N(xc[j])*Ψ_p/Ω_p*np.cos(theta_x[i]) + \
            Y_c[0][i]/Ω
    x_p_c[0] = X_c[0][i]/Ω
    x_p_c[x_p+2] = X_c[1][i]/Ω
    y_p_c[0] = Y_c[0][i]/Ω
    y_p_c[x_p+2] = Y_c[1][i]/Ω
    X_pp_c[:, i] = x_p_c
    X_pp_c[:, i+N_b] = y_p_c

X_ppp_c = np.zeros((nnp+2, 2*N_b))
A = np.zeros(N_b)
B = np.zeros(N_b)
x_p = nnp - 1
xc = np.zeros(x_p+3)
x_p_c = np.zeros(x_p+3)
y_p_c = np.zeros(x_p+3)

for i in range(N_b):
    Ω = longitud
    Ω_p = fuerza
    n = L[i]/x_p
    A[i] = (q_p_localTO[i][3, 0] + q_p_localTO[i][0, 0])/L[i]
    B[i] = -q_p_localTO[i][0, 0]
    def N(x): return A[i]*x + B[i]
    for j in range(1, x_p+2):
        xc[j] = n*(j-1)
        x_p_c[j] = xc[j]*np.cos(theta_x[i])/Ω + \
            X_c[0][i]/Ω
        y_p_c[j] = xc[j]*np.sin(theta_x[i])/Ω + \
            Y_c[0][i]/Ω
    x_p_c[0] = X_c[0][i]/Ω
    x_p_c[x_p+2] = X_c[1][i]/Ω
    y_p_c[0] = Y_c[0][i]/Ω
    y_p_c[x_p+2] = Y_c[1][i]/Ω
    X_ppp_c[:, i] = x_p_c
    X_ppp_c[:, i+N_b] = y_p_c


xi = np.zeros((1, N_b*nnp))
for i in range(N_b):
    for j in range(nnp):
        xi[0, (i-1)*nnp+j] = X_pp_c[j, i]

x_p_i = np.zeros((1, N_b*nnp))
for i in range(N_b):
    for j in range(nnp):
        x_p_i[0, (i-1)*nnp+j] = X_ppp_c[j, i]

yi = np.zeros((1, N_b*nnp))
for i in range(N_b, 2*N_b):
    for j in range(nnp):
        yi[0, (i-N_b-1)*nnp+j] = X_pp_c[j, i]

y_p_i = np.zeros((1, N_b*nnp))
for i in range(N_b, 2*N_b):
    for j in range(nnp):
        y_p_i[0, (i-N_b-1)*nnp+j] = X_ppp_c[j, i]

xif = np.stack((xi[0], x_p_i[0]), axis=0)
yif = np.stack((yi[0], y_p_i[0]), axis=0)

X_F = X_pp_c[0:, :N_b]
Y_F = X_pp_c[0:, N_b:2*N_b]
X_p_F = X_ppp_c[0:, :N_b]
Y_p_F = X_ppp_c[0:, N_b:2*N_b]

# # PLOTEO DEL DIAGRAMA DE FUERZAS AXIALES
plt.figure(figsize=(9, 9))
plt.plot(X_c, Y_c, color='b')
plt.plot(X_F, Y_F, color='r', linestyle='dashed', linewidth=1, alpha=0.5)
# plt.plot(xif, yif, color='#fff7e3', linewidth=2, alpha=0.1)
plt.fill(X_F, Y_F, color='#fff7e3', alpha=0.3)
plt.plot(X_p_F, Y_p_F, color='b')
plt.axis('equal')
plt.tight_layout()
plt.show()

# I) FUERZAS CORTANTES EN TODOS LOS ELEMENTOS:
X_pp_c = np.zeros((nnp+2, 2*N_b))
x_p = nnp - 1
xc = np.zeros(x_p+3)
x_p_c = np.zeros(x_p+3)
y_p_c = np.zeros(x_p+3)

for i in range(N_b):
    Ω = longitud
    Ω_p = fuerza
    n = L[i]/x_p
    def V(x): return -A_D[i, 0]*x**2/2 - B_D[i, 0]*x + q_p_localTO[i][1, 0]
    for j in range(1, x_p+2):
        xc[j] = n*(j-1)
        x_p_c[j] = xc[j]*np.cos(theta_x[i])/Ω - \
            V(xc[j])*Ψ_p/Ω_p*np.sin(theta_x[i]) + \
            X_c[0][i]/Ω
        y_p_c[j] = xc[j]*np.sin(theta_x[i])/Ω + \
            V(xc[j])*Ψ_p/Ω_p*np.cos(theta_x[i]) + \
            Y_c[0][i]/Ω
    x_p_c[0] = X_c[0][i]/Ω
    x_p_c[x_p+2] = X_c[1][i]/Ω
    y_p_c[0] = Y_c[0][i]/Ω
    y_p_c[x_p+2] = Y_c[1][i]/Ω
    X_pp_c[:, i] = x_p_c
    X_pp_c[:, i+N_b] = y_p_c


xi = np.zeros((1, N_b*nnp))
for i in range(N_b):
    for j in range(nnp):
        xi[0, (i-1)*nnp+j] = X_pp_c[j, i]

yi = np.zeros((1, N_b*nnp))
for i in range(N_b, 2*N_b):
    for j in range(nnp):
        yi[0, (i-N_b-1)*nnp+j] = X_pp_c[j, i]


xif = np.stack((xi[0], x_p_i[0]), axis=0)
yif = np.stack((yi[0], y_p_i[0]), axis=0)

X_F = X_pp_c[0:, :N_b]
Y_F = X_pp_c[0:, N_b:2*N_b]

# df = pd.DataFrame(X_F)
# df1 = pd.DataFrame(Y_F)
# df.to_excel("Matriz1.xlsx", index=False, header=False)
# df1.to_excel("Matriz2.xlsx", index=False, header=False)

# PLOTEO DEL DIAGRAMA DE FUERZAS CORTANTES
plt.figure(figsize=(9, 9))
plt.plot(X_c, Y_c, color='b')
plt.plot(X_F, Y_F, color='r', linestyle='dashed', linewidth=1, alpha=0.5)
plt.fill(X_F, Y_F, color='#fff7e3', alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()


# J) MOMENTOS FLECTORES EN TODOS LOS ELEMENTOS:
X_pp_c = np.zeros((nnp+2, 2*N_b))
x_p = nnp - 1
xc = np.zeros(x_p+3)
x_p_c = np.zeros(x_p+3)
y_p_c = np.zeros(x_p+3)

for i in range(N_b):
    Ω = longitud
    Ω_p = fuerza
    n = L[i]/x_p

    def M(x): return -(-A_D[i, 0]*x**3/6 - B_D[i, 0]*x**2 /
                       2 + q_p_localTO[i][1, 0]*x - q_p_localTO[i][2, 0])
    for j in range(1, x_p+2):
        xc[j] = n*(j-1)
        x_p_c[j] = xc[j]*np.cos(theta_x[i])/Ω - \
            M(xc[j])*Ψ_p/(Ω*Ω_p)*np.sin(theta_x[i]) + \
            X_c[0][i]/Ω
        y_p_c[j] = xc[j]*np.sin(theta_x[i])/Ω + \
            M(xc[j])*Ψ_p/(Ω*Ω_p)*np.cos(theta_x[i]) + \
            Y_c[0][i]/Ω
    x_p_c[0] = X_c[0][i]/Ω
    x_p_c[x_p+2] = X_c[1][i]/Ω
    y_p_c[0] = Y_c[0][i]/Ω
    y_p_c[x_p+2] = Y_c[1][i]/Ω
    X_pp_c[:, i] = x_p_c
    X_pp_c[:, i+N_b] = y_p_c

xi = np.zeros((1, N_b*nnp))
for i in range(N_b):
    for j in range(nnp):
        xi[0, (i-1)*nnp+j] = X_pp_c[j, i]

yi = np.zeros((1, N_b*nnp))
for i in range(N_b, 2*N_b):
    for j in range(nnp):
        yi[0, (i-N_b-1)*nnp+j] = X_pp_c[j, i]

xif = np.stack((xi[0], x_p_i[0]), axis=0)
yif = np.stack((yi[0], y_p_i[0]), axis=0)

X_F = X_pp_c[0:, :N_b]
Y_F = X_pp_c[0:, N_b:2*N_b]


# PLOTEO DEL DIAGRAMA DE MOMENTOS FLECTORES
plt.figure(figsize=(9, 9))
plt.plot(X_c, Y_c, color='b')
plt.plot(X_F, Y_F, color='r', linestyle='dashed', linewidth=1, alpha=0.5)
plt.fill(X_F, Y_F, color='#fff7e3', alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()


# DESPLAZAMIENTOS:
# Factor de escala de dibujo
Ψ = 40
# numero de puntos para graficar
nnp = 30

X_pp_c = np.zeros((nnp, 2*N_b))
x_p = nnp - 1
xc = np.zeros(nnp)
x_p_c = np.zeros(nnp)
y_p_c = np.zeros(nnp)
L_p = np.zeros(N_b)

for i in range(N_b):
    Ω = longitud
    Ω_p = fuerza
    L_p[i] = L[i] + (u_localTO[i][3][0] - u_localTO[i][0][0])*Ω*0
    n = L_p[i]/x_p
    Nodo = barras["inicio"][i]
    xx = int(Mgl[Nodo, 0])
    yy = int(Mgl[Nodo, 1])
    Dnu = np.array([
        [u_toTO[xx][0]],
        [u_toTO[yy][0]]
    ])

    def y(x): return (-A_D[i, 0]*L_p[i]**2*x**3/72*(0.6*(x/L_p[i])**2-β_y[i]) -
                      B_D[i, 0]*L_p[i]**2*x**2/24*((x/L_p[i])**2-β_y[i]) +
                      C_TO[i][0, 0]*x*L_p[i]**2/12*(2*(x/L_p[i])**2-β_y[i]) +
                      C_TO[i][1, 0]*x**2/2 + C_TO[i][2, 0]*x +
                      C_TO[i][3, 0])/EI[i]

    for j in range(x_p+1):
        xc[j] = n*(j)

        x_p_c[j] = xc[j]*np.cos(theta_x[i])/Ω - \
            y(xc[j])*Ψ/(Ω)*np.sin(theta_x[i]) + \
            (X_c[0][i] + Dnu[0][0]*Ω*Ψ*np.cos(theta_x[i]))/Ω

        y_p_c[j] = xc[j]*np.sin(theta_x[i])/Ω + \
            y(xc[j])*Ψ/(Ω)*np.cos(theta_x[i]) + \
            (Y_c[0][i] + Dnu[1][0]*Ω*Ψ*np.sin(theta_x[i]))/Ω

    X_pp_c[:, i] = x_p_c
    X_pp_c[:, i+N_b] = y_p_c

X_F = X_pp_c[0:, :N_b]
Y_F = X_pp_c[0:, N_b:2*N_b]

# PLOTEO DEL DIAGRAMA DE DESPLAZAMIENTOS
plt.figure(figsize=(9, 9))
plt.plot(X_F, Y_F, color='b')
plt.plot(X_c, Y_c, color='#8d8d8d', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axis('equal')
plt.tight_layout()
plt.show()

# K) DESPLAZAMIENTOS Y REACCIONES EN UN PUNTO:
Apo = 4
Nodo = 1

Fapo = np.zeros((3, 1))
for i in range(3):
    Fapo[i] = Reato[int(Mgl[Apo, i])]

Dnu = np.zeros((3, 1))
for i in range(3):
    Dnu[i] = u_toTO[int(Mgl[Nodo, i])]

# L) FUERZAS INTERNAS EN UN ELEMENTO ESPECIFICO:
Elem = 8

x = np.linspace(0, L[Elem], 100)
A_axialTO = (q_p_localTO[Elem][3, 0] + q_p_localTO[Elem][0, 0])/(L[Elem])
B_axialTO = -q_p_localTO[Elem][0, 0]
def N(x): return A_axialTO*x + B_axialTO


def V(x): return -A_D[Elem, 0]*x**2/2 - \
    B_D[Elem, 0]*x + q_p_localTO[Elem][1, 0]


def M(x): return (-A_D[Elem, 0]*x**3/6 - B_D[Elem, 0]*x**2 /
                  2 + q_p_localTO[Elem][1, 0]*x - q_p_localTO[Elem][2, 0])


def θ(x): return (-A_D[Elem, 0]*x**4/24 - B_D[Elem, 0]*x**3/6 + C_TO[Elem]
                  [0, 0]*x**2/2 + C_TO[Elem][1, 0]*x + C_TO[Elem][2, 0])/EI[Elem]


def y(x): return (-A_D[Elem, 0]*L[Elem]**2*x**3/72*(0.6*(x/L[Elem])**2-β_y[Elem]) -
                  B_D[Elem, 0]*L[Elem]**2*x**2/24*((x/L[Elem])**2-β_y[Elem]) +
                  C_TO[Elem][0, 0]*x*L[Elem]**2/12*(2*(x/L[Elem])**2-β_y[Elem]) +
                  C_TO[Elem][1, 0]*x**2/2 + C_TO[Elem][2, 0]*x +
                  C_TO[Elem][3, 0])/EI[Elem]



# mostrar en una widget
diagrams = {
    'N(x)': DiagramConfig(
        name='Diagrama de Fuerza Normal',
        values=N(x),
        units='tonf',
    ),
    'V(x)': DiagramConfig(
        name='Diagrama de Fuerza Cortante',
        values=V(x),
        units='tonf',
    ),
    'M(x)': DiagramConfig(
        name='Diagrama de Momento Flector',
        values=M(x),
        units='tonf-m',

    ),
'θ(x)': DiagramConfig(
        name='Diagrama de Rotación',
        values=θ(x),
        units='rad',
        precision=5

    ),
    'y(x)': DiagramConfig(
        name='Diagrama de Deflexión',
        values=y(x),
        units='cm',
        precision=5

    )
}

app = InternalForceDiagramWidget(Elem, x, diagrams, grafigcalor=True, cmap='rainbow')