# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.path import Path


# # ==================================================
# # =============     FancyArrowPatch    =============
# # ==================================================

# from matplotlib.patches import FancyArrowPatch
# import matplotlib.pyplot as plt

# # Crear una figura y un eje
# fig, ax = plt.subplots()

# # Plantilla de FancyArrowPatch con todos los parámetros
# arrow = FancyArrowPatch(
#     posA=(0.2, 0.2),  # Coordenadas de inicio (x, y) de la flecha.
#     posB=(0.8, 0.8),  # Coordenadas de fin (x, y) de la flecha.
#     path=None,  # Ruta personalizada (opcional). Si es None, se genera automáticamente.
#     arrowstyle="simple",  # Estilo de la flecha. Opciones: "->", "<-", "<->", "|-|", "fancy", "simple", etc.
#     mutation_scale=15,  # Escala del tamaño de la cabeza de la flecha (en píxeles).
#     # color="blue",  # Color de la flecha. Puede ser un nombre ("red") o una tupla RGB/RGBA.
#     edgecolor='black',  # Color del borde de la flecha
#     facecolor='blue',  # Color de relleno de la cabeza de la flecha
#     linewidth=1.5,  # Grosor de la línea de la flecha.
#     linestyle="-",  # Estilo de la línea. Opciones: "-", "--", ":", "-.".
#     transform=ax.transData,  # Sistema de coordenadas. Opciones: transData, transAxes, etc.
#     patchA=None,  # Objeto Patch en el extremo A (opcional).
#     patchB=None,  # Objeto Patch en el extremo B (opcional).
#     shrinkA=0.0,  # Espacio para reducir la flecha en el extremo A (fracción).
#     shrinkB=0.0,  # Espacio para reducir la flecha en el extremo B (fracción).
#     connectionstyle="arc3,rad=0.2",  # Estilo de conexión. Opciones: "arc3", "angle3", "bar", etc.
#     mutation_aspect=1.0,  # Relación de aspecto de la cabeza de la flecha.
#     clip_on=True,  # Si la flecha debe ser recortada por los límites del gráfico
#     joinstyle="round",  # Estilo de unión de las líneas. Opciones: "miter", "round", "bevel".
#     capstyle="butt",  # Estilo de los extremos de la línea. Opciones: "butt", "round", "projecting".
#     alpha=1.0,  # Transparencia de la flecha (0 = transparente, 1 = opaco).
#     zorder=1,  # Orden de apilamiento (controla qué objetos se dibujan encima de otros).
# )

# # Añadir la flecha al eje
# ax.add_patch(arrow)

# # Configurar límites del eje
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)

# # Mostrar la figura
# plt.show()




# # ==================================================
# # =============     ConnectionPatch    =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# # Crear un ConnectionPatch con todos los parámetros
# conn = patches.ConnectionPatch(
#     xyA=(0.8, 0.2),  # Punto en el eje A (ax1)
#     xyB=(0.2, 0.8),  # Punto en el eje B (ax2)

#     coordsA="data",  # Sistema de coordenadas de A ('data', 'axes fraction', etc.)
#     coordsB="data",  # Sistema de coordenadas de B

#     arrowstyle='-|>',  # Estilo de flecha (ver lista abajo)
#     connectionstyle='arc3,rad=0.2',  # Estilo de conexión entre A y B
#     mutation_scale=15,  # Escala de la cabeza de la flecha

#     linestyle='solid',  # Tipo de línea: 'solid', 'dashed', 'dashdot', 'dotted'
#     linewidth=2,  # Grosor de la línea
#     edgecolor='black',  # Color del borde de la línea
#     facecolor='red',  # Color de relleno (si aplica)
#     alpha=0.8,  # Transparencia (0=transparente, 1=opaco)

#     shrinkA=5,  # Reduce la línea en el extremo A
#     shrinkB=5,  # Reduce la línea en el extremo B

#     zorder=3,  # Controla el orden de dibujo (mayor zorder = encima)
#     clip_on=True,  # Si la línea se debe recortar según los límites del gráfico
#     transform=None,  # Transformación de coordenadas (None usa la predeterminada)
# )

# # Agregar el ConnectionPatch al gráfico
# ax2.add_patch(conn)

# # Configurar los ejes
# ax1.set_xlim(0, 1)
# ax1.set_ylim(0, 1)
# ax1.set_title("Eje A")

# ax2.set_xlim(0, 1)
# ax2.set_ylim(0, 1)
# ax2.set_title("Eje B")

# # Mostrar la figura
# plt.show()











# # ==================================================
# # =============     FancyBboxPatch     =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear un FancyBboxPatch con todos los parámetros
# rect = patches.FancyBboxPatch(
#     (0.2, 0.2),  # (x, y) - Coordenadas de la esquina inferior izquierda
#     width=0.5,  # Ancho del rectángulo
#     height=0.3,  # Altura del rectángulo
    
#     boxstyle="round,pad=0.1",  # Estilo del borde (ver lista abajo)
    
#     linewidth=2,  # Grosor del borde
#     edgecolor="black",  # Color del borde
#     facecolor="lightblue",  # Color de relleno
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = transparente)

#     linestyle="dashed",  # Estilo de línea ('solid', 'dashed', 'dotted', etc.)
#     capstyle="round",  # Estilo de terminación ('butt', 'round', 'projecting')
#     joinstyle="round",  # Estilo de unión en las esquinas ('miter', 'round', 'bevel')

#     mutation_scale=20,  # Escala del estilo de borde
#     mutation_aspect=1,  # Relación de aspecto del borde (1 = sin cambios)
#     snap=True,  # Ajuste a la cuadrícula de píxeles (puede mejorar el renderizado)

#     clip_on=True,  # Si se recorta el rectángulo según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar el rectángulo al gráfico
# ax.add_patch(rect)

# # Configurar límites y mostrar
# ax.set_xlim(-5, 5)
# ax.set_ylim(-2, 3)
# ax.set_title("Ejemplo de FancyBboxPatch")
# plt.show()








# # ==================================================
# # =============         Path           =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.path import Path

# # Definir una función para crear una figura con Path
# def draw_custom_shape():
#     fig, ax = plt.subplots()

#     # Definir los vértices y códigos para un Path personalizado (Ejemplo: un pentágono)
#     vertices = [
#         (0, 0),  # Punto inicial
#         (1, 2),  # Línea al siguiente punto
#         (2, 3),
#         (3, 2),
#         (4, 0),
#         (0, 0)  # Cierre de la figura
#     ]
    
#     codes = [
#         Path.MOVETO,  # Mueve el lápiz al punto inicial
#         Path.LINETO,  # Dibuja una línea
#         Path.LINETO,
#         Path.LINETO,
#         Path.LINETO,
#         Path.CLOSEPOLY  # Cierra la forma
#     ]
    
#     # Crear el Path
#     path = Path(vertices, codes)

#     # Crear un parche con el Path
#     patch = patches.PathPatch(path, facecolor='lightblue', edgecolor='black', lw=2)

#     # Agregar el parche al gráfico
#     ax.add_patch(patch)

#     # Configuración del gráfico
#     ax.set_xlim(-1, 5)
#     ax.set_ylim(-1, 4)
#     ax.set_aspect('equal')  # Mantiene las proporciones
#     ax.grid(True, linestyle="--", alpha=0.5)  # Agregar una cuadrícula

#     plt.show()

# # Llamar a la función para dibujar la figura
# draw_custom_shape()






# # ==================================================
# # =============     ArrowStyle         =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))


# # arroystyle_1 = patches.ArrowStyle.Simple(
# #     head_length=0.4,  # Longitud de la cabeza de la flecha (fracción del ancho total).
# #     head_width=0.4,   # Ancho de la cabeza de la flecha (fracción del ancho total).
# #     tail_width=0.4    # Ancho de la cola de la flecha (fracción del ancho total).
# # )

# # arroystyle_1 = patches.ArrowStyle.Fancy(
# #     head_length=0.4,  # Longitud de la cabeza de la flecha.
# #     head_width=0.4,   # Ancho de la cabeza de la flecha.
# #     tail_width=0.4    # Ancho de la cola de la flecha.
# # )

# arroystyle_1 = patches.ArrowStyle.Wedge(
#     tail_width=0.3,  # Ancho de la base de la flecha.
#     shrink_factor=0.5  # Factor de reducción de la cabeza de la flecha.
# )

# # arroystyle_1 = patches.ArrowStyle.BarAB(
# #     widthA=0.5,  # Ancho de la barra en el punto A.
# #     widthB=0.5   # Ancho de la barra en el punto B.
# # )

# # arroystyle_1 = patches.ArrowStyle.CurveAB()

# # arroystyle_1 = patches.ArrowStyle.CurveB()

# # arroystyle_1 = patches.ArrowStyle.CurveFilledB()

# # arroystyle_1 = patches.ArrowStyle.BracketAB(
# #     widthA=1.0,  # Ancho del soporte en el punto A.
# #     widthB=1.0,  # Ancho del soporte en el punto B.
# #     angleA=90,   # Ángulo de la barra en el punto A.
# #     angleB=90    # Ángulo de la barra en el punto B.
# # )

# # arroystyle_1 = patches.ArrowStyle.BracketA(
# #     widthA=1.0,  # Ancho del soporte en el punto A.
# #     angleA=90    # Ángulo de la barra en el punto A.
# # )

# # arroystyle_1 = patches.ArrowStyle.BracketB(
# #     widthB=1.0,  # Ancho del soporte en el punto B.
# #     angleB=90    # Ángulo de la barra en el punto B.
# # )


# # Crear una flecha con un estilo personalizado usando ArrowStyle
# arrow = patches.FancyArrowPatch(
#     (0.2, 0.2),  # Punto de inicio (x1, y1)
#     (0.8, 0.8),  # Punto final (x2, y2)
    
#     # -------- Estilo de la flecha --------
#     arrowstyle=arroystyle_1,
#     # arrowstyle define la forma de la flecha. Algunos valores comunes:
#     # - '->'      : Flecha simple
#     # - '<-'      : Flecha simple invertida
#     # - '<->'     : Flecha doble
#     # - 'Fancy'   : Flecha con cabecera decorativa
#     # - 'Simple'  : Flecha simple con ajuste de longitud
#     # - 'Wedge'   : Flecha con base ancha
#     # - 'Bar'     : Línea con terminaciones en barra

#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde
#     edgecolor="black",  # Color del borde

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="dashed",  # Estilo de la línea ('solid', 'dashed', 'dotted', etc.)

#     # -------- Control del renderizado --------
#     mutation_scale=20,  # Escala del estilo de la flecha (tamaño relativo)
#     mutation_aspect=1,  # Relación de aspecto de la flecha (1 = sin cambios)
    
#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta la flecha según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar la flecha al gráfico
# ax.add_patch(arrow)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_title("Ejemplo de FancyArrowPatch con ArrowStyle")
# plt.show()





# # ==================================================
# # =============     ConnectionStyle         =============
# # ==================================================


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear una flecha con un estilo de conexión personalizado usando ConnectionStyle
# arrow = patches.FancyArrowPatch(
#     (0.2, 0.2),  # Punto de inicio (x1, y1)
#     (0.8, 0.8),  # Punto final (x2, y2)
    
#     # -------- Estilo de la conexión (curva entre los puntos) --------
#     connectionstyle=patches.ConnectionStyle("arc3", rad=0.2),
#     # connectionstyle define la forma en que la línea se curva entre los puntos.
#     # Algunos valores comunes:
#     # - "arc3"     : Arco de tercer grado (suave)
#     # - "arc"      : Arco circular
#     # - "angle"    : Conexión en ángulo
#     # - "angle3"   : Conexión en ángulo con ajuste suave
#     # - "bar"      : Línea con extremos en barra (parámetros: armA, armB, fraction)

#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde
#     edgecolor="black",  # Color del borde

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="dashed",  # Estilo de la línea ('solid', 'dashed', 'dotted', etc.)

#     # -------- Control del renderizado --------
#     mutation_scale=20,  # Escala del estilo de la flecha (tamaño relativo)
#     mutation_aspect=1,  # Relación de aspecto de la flecha (1 = sin cambios)
    
#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta la flecha según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar la flecha al gráfico
# ax.add_patch(arrow)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_title("Ejemplo de FancyArrowPatch con ConnectionStyle")
# plt.show()





# # ==================================================
# # =============     boxstyle         =============
# # ==================================================
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear un cuadro de texto con un estilo personalizado usando BoxStyle
# text = ax.text(
#     0.5, 0.5,  # Posición (x, y) en coordenadas del gráfico
#     "Texto con BoxStyle",  # Contenido del texto
    
#     # -------- Estilo del cuadro alrededor del texto --------
#     bbox=dict(
#         boxstyle=patches.BoxStyle("Round", pad=0.3),
#         # boxstyle define la forma del cuadro alrededor del texto. Opciones comunes:
#         # - "Square"    : Rectángulo normal
#         # - "Circle"    : Caja en forma de círculo
#         # - "Round"     : Rectángulo con esquinas redondeadas (parámetro: pad)
#         # - "Round4"    : Más redondeado que "Round"
#         # - "Sawtooth"  : Bordes con forma de sierra (parámetro: tooth_size)
#         # - "DArrow"    : Flecha doble alrededor del texto
#         # - "LArrow"    : Flecha simple apuntando a la izquierda
#         # - "RArrow"    : Flecha simple apuntando a la derecha

#         # -------- Personalización del cuadro --------
#         linewidth=2,      # Grosor del borde del cuadro
#         edgecolor="black",  # Color del borde
#         facecolor="lightblue",  # Color de relleno del cuadro
#         alpha=0.8,  # Transparencia (1 = sólido, 0 = transparente)

#         # -------- Estilo de línea --------
#         linestyle="dashed",  # Estilo del borde ('solid', 'dashed', 'dotted', etc.)
#     ),

#     fontsize=12,  # Tamaño del texto
#     ha="center",  # Alineación horizontal ('left', 'center', 'right')
#     va="center",  # Alineación vertical ('top', 'center', 'bottom')
# )

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_title("Ejemplo de BoxStyle en un cuadro de texto")
# plt.show()




# # ==================================================
# # =============     arc         =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear un arco con un estilo personalizado usando Arc
# arc = patches.Arc(
#     (0.5, 0.5),  # Centro del arco (x, y)
#     width=0.6,  # Ancho total del arco (diámetro horizontal)
#     height=0.4,  # Altura total del arco (diámetro vertical)
    
#     angle=0,  # Rotación del arco en grados (en sentido antihorario)
#     theta1=0,  # Ángulo inicial del arco (en grados)
#     theta2=270,  # Ángulo final del arco (en grados)

#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor de la línea del arco
#     edgecolor="black",  # Color de la línea del arco
#     linestyle="dashed",  # Estilo de línea ('solid', 'dashed', 'dotted', etc.)
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Control del renderizado --------
#     capstyle="round",  # Estilo de terminación ('butt', 'round', 'projecting')
#     joinstyle="round",  # Estilo de unión en esquinas ('miter', 'round', 'bevel')

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta el arco según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar el arco al gráfico
# ax.add_patch(arc)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_title("Ejemplo de Arc en Matplotlib")
# plt.show()

# # ==================================================
# # =============     circle         =============
# # ==================================================
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear un círculo con un estilo personalizado usando Circle
# circle = patches.Circle(
#     (0.5, 0.5),  # Centro del círculo (x, y)
#     radius=0.3,  # Radio del círculo

#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde
#     edgecolor="black",  # Color del borde

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="dashed",  # Estilo del borde ('solid', 'dashed', 'dotted', etc.)

#     # -------- Control del renderizado --------
#     capstyle="round",  # Estilo de terminación ('butt', 'round', 'projecting')
#     joinstyle="round",  # Estilo de unión en esquinas ('miter', 'round', 'bevel')

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta el círculo según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar el círculo al gráfico
# ax.add_patch(circle)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1 para que el círculo no se deforme
# ax.set_title("Ejemplo de Circle en Matplotlib")
# plt.show()


# # ==================================================
# # =============     annulus         =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear un anillo (annulus) usando Wedge con theta1=0 y theta2=360
# annulus = patches.Wedge(
#     (0.5, 0.5),  # Centro del anillo (x, y)
#     r=0.3,  # Radio externo
#     theta1=0,  # Ángulo inicial (0°)
#     theta2=360,  # Ángulo final (360°)
    
#     width=0.1,  # Grosor del anillo (r_externo - r_interno)
    
#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde del anillo
#     edgecolor="black",  # Color del borde

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno del anillo
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="dashed",  # Estilo del borde ('solid', 'dashed', 'dotted', etc.)

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta el anillo según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar el anillo al gráfico
# ax.add_patch(annulus)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1 para que el anillo no se deforme
# ax.set_title("Ejemplo de Annulus en Matplotlib")
# plt.show()


# # ==================================================
# # =============     Ellipse         =============
# # ==================================================


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear una elipse con un estilo personalizado usando Ellipse
# ellipse = patches.Ellipse(
#     (0.5, 0.5),  # Centro de la elipse (x, y)
#     width=0.6,  # Ancho total (diámetro horizontal)
#     height=0.4,  # Altura total (diámetro vertical)
    
#     angle=30,  # Rotación de la elipse en grados (sentido antihorario)

#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde
#     edgecolor="black",  # Color del borde

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="dashed",  # Estilo del borde ('solid', 'dashed', 'dotted', etc.)

#     # -------- Control del renderizado --------
#     capstyle="round",  # Estilo de terminación ('butt', 'round', 'projecting')
#     joinstyle="round",  # Estilo de unión en esquinas ('miter', 'round', 'bevel')

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta la elipse según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar la elipse al gráfico
# ax.add_patch(ellipse)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1 para que la elipse no se deforme
# ax.set_title("Ejemplo de Ellipse en Matplotlib")
# plt.show()




# # ==================================================
# # =============     CirclePolygon         =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear un polígono circular usando CirclePolygon
# circle_polygon = patches.CirclePolygon(
#     (0.5, 0.5),  # Centro del polígono (x, y)
#     radius=0.3,  # Radio del polígono circular
#     resolution=6,  # Número de lados (3 = triángulo, 4 = cuadrado, etc.)

#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde
#     edgecolor="black",  # Color del borde

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="dashed",  # Estilo del borde ('solid', 'dashed', 'dotted', etc.)

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta el polígono según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar el polígono circular al gráfico
# ax.add_patch(circle_polygon)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1 para que el polígono no se deforme
# ax.set_title("Ejemplo de CirclePolygon en Matplotlib")
# plt.show()


# # ==================================================
# # =============     FancyArrow         =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear una flecha usando FancyArrow
# arrow = patches.FancyArrow(
#     x=0.2,  # Coordenada x de inicio
#     y=0.2,  # Coordenada y de inicio
#     dx=0.5,  # Desplazamiento en x (longitud de la flecha)
#     dy=0.3,  # Desplazamiento en y (altura de la flecha)

#     width=0.05,  # Ancho de la base de la flecha
#     head_width=0.1,  # Ancho de la cabeza de la flecha
#     head_length=0.15,  # Longitud de la cabeza de la flecha

#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde de la flecha
#     edgecolor="black",  # Color del borde de la flecha

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno de la flecha
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="solid",  # Estilo del borde ('solid', 'dashed', 'dotted', etc.)

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta la flecha según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar la flecha al gráfico
# ax.add_patch(arrow)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1
# ax.set_title("Ejemplo de FancyArrow en Matplotlib")
# plt.show()



# # ==================================================
# # =============     Arrow         =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear una flecha usando Arrow
# arrow = patches.Arrow(
#     x=0.2,  # Coordenada x de inicio
#     y=0.2,  # Coordenada y de inicio
#     dx=0.5,  # Desplazamiento en x (longitud de la flecha)
#     dy=0.3,  # Desplazamiento en y (altura de la flecha)

#     width=0.1,  # Ancho total de la flecha

#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde de la flecha
#     edgecolor="black",  # Color del borde de la flecha

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno de la flecha
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta la flecha según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar la flecha al gráfico
# ax.add_patch(arrow)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1
# ax.set_title("Ejemplo de Arrow en Matplotlib")
# plt.show()




# # ==================================================
# # =============     Wedge         =============
# # ==================================================


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear un sector circular usando Wedge
# wedge = patches.Wedge(
#     center=(0.5, 0.5),  # Centro del sector (x, y)
#     r=0.3,  # Radio del sector
#     theta1=30,  # Ángulo inicial en grados (desde el eje positivo x)
#     theta2=150,  # Ángulo final en grados

#     width=0.1,  # Grosor del sector (si se omite, es un sector sólido)
    
#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde del sector
#     edgecolor="black",  # Color del borde del sector

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno del sector
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="solid",  # Estilo del borde ('solid', 'dashed', 'dotted', etc.)

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta el sector según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar el sector al gráfico
# ax.add_patch(wedge)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1
# ax.set_title("Ejemplo de Wedge en Matplotlib")
# plt.show()






# # ==================================================
# # =============       Polygon            =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Definir los vértices del polígono como una lista de tuplas (x, y)
# vertices = [(0.2, 0.2), (0.8, 0.2), (0.6, 0.6), (0.4, 0.8), (0.2, 0.6)]

# # Crear un polígono usando Polygon
# polygon = patches.Polygon(
#     vertices,  # Lista de coordenadas de los vértices
    
#     closed=True,  # Si el polígono debe cerrarse automáticamente (último punto conecta con el primero)
    
#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde del polígono
#     edgecolor="black",  # Color del borde del polígono

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno del polígono
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="solid",  # Estilo del borde ('solid', 'dashed', 'dotted', etc.)

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta el polígono según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar el polígono al gráfico
# ax.add_patch(polygon)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1 para evitar deformaciones
# ax.set_title("Ejemplo de Polygon en Matplotlib")
# plt.show()



# # ==================================================
# # =============       Path            =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.path as mpath

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Definir el Path con una serie de coordenadas y códigos de trazado
# Path = mpath.Path

# # Definir los vértices del camino
# vertices = [
#     (0.1, 0.1),  # Punto inicial
#     (0.4, 0.1),  # Línea hasta aquí
#     (0.4, 0.4),  # Línea hasta aquí
#     (0.6, 0.6),  # Línea hasta aquí
#     (0.9, 0.2),  # Línea hasta aquí
#     (0.1, 0.1)   # Cierre del camino
# ]

# # Definir los códigos que indican cómo se trazará el camino
# codes = [
#     Path.MOVETO,   # Moverse al primer punto sin trazar
#     Path.LINETO,   # Línea hasta el segundo punto
#     Path.LINETO,   # Línea hasta el tercer punto
#     Path.LINETO,   # Línea hasta el cuarto punto
#     Path.LINETO,   # Línea hasta el quinto punto
#     Path.CLOSEPOLY # Cerrar el camino volviendo al inicio
# ]

# # Crear el Path
# path = Path(vertices, codes)

# # Crear el PathPatch a partir del Path definido
# path_patch = patches.PathPatch(
#     path,  # Objeto Path

#     # -------- Personalización del contorno --------
#     linewidth=2,  # Grosor del borde del camino
#     edgecolor="black",  # Color del borde

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="solid",  # Estilo del borde ('solid', 'dashed', 'dotted', etc.)

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta según los límites del gráfico
#     zorder=3  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar el PathPatch al gráfico
# ax.add_patch(path_patch)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1
# ax.set_title("Ejemplo de PathPatch en Matplotlib")
# plt.show()











# # ==================================================
# # ==========    RegularPolygon         =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear un polígono regular (hexágono en este caso)
# polygon = patches.RegularPolygon(
#     (0.5, 0.5),  # Coordenadas del centro (x, y)
#     numVertices=6,  # Número de vértices (6 para un hexágono)
#     radius=0.3,  # Radio del polígono (distancia del centro a los vértices)
    
#     # -------- Personalización del contorno --------
#     linewidth=2,      # Grosor del borde del polígono
#     edgecolor="black",  # Color del borde del polígono

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno del polígono
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="solid",  # Estilo del borde ('solid', 'dashed', 'dotted', etc.)

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta el polígono según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)
# )

# # Agregar el polígono regular al gráfico
# ax.add_patch(polygon)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1
# ax.set_title("Ejemplo de RegularPolygon en Matplotlib")
# plt.show()





# # ==================================================
# # =============       Rectangle            =============
# # ==================================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Crear figura y ejes
# fig, ax = plt.subplots(figsize=(6, 4))

# # Crear un rectángulo con todos los parámetros
# rectangle = patches.Rectangle(
#     (0.2, 0.2),  # Coordenadas de la esquina inferior izquierda (x, y)
#     width=0.5,  # Ancho del rectángulo
#     height=0.3,  # Altura del rectángulo
    
#     # -------- Personalización del contorno --------
#     linewidth=2,  # Grosor del borde del rectángulo
#     edgecolor="black",  # Color del borde

#     # -------- Color y transparencia --------
#     facecolor="lightblue",  # Color de relleno del rectángulo
#     alpha=0.8,  # Transparencia (1 = sólido, 0 = completamente transparente)

#     # -------- Estilo de línea --------
#     linestyle="solid",  # Estilo de borde ('solid', 'dashed', 'dotted', etc.)

#     # -------- Recorte y orden de dibujo --------
#     clip_on=True,  # Si se recorta según los límites del gráfico
#     zorder=3,  # Prioridad de dibujo (mayor zorder = más arriba)

#     # -------- Otros parámetros --------
#     joinstyle="round",  # Estilo de unión en las esquinas ('miter', 'round', 'bevel')
#     capstyle="round",  # Estilo de las terminaciones de las líneas ('butt', 'round', 'projecting')
#     snap=True,  # Ajuste a la cuadrícula de píxeles (puede mejorar el renderizado)
# )

# # Agregar el rectángulo al gráfico
# ax.add_patch(rectangle)

# # Configurar límites y mostrar
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_aspect("equal")  # Mantener proporción 1:1
# ax.set_title("Ejemplo de Rectangle en Matplotlib")
# plt.show()
