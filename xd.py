# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # fig, ax = plt.subplots()
# # # ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
# # # plt.pause(1)
# # # # plt.show()

# # # ax.clear()
# # # line, = ax.plot([1, 2, 3, 4], [100, 75, 43, 35])
# # # x = np.linspace(0, 10, 1000)  # Valores de X
# # # new_y = np.cos(x)  # Nuevos valores de Y
# # # line.set_data(x, new_y)
# # # fig.canvas.draw()  # Fuerza la actualización de la figura
# # # ax.relim()
# # # ax.autoscale_view()  # Redimensiona la vista de los ejes
# # # plt.pause(1)
# # # fig.clf()
# # # plt.show()


# # import matplotlib.pyplot as plt
# # import numpy as np
# # import time

# # fig, ax = plt.subplots()
# # x = np.linspace(0, 10, 100)
# # line, = ax.plot(x, np.sin(x), label="Seno")

# # # for i in range(10):
# # #     new_y = np.sin(x + i * 0.1)**(2*x**2)  # Actualizar la onda
# # #     line.set_data(x, new_y)  # Modificar la línea

# # #     ax.relim()  # Recalcular límites
# # #     ax.autoscale_view()  # Ajustar vista de ejes
# # #     fig.canvas.draw()  # Redibujar
# # #     fig.canvas.flush_events()  # Forzar actualización en tiempo real
# # #     plt.pause(0.1)  # Pequeña pausa para ver la animación
# # #     line.set_visible(False)
# # #     plt.pause(0.1)
# # #     line.set_visible(True)
# # #     plt.pause(0.1)
# # #     # time.sleep(0.5)  # Pequeña pausa para ver la animación
    

# # def on_click(event):
# #     print(f"Clic en ({event.xdata}, {event.ydata})")

# # # cid = fig.canvas.mpl_connect("button_press_event", on_click)
# # # cid = fig.canvas.mpl_connect("motion_notify_event", on_click)
# # # cid = fig.canvas.mpl_connect("scroll_event", on_click)


# # # plt.ion()  # Activa el modo interactivo
# # # plt.pause(10)  # Pausa para ver la figura
# # # plt.ioff() # Desactiva el modo interactivo
# # plt.figure()  # Crea una nueva ventana
# # plt.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Grafica en la nueva ventana
# # plt.show()



# import tkinter as tk
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# # Crear ventana de Tkinter
# root = tk.Tk()
# root.title("Gráfico Interactivo")

# # Crear figura y Axes
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# line, = ax.plot(x, y, label="Seno")

# # Integrar Matplotlib con Tkinter
# canvas = FigureCanvasTkAgg(fig, master=root)
# canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# # Función para actualizar el gráfico
# def actualizar():
#     new_y = np.sin(x + np.random.uniform(-0.5, 0.5))  # Cambia la función
#     line.set_data(x, new_y)
#     ax.relim()
#     ax.autoscale_view()
#     fig.canvas.draw_idle()

# # Botón para actualizar la gráfica
# btn = tk.Button(root, text="Actualizar", command=actualizar)
# btn.pack()

# root.mainloop()



import examples
import examples.anastuctures
import examples.axial_load_column
import examples.portico_2_aguas
import examples.portico_3_niveles
import examples.portico_arriostrado
import examples.pushover
import examples.vigas

# examples.axial_load_column.run()
# examples.portico_2_aguas.run()
# examples.portico_3_niveles.run()
# examples.pushover.run()
# examples.anastuctures.run()
# examples.portico_arriostrado.run()
examples.vigas.run()