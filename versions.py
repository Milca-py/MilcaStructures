from display.load import moment_fancy_arrow
import numpy as np
import matplotlib.pyplot as plt
from display.load import graphic_n_arrow, graphic_one_arrow, moment_n_arrow

# parameters
x = 0
y = 0
q_i = 100
q_j = 200
length = 1
angle = 53*np.pi/180
ratio_scale = 0.01
nrof_arrows = 10
color = "blue"
angle_rotation = 37*np.pi/180


fig, ax = plt.subplots()
# viga plot
ax.plot(
    [x, x+length*np.cos(angle_rotation)],
    [y, y+length*np.sin(angle_rotation)],
    color='black',
    linewidth=2
)
# distributed load plot
graphic_n_arrow(
    x=x,
    y=y,
    load_i=q_i,
    load_j=q_j,
    angle=angle,
    length=length,
    ax=ax,
    ratio_scale=ratio_scale,
    nrof_arrows=nrof_arrows,
    color=color,
    angle_rotation=angle_rotation
)
plt.axis('equal')


# parameters
x = 0
y = 0
load = -100
# angulo que forma la (cola - cabeza) de la carga con el eje x
angle = 0*np.pi/180
ratio_scale = 0.001
color = "blue"

fig, ax = plt.subplots()
# viga plot
ax.plot(
    [x, x+length*np.cos(angle_rotation)],
    [y, y+length*np.sin(angle_rotation)],
    color='black',
    linewidth=2
)
moment_n_arrow(
    ax=ax,
    x=x,
    y=y,
    load_i=q_i,
    load_j=q_j,
    length=length,
    ratio_scale=ratio_scale,
    nrof_arrows=10,
    color=color,
    angle_rotation=angle_rotation,
    clockwise=True
)
graphic_one_arrow(
    x=x,
    y=y,
    load=load,
    angle=angle,
    ax=ax,
    ratio_scale=ratio_scale,
    color=color
)
moment_fancy_arrow(
    ax=ax,
    x=x+0.5,
    y=y,
    moment=load,
    ratio_scale=ratio_scale,
    color=color,
)


plt.axis('equal')
plt.show()
