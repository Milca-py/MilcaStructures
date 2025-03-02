# ***DOCUMENTACION DE ANALISIS MATRICIAL***

matriz de regidez local para elementos de tipo marco, con consideraciones de defkeciones de corte.

$$
\begin{bmatrix}
\frac{E A}{L} & 0 & 0 & -\frac{E A}{L} & 0 & 0 \\
0 & \frac{12 E I}{L^3 (1 + \phi)} & \frac{6 E I}{L^2 (1 + \phi)} & 0 & -\frac{12 E I}{L^3 (1 + \phi)} & \frac{6 E I}{L^2 (1 + \phi)} \\
0 & \frac{6 E I}{L^2 (1 + \phi)} & \frac{(4 + \phi) E I}{L (1 + \phi)} & 0 & -\frac{6 E I}{L^2 (1 + \phi)} & \frac{(2 - \phi) E I}{L (1 + \phi)} \\
-\frac{E A}{L} & 0 & 0 & \frac{E A}{L} & 0 & 0 \\
0 & -\frac{12 E I}{L^3 (1 + \phi)} & -\frac{6 E I}{L^2 (1 + \phi)} & 0 & \frac{12 E I}{L^3 (1 + \phi)} & -\frac{6 E I}{L^2 (1 + \phi)} \\
0 & \frac{6 E I}{L^2 (1 + \phi)} & \frac{(2 - \phi) E I}{L (1 + \phi)} & 0 & -\frac{6 E I}{L^2 (1 + \phi)} & \frac{(4 + \phi) E I}{L (1 + \phi)}
\end{bmatrix}
$$

---
matriz de transformacion de coordenadas (rotacion) dezplamientos en "x", "y" y rotacion en "z".


$$
T =
\begin{bmatrix}
\cos\theta & \sin\theta & 0 & 0 & 0 & 0 \\
-\sin\theta & \cos\theta & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & \cos\theta & \sin\theta & 0 \\
0 & 0 & 0 & -\sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

---
vector de fuerzas debido a una carga trapezoidal.

$$
F =
\begin{bmatrix}
0 \\
\frac{7}{20} q_i L + \frac{3}{20} q_j L \\
\frac{1}{20} q_i L^2 + \frac{1}{30} q_j L^2 \\
0 \\
\frac{3}{20} q_i L + \frac{7}{20} q_j L \\
-\left(\frac{1}{30} q_i L^2 + \frac{1}{20} q_j L^2\right)
\end{bmatrix}
$$
