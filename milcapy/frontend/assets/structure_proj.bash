fem_library/
│
├── core/                           # Núcleo fundamental de la librería
│   ├── __init__.py                 # Exporta clases principales
│   ├── node.py                     # Definición de nodos
│   ├── element.py                  # Clase base abstracta para elementos
│   ├── dof.py                      # Sistema de grados de libertad
│   ├── material.py                 # Clase base para materiales
│   ├── section.py                  # Clase base para secciones
│   ├── model.py                    # Contenedor principal del modelo estructural
│   └── result.py                   # Clases base para resultados
│
├── elements/                       # Implementaciones específicas de elementos
│   ├── base/                       # Clases base por categoría de elementos
│   │   ├── element.py              # Elemento de viga (base)
│   │
│   ├── bars/                       # Elementos tipo barra
│   │   ├── truss.py                # Elemento de armadura 3D
│   │
│   ├── beams/                      # Elementos tipo viga
│       └── frame.py                # Elemento de marco
│    
├── materials/                      # Implementaciones de materiales
│   ├── elastic.py                  # Material elástico lineal
│   ├── concrete.py                 # Modelos de concreto
│
├── sections/                       # Propiedades de sección
│   ├── geometric/                  # Propiedades geométricas
│   │   ├── area.py                 # Cálculo de áreas
│   │   ├── inertia.py              # Cálculo de inercias
│   │   └── torsion.py              # Constantes de torsión
│   │
│   ├── standard/                   # Secciones estándar
│   │   ├── rectangular.py          # Sección rectangular
│   │   ├── circular.py             # Sección circular
│
├── loads/                          # Sistema de cargas
│   ├── load.py                     # Clase base para cargas
│   ├── nodal_load.py               # Cargas puntuales en nodos
│   ├── distributed_load.py         # Cargas distribuidas
│   ├── self_weight.py              # Peso propio
│   ├── load_combination.py         # Combinaciones de carga
│   └── load_case.py                # Casos de carga
│
├── constraints/                    # Restricciones y condiciones de frontera
│   ├── support.py                  # Apoyos estructurales
│   ├── constraint.py               # Restricciones generales
│
├── assembly/                       # Ensamblaje del sistema
│   ├── assembler.py                # Ensamblador general
│   ├── dof_mapper.py               # Mapeador de DOFs
│   ├── stiffness_assembly.py       # Ensamblaje de matriz de rigidez
│   └── load_assembly.py            # Ensamblaje de vector de carga
│
├── solvers/                        # Resolvedores
│   ├── linear/                     # Solvers lineales
│       ├── direct_solver.py        # Solucionador directo
│   
├── analysis/                       # Tipos de análisis
│   ├── static.py                   # Análisis estático
│
├── postprocess/                    # Postprocesado
│   │   ├── displacement.py         # Campo de desplazamientos
│   │
│   ├── derived/                    # Resultados derivados
│   │   ├── reaction.py             # Reacciones
│   │   ├── internal_forces.py      # Fuerzas internas
│   │
│   └── result_container.py         # Contenedor de resultados
│
├── utilities/                      # Utilidades
│   ├── geometry/                   # Geometría
│   │   ├── vectex.py               # Punto en el espacio
│   │   ├── coordinate_systems.py   # Sistemas de coordenadas
│   │   ├── transformations.py      # Transformaciones
│
├── visualization/                  # Visualización
│   ├── model_viewer.py             # Visualizador de modelo
│   ├── result_viewer.py            # Visualizador de resultados
│   ├── deformed_shape.py           # Forma deformada