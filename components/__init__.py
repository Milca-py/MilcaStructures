"""
Módulo de análisis estructural.

Este módulo proporciona herramientas para resolver problemas de análisis estructural
mediante el método de rigidez y análisis matricial.
"""

from .system_components import (
    solve,
    process_conditions,
    assemble_global_load_vector,
    assemble_global_stiffness_matrix
)


from .element_components import (
    local_stiffness_matrix,
    transformation_matrix,
    local_load_vector
)