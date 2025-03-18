"""
Módulo de análisis estructural.

Este módulo proporciona herramientas para resolver problemas de análisis estructural
mediante el método de rigidez y análisis matricial.
"""

from .element import (
    local_stiffness_matrix,
    transformation_matrix,
    local_load_vector
)