"""
Módulo de cargas estructurales.

Proporciona clases y funciones para definir y manipular cargas en un modelo estructural, 
incluyendo cargas puntuales y patrones de carga.
"""

from .load import PointLoad, DistributedLoad
from .load_pattern import LoadPattern