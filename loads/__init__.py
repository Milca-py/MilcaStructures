"""
Módulo de cargas estructurales.

Proporciona clases y funciones para definir y manipular cargas en un modelo estructural, 
incluyendo cargas puntuales y patrones de carga.
"""

from .load import PointLoad, DistributedLoad
from .load_pattern import LoadPattern, loads_to_global_system

__all__ = ["PointLoad", "LoadPattern", "loads_to_global_system"]
