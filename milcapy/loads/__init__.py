"""
Módulo de cargas estructurales.

Proporciona clases y funciones para definir y manipular cargas en un modelo estructural, 
incluyendo cargas puntuales y patrones de carga.
"""

from ..loads.load import PointLoad, DistributedLoad, PrescribedDOF, ElasticSupport, LocalAxis, EndLengthOffset, CSTLoad
from ..loads.load_pattern import LoadPattern