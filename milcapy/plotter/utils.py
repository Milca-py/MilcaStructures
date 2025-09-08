import numpy as np


def calculate_x_intersection(x1, y1, x2, y2):
    """Calcula la intersección con el eje X entre dos puntos"""
    if y1 == y2:
        return x1
    return x1 - y1 * (x2 - x1) / (y2 - y1)

def separate_areas(array, L, x_val=None):
    """Separa las áreas positivas y negativas con puntos de intersección"""
    if x_val is None:
        x_val = np.linspace(0, L, len(array))
    positive_areas = []
    negative_areas = []
    current_pos = []
    current_neg = []
    prev_sign = None

    for i, (x, y) in enumerate(zip(x_val, array)):
        current_sign = 'pos' if y >= 0 else 'neg'

        if i == 0:
            if current_sign == 'pos':
                current_pos.append([x, y])
            else:
                current_neg.append([x, y])
            prev_sign = current_sign
            continue

        if current_sign != prev_sign:
            # Calcular punto de intersección
            x_prev = x_val[i-1]
            y_prev = array[i-1]
            x_intersect = calculate_x_intersection(
                x_prev, y_prev, x, y)

            # Cerrar el área anterior
            if prev_sign == 'pos':
                current_pos.append([x_intersect, 0])
                positive_areas.append(current_pos)
                current_pos = []
            else:
                current_neg.append([x_intersect, 0])
                negative_areas.append(current_neg)
                current_neg = []

            # Iniciar nueva área
            if current_sign == 'pos':
                current_pos = [[x_intersect, 0], [x, y]]
            else:
                current_neg = [[x_intersect, 0], [x, y]]
            prev_sign = current_sign
        else:
            if current_sign == 'pos':
                current_pos.append([x, y])
            else:
                current_neg.append([x, y])
            prev_sign = current_sign

    # Añadir áreas restantes
    if current_pos:
        positive_areas.append(current_pos)
    if current_neg:
        negative_areas.append(current_neg)

    return positive_areas, negative_areas

def process_segments(segments, L):
    """Añade puntos en el eje X para segmentos en los bordes del dominio"""
    processed = []
    for seg in segments:
        new_seg = []
        # Verificar inicio
        if seg and seg[0][0] == 0 and seg[0][1] != 0:
            new_seg.append([0.0, 0.0])

        new_seg.extend(seg)

        # Verificar final
        if seg and seg[-1][0] == L and seg[-1][1] != 0:
            new_seg.append([L, 0.0])

        processed.append(new_seg)
    return processed



def redondear_si_decimal(x: float, ndigits: int = 2):
    """
    Redondea x a 'ndigits' decimales solo si x no es un entero exacto.
    
    Args:
        x (float | int): Número a evaluar.
        ndigits (int): Número de decimales a redondear si corresponde.
    
    Returns:
        float | int: x redondeado o mantenido.
    """
    if float(x).is_integer():
        return int(x)   # mantenerlo como entero
    else:
        return round(x, ndigits)
