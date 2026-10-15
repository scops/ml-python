"""Funciones propias del proyecto integrador.

Las importan el notebook 12_proyecto.ipynb y cualquier script que cargue los modelos. Vivir en un
módulo, y no en una celda, es lo que permite que joblib las encuentre al cargar.
"""
import numpy as np
import pandas as pd

MARGEN = 0.12      # parte de lo que gasta un cliente que se queda la tienda como beneficio
OFERTA = 15        # euros que cuesta cada llamada con su descuento


def log_pos(x):
    """Logaritmo que no falla con ceros ni negativos. La usa modelos/gasto.joblib."""
    return np.log(np.clip(x, 1, None))


def puntuar(clientes, abandono, gasto, segmentos, smear):
    """Una fila por cliente con todo lo que necesita la campaña de retención."""
    p = abandono.predict_proba(clientes)[:, 1]
    g = np.exp(gasto.predict(clientes)) * smear
    return pd.DataFrame({
        "p_abandono": p,
        "gasto_si_se_queda": g,
        "valor_esperado": (1 - p) * g,
        "perdida_esperada": p * MARGEN * g,
        "segmento": segmentos.predict(clientes),
        "llamar": p * MARGEN * g > OFERTA,
    }, index=clientes.index)
