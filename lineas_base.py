"""Modelos de referencia: persistencia y regresion logistica.

Cada funcion devuelve (p_val, p_test), con forma (n, len(HORIZONTES)), para
evaluarlas igual que las redes neuronales.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

import config as cfg


def persistencia(conjuntos):
    """Regla simple: habra helada si hoy ya hubo helada (tmin de hoy < umbral)."""
    salida = []
    for nombre in ("val", "test"):
        hoy = (conjuntos[nombre].tmin_hoy < cfg.UMBRAL_HELADA).astype(float)
        salida.append(np.repeat(hoy[:, None], len(cfg.HORIZONTES), axis=1))
    return tuple(salida)


def regresion_logistica(conjuntos):
    """Un modelo por horizonte, sobre la ventana aplanada."""
    plano = {n: c.x.reshape(len(c.x), -1) for n, c in conjuntos.items()}
    p_val, p_test = [], []
    for j in range(len(cfg.HORIZONTES)):
        modelo = LogisticRegression(max_iter=2000, class_weight="balanced")
        modelo.fit(plano["train"], conjuntos["train"].y[:, j])
        p_val.append(modelo.predict_proba(plano["val"])[:, 1])
        p_test.append(modelo.predict_proba(plano["test"])[:, 1])
    return np.column_stack(p_val), np.column_stack(p_test)
