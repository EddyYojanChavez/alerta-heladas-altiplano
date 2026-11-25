"""Metricas: umbral de decision, tabla por horizonte y analisis por punto o mes."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import config as cfg


def mejor_umbral(y, p):
    """Umbral que maximiza el F1 sobre los datos recibidos (se usa en validacion)."""
    y = y.astype(int)
    umbrales = np.arange(0.05, 0.96, 0.05)
    return float(max(umbrales,
                     key=lambda u: f1_score(y, (p >= u).astype(int), zero_division=0)))


def calcular(y, p, umbral):
    """Precision, exhaustividad, F1 de la clase helada y AUC de un horizonte."""
    y = y.astype(int)
    pred = (p >= umbral).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "exhaustividad": recall_score(y, pred, zero_division=0),
        "f1_helada": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
    }


def evaluar(nombre, conjuntos, p_val, p_test, umbral_fijo=None):
    """Una fila por horizonte. El umbral se elige en validacion y se mide en prueba."""
    y_val, y_test = conjuntos["val"].y, conjuntos["test"].y
    filas = []
    for j, h in enumerate(cfg.HORIZONTES):
        if umbral_fijo is None:
            umbral = mejor_umbral(y_val[:, j], p_val[:, j])
        else:
            umbral = umbral_fijo
        fila = {
            "modelo": nombre,
            "horizonte_dias": h,
            "umbral": round(umbral, 2),
            "f1_val": f1_score(y_val[:, j].astype(int),
                               (p_val[:, j] >= umbral).astype(int), zero_division=0),
        }
        fila.update(calcular(y_test[:, j], p_test[:, j], umbral))
        filas.append(fila)
    return filas


def por_grupo(conjunto, p, umbrales, por):
    """Metricas del conjunto de prueba agrupadas por 'punto' o por 'mes'."""
    meta = conjunto.meta.assign(mes=conjunto.meta["fecha"].dt.month)
    filas = []
    for j, h in enumerate(cfg.HORIZONTES):
        for grupo, pos in meta.groupby(por).indices.items():
            y = conjunto.y[pos, j]
            fila = {"horizonte_dias": h, por: grupo, "n": len(pos),
                    "heladas": int(y.sum())}
            fila.update(calcular(y, p[pos, j], umbrales[j]))
            if fila["heladas"] == 0:  # sin heladas, estas metricas no significan nada
                for k in ("precision", "exhaustividad", "f1_helada"):
                    fila[k] = np.nan
            filas.append(fila)
    return pd.DataFrame(filas)
