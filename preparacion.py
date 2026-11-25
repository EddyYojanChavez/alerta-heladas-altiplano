"""Preparacion de datos: etiquetas de helada, ventanas de 7 dias y division por fecha.

No usa TensorFlow: se puede probar solo con pandas y scikit-learn.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as cfg


@dataclass
class Conjunto:
    """Un subconjunto de datos: entrenamiento, validacion o prueba."""
    x: np.ndarray         # ventanas escaladas, forma (n, VENTANA, len(ENTRADAS))
    y: np.ndarray         # etiquetas 0/1, forma (n, len(HORIZONTES))
    tmin_hoy: np.ndarray  # tmin del ultimo dia de cada ventana, sin escalar
    meta: pd.DataFrame    # columnas "punto" y "fecha" de cada ventana


def agregar_estacionalidad(df):
    """Agrega el dia del anio como seno y coseno, para que el modelo vea la estacion."""
    df = df.copy()
    doy = df["fecha"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def agregar_etiquetas(df):
    """y{h} = 1 si la tmin del dia t+h es menor al umbral.

    Se descartan los dias que no tienen futuro conocido.

    Requiere que df este ordenado por punto y fecha.
    """
    df = df.copy()
    for h in cfg.HORIZONTES:
        futuro = df.groupby("punto")["tmin"].shift(-h)
        helada = (futuro < cfg.UMBRAL_HELADA).astype(float)
        df[f"y{h}"] = np.where(futuro.isna(), np.nan, helada)
    return df.dropna(subset=cfg.COLUMNAS_Y)


def cargar_diario(ruta):
    """Lee el CSV diario, deja solo los puntos de cfg.FOCO y avisa si faltan dias."""
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df[df["punto"].isin(cfg.FOCO)]
    df = df.dropna(subset=cfg.VARIABLES_METEO)
    df = df.sort_values(["punto", "fecha"]).reset_index(drop=True)
    for punto, g in df.groupby("punto"):
        saltos = (g["fecha"].diff().dropna() != pd.Timedelta(days=1)).sum()
        if saltos:
            print(f"Aviso: {punto} tiene {saltos} saltos de fecha")
    return df


def cargar(ruta):
    """CSV diario listo para armar ventanas: con estacionalidad y etiquetas."""
    df = agregar_etiquetas(agregar_estacionalidad(cargar_diario(ruta)))
    return df.reset_index(drop=True)


def construir(df):
    """Escala, arma ventanas y divide por fecha.

    Devuelve ({"train": Conjunto, "val": Conjunto, "test": Conjunto}, escalador).
    El escalador se ajusta solo con los anios de entrenamiento para no filtrar
    informacion del futuro.
    """
    anio_fila = df["fecha"].dt.year
    entrenamiento = df.loc[anio_fila < cfg.ANIO_VAL, cfg.ENTRADAS].values
    escalador = StandardScaler().fit(entrenamiento)

    ultimo = cfg.VENTANA - 1  # posicion del ultimo dia de la primera ventana
    xs, ys, tmins, metas = [], [], [], []
    for punto, g in df.groupby("punto"):
        escaladas = escalador.transform(g[cfg.ENTRADAS].values)
        ventanas = np.lib.stride_tricks.sliding_window_view(
            escaladas, (cfg.VENTANA, len(cfg.ENTRADAS)))[:, 0]
        xs.append(ventanas)
        ys.append(g[cfg.COLUMNAS_Y].values[ultimo:])
        tmins.append(g["tmin"].values[ultimo:])
        metas.append(pd.DataFrame({"punto": punto,
                                   "fecha": g["fecha"].values[ultimo:]}))

    x = np.concatenate(xs).astype("float32")
    y = np.concatenate(ys).astype("float32")
    tmin_hoy = np.concatenate(tmins)
    meta = pd.concat(metas, ignore_index=True)

    anio = meta["fecha"].dt.year.to_numpy()
    mascaras = {
        "train": anio < cfg.ANIO_VAL,
        "val": (anio >= cfg.ANIO_VAL) & (anio < cfg.ANIO_TEST),
        "test": anio >= cfg.ANIO_TEST,
    }
    conjuntos = {
        nombre: Conjunto(x[m], y[m], tmin_hoy[m], meta[m].reset_index(drop=True))
        for nombre, m in mascaras.items()
    }
    return conjuntos, escalador


def pesos_de_clase(prevalencia):
    """Peso de la clase helada por horizonte: mayor mientras mas rara sea."""
    return np.clip((1 - prevalencia) / np.maximum(prevalencia, 1e-3), 1, 10)
