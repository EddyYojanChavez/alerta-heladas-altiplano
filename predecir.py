"""Aviso de helada para los proximos dias, a partir de los ultimos 7 dias medidos.

Uso:  python predecir.py ultimos_7_dias.csv
El CSV debe traer 7 filas (una por dia, en orden) con las columnas:
fecha, tmin, tmax, tmean, hr, td, presion, nubes, viento, radiacion, elevacion
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

import config as cfg
import preparacion


def predecir(ultimos_dias, modelo, escalador, umbrales):
    """Devuelve {dias: (probabilidad, hay_alerta)} para cada horizonte."""
    if len(ultimos_dias) != cfg.VENTANA:
        raise ValueError(f"Se necesitan {cfg.VENTANA} dias exactos, llegaron "
                         f"{len(ultimos_dias)}")
    faltan = set(["fecha"] + cfg.VARIABLES_METEO) - set(ultimos_dias.columns)
    if faltan:
        raise ValueError(f"Faltan columnas: {sorted(faltan)}")

    datos = preparacion.agregar_estacionalidad(ultimos_dias.sort_values("fecha"))
    x = escalador.transform(datos[cfg.ENTRADAS].values)[None, :, :].astype("float32")
    p = np.asarray(modelo.predict(x, verbose=0))[0]
    return {h: (float(p[j]), bool(p[j] >= umbrales[j]))
            for j, h in enumerate(cfg.HORIZONTES)}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    import modelos  # solo aqui, para que predecir() se pueda probar sin TensorFlow
    ultimos = pd.read_csv(sys.argv[1], parse_dates=["fecha"])
    modelo = modelos.cargar_red(cfg.MODELO_KERAS)
    escalador = joblib.load(cfg.ESCALADOR)
    umbrales = json.loads(cfg.UMBRALES.read_text())["umbrales"]
    for h, (prob, alerta) in predecir(ultimos, modelo, escalador, umbrales).items():
        estado = "ALERTA DE HELADA" if alerta else "sin alerta"
        print(f"A {h} dia(s): probabilidad {prob:.0%} -> {estado}")


if __name__ == "__main__":
    main()
