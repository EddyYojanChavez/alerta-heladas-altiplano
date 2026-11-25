"""Entrena y compara los modelos. Uso: python entrenar.py

Pasos: cargar datos -> lineas base -> redes (varias semillas) -> tablas de
resultados -> guardar la mejor red. La mejor semilla y la mejor red se eligen
con el F1 de validacion; el conjunto de prueba solo se usa para informar.
"""
import json

import joblib
import numpy as np
import pandas as pd

import config as cfg
import lineas_base
import metricas
import modelos
import preparacion


def evaluar_lineas_base(conjuntos):
    """Filas de resultados de persistencia y regresion logistica."""
    p_val, p_test = lineas_base.persistencia(conjuntos)
    filas = metricas.evaluar("Persistencia", conjuntos, p_val, p_test,
                             umbral_fijo=cfg.UMBRAL_PERSISTENCIA)
    p_val, p_test = lineas_base.regresion_logistica(conjuntos)
    filas += metricas.evaluar("Regresion logistica", conjuntos, p_val, p_test)
    return filas


def entrenar_varias_semillas(tipo, conjuntos, pesos):
    """Entrena la misma red con N_SEMILLAS semillas distintas.

    Devuelve una lista de dicts con modelo, probabilidades y filas de metricas.
    """
    corridas = []
    for i in range(cfg.N_SEMILLAS):
        semilla = cfg.SEMILLA + i
        print(f"Entrenando {tipo.upper()} (semilla {semilla})")
        modelo, p_val, p_test = modelos.entrenar_red(tipo, conjuntos, pesos, semilla)
        filas = metricas.evaluar(tipo.upper(), conjuntos, p_val, p_test)
        corridas.append({"modelo": modelo, "p_val": p_val, "p_test": p_test,
                         "filas": filas,
                         "f1_val": float(np.mean([f["f1_val"] for f in filas]))})
    return corridas


def mejor_corrida(corridas):
    """La corrida con mayor F1 de validacion (promedio de los tres horizontes)."""
    return max(corridas, key=lambda c: c["f1_val"])


def estabilidad(tipo, corridas):
    """Media y desviacion del F1 de prueba entre semillas, por horizonte."""
    tabla = pd.DataFrame([f for c in corridas for f in c["filas"]])
    resumen = (tabla.groupby("horizonte_dias")["f1_helada"].agg(["mean", "std"])
               .reset_index().rename(columns={"mean": "f1_media", "std": "f1_desv"}))
    resumen.insert(0, "modelo", tipo.upper())
    return resumen


def quitar_auxiliar(filas):
    """Quita la columna auxiliar f1_val antes de guardar la tabla final."""
    return pd.DataFrame(filas).drop(columns="f1_val").round(3)


def main():
    df = preparacion.cargar(cfg.CSV_DIARIO)
    conjuntos, escalador = preparacion.construir(df)
    for nombre, c in conjuntos.items():
        print(f"{nombre}: {len(c.x)} ventanas, heladas {c.y.mean(axis=0).round(3)}")

    pesos = preparacion.pesos_de_clase(conjuntos["train"].y.mean(axis=0))
    filas = evaluar_lineas_base(conjuntos)

    mejores, tablas_estabilidad = {}, []
    for tipo in cfg.REDES:
        corridas = entrenar_varias_semillas(tipo, conjuntos, pesos)
        mejores[tipo] = mejor_corrida(corridas)
        filas += mejores[tipo]["filas"]
        tablas_estabilidad.append(estabilidad(tipo, corridas))

    resultados = quitar_auxiliar(filas)
    resultados.to_csv(cfg.CSV_RESULTADOS, index=False)
    print(resultados.to_string(index=False))
    print(pd.concat(tablas_estabilidad).round(3).to_string(index=False))

    # La red ganadora se elige con validacion, no con prueba.
    ganadora = max(mejores, key=lambda t: mejores[t]["f1_val"])
    corrida = mejores[ganadora]
    umbrales = [f["umbral"] for f in corrida["filas"]]
    print(f"Red elegida (F1 de validacion): {ganadora.upper()}")

    for por, ruta in (("punto", cfg.CSV_POR_PUNTO), ("mes", cfg.CSV_POR_MES)):
        tabla = metricas.por_grupo(conjuntos["test"], corrida["p_test"], umbrales, por)
        tabla.round(3).to_csv(ruta, index=False)

    modelos.guardar_red(corrida["modelo"], cfg.MODELO_KERAS)
    joblib.dump(escalador, cfg.ESCALADOR)
    cfg.UMBRALES.write_text(json.dumps(
        {"red": ganadora, "umbrales": umbrales}, indent=2))
    print("Guardados: modelo, escalador y umbrales")


if __name__ == "__main__":
    main()
