"""Descarga datos horarios de Open-Meteo (reanalisis ERA5) y los resume por dia.

Salida: datos_heladas_diarios.csv, una fila por punto y por dia.
Uso:    python descargar_datos.py
"""
import time

import pandas as pd
import requests

import config as cfg


def tramos():
    """Divide el periodo en bloques de varios anios para no pedir demasiado."""
    paso = cfg.ANIOS_POR_TRAMO
    return [(f"{a}-01-01", f"{min(a + paso - 1, cfg.ANIO_FIN)}-12-31")
            for a in range(cfg.ANIO_INI, cfg.ANIO_FIN + 1, paso)]


def pedir(lat, lon, ini, fin):
    """Consulta la API; si responde con limite de uso, espera y reintenta."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": ini,
        "end_date": fin,
        "hourly": ",".join(cfg.VARIABLES_HORARIAS),
        "timezone": cfg.ZONA_HORARIA,
        "models": "era5",  # si la API devuelve error por este campo, quitalo
    }
    for intento in range(1, cfg.REINTENTOS + 1):
        r = requests.get(cfg.URL_API, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if intento < cfg.REINTENTOS:
            time.sleep(15 * intento)
    r.raise_for_status()
    raise RuntimeError(f"La API respondio con el codigo {r.status_code}")


def a_diario(respuesta, nombre, zona):
    """Convierte las lecturas horarias de un punto en un resumen diario."""
    horas = pd.DataFrame(respuesta["hourly"])
    horas["fecha"] = pd.to_datetime(horas["time"]).dt.normalize()
    dias = horas.groupby("fecha").agg(
        tmin=("temperature_2m", "min"),
        tmax=("temperature_2m", "max"),
        tmean=("temperature_2m", "mean"),
        hr=("relative_humidity_2m", "mean"),
        td=("dew_point_2m", "mean"),
        presion=("surface_pressure", "mean"),
        nubes=("cloud_cover", "mean"),
        viento=("wind_speed_10m", "mean"),
        radiacion=("shortwave_radiation", "sum"),
    ).reset_index()
    dias["punto"] = nombre
    dias["zona"] = zona
    dias["elevacion"] = respuesta.get("elevation")  # altitud de la celda, en metros
    return dias


def descargar():
    """Baja todos los puntos y tramos y devuelve una sola tabla diaria."""
    bloques = tramos()
    partes = []
    for nombre, (lat, lon, zona) in cfg.PUNTOS.items():
        for ini, fin in bloques:
            print(f"Descargando {nombre} {ini} a {fin}")
            partes.append(a_diario(pedir(lat, lon, ini, fin), nombre, zona))
            time.sleep(2)  # pausa corta entre llamadas
    return (pd.concat(partes)
            .drop_duplicates(["punto", "fecha"])
            .sort_values(["punto", "fecha"]))


def main():
    datos = descargar()
    datos.to_csv(cfg.CSV_DIARIO, index=False)
    print(f"Filas guardadas: {len(datos)} en {cfg.CSV_DIARIO}")


if __name__ == "__main__":
    main()
