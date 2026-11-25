"""Descarga datos horarios de Open-Meteo (reanalisis ERA5) y los resume por dia.
Salida: datos_heladas_diarios.csv, una fila por punto y por dia.

Uso:
    python descargar_datos.py          # todos los puntos
    python descargar_datos.py --foco   # solo los puntos del Altiplano (los que usa el modelo)

Cada tramo descargado se guarda en la carpeta cache_descarga/. Si el programa se
corta (por ejemplo por el limite de uso de la API), al volver a ejecutarlo
continua desde donde quedo, sin repetir lo que ya bajo.
"""
import sys
import time
import pandas as pd
import requests
import config as cfg

CACHE = cfg.CARPETA / "cache_descarga"
MAX_ESPERAS = 40  # tope de esperas por limite de uso en un mismo tramo


def tramos():
    """Divide el periodo en bloques de varios anios para no pedir demasiado."""
    paso = cfg.ANIOS_POR_TRAMO
    return [(f"{a}-01-01", f"{min(a + paso - 1, cfg.ANIO_FIN)}-12-31")
            for a in range(cfg.ANIO_INI, cfg.ANIO_FIN + 1, paso)]


def espera_por_limite(respuesta):
    """Segundos a esperar segun el tipo de limite que reporta la API.
    Devuelve None si es el limite diario (no tiene sentido esperar unas horas)."""
    try:
        motivo = respuesta.json().get("reason", "")
    except ValueError:
        motivo = ""
    texto = motivo.lower()
    if "daily" in texto:
        return None, motivo
    if "hourly" in texto:
        return 300, motivo
    if "minutely" in texto:
        return 65, motivo
    return 120, motivo or "limite de uso"


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
    esperas = 0
    fallos = 0
    while True:
        try:
            r = requests.get(cfg.URL_API, params=params, timeout=120)
        except requests.RequestException as e:
            fallos += 1
            if fallos >= cfg.REINTENTOS:
                raise
            print(f"  Error de red ({e.__class__.__name__}); reintento {fallos}")
            time.sleep(15 * fallos)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            segundos, motivo = espera_por_limite(r)
            if segundos is None:
                sys.exit(f"Limite diario de la API alcanzado ({motivo}).\n"
                         "Vuelve a ejecutar manana: continuara desde la cache.")
            esperas += 1
            if esperas > MAX_ESPERAS:
                sys.exit("La API sigue bloqueando. Vuelve a ejecutar mas tarde: "
                         "continuara desde la cache.")
            print(f"  Limite de uso ({motivo}). Espero {segundos} s "
                  f"({esperas}/{MAX_ESPERAS})...")
            time.sleep(segundos)
            continue
        fallos += 1
        if fallos >= cfg.REINTENTOS:
            r.raise_for_status()
            raise RuntimeError(f"La API respondio con el codigo {r.status_code}")
        time.sleep(15 * fallos)


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


def tramo_con_cache(nombre, lat, lon, zona, ini, fin):
    """Devuelve el resumen diario de un tramo; lo baja solo si no esta en cache."""
    CACHE.mkdir(exist_ok=True)
    archivo = CACHE / f"{nombre.replace(' ', '_')}_{ini}_{fin}.csv"
    if archivo.exists():
        print(f"En cache: {nombre} {ini} a {fin}")
        return pd.read_csv(archivo, parse_dates=["fecha"]), False
    print(f"Descargando {nombre} {ini} a {fin}")
    dias = a_diario(pedir(lat, lon, ini, fin), nombre, zona)
    dias.to_csv(archivo, index=False)
    return dias, True


def descargar(solo_foco=False):
    """Baja todos los puntos y tramos y devuelve una sola tabla diaria."""
    bloques = tramos()
    partes = []
    for nombre, (lat, lon, zona) in cfg.PUNTOS.items():
        if solo_foco and nombre not in cfg.FOCO:
            continue
        for ini, fin in bloques:
            dias, descargado = tramo_con_cache(nombre, lat, lon, zona, ini, fin)
            partes.append(dias)
            if descargado:
                time.sleep(2)  # pausa corta entre llamadas
    return (pd.concat(partes)
            .drop_duplicates(["punto", "fecha"])
            .sort_values(["punto", "fecha"]))


def main():
    solo_foco = "--foco" in sys.argv[1:]
    datos = descargar(solo_foco)
    datos.to_csv(cfg.CSV_DIARIO, index=False)
    print(f"Filas guardadas: {len(datos)} en {cfg.CSV_DIARIO}")
    if solo_foco:
        print("Solo se incluyeron los puntos del Altiplano. Para completar los demas "
              "puntos, ejecuta de nuevo sin --foco (usa la cache).")


if __name__ == "__main__":
    main()