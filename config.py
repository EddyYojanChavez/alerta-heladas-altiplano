"""Parametros del proyecto. Es la unica fuente de verdad para todos los modulos."""
from pathlib import Path

# --- Rutas de entrada y salida -------------------------------------------------
CARPETA = Path(__file__).resolve().parent
CSV_DIARIO = CARPETA / "datos_heladas_diarios.csv"
CSV_RESULTADOS = CARPETA / "resultados.csv"
CSV_POR_PUNTO = CARPETA / "resultados_por_punto.csv"
CSV_POR_MES = CARPETA / "resultados_por_mes.csv"
MODELO_KERAS = CARPETA / "modelo_heladas.keras"
ESCALADOR = CARPETA / "escalador_heladas.joblib"
UMBRALES = CARPETA / "umbrales_heladas.json"

# --- Puntos de estudio: nombre -> (latitud, longitud, zona) --------------------
# Las coordenadas son aproximadas: verificalas en un mapa.
PUNTOS = {
    "El Alto": (-16.51, -68.18, "altiplano"),
    "Viacha": (-16.65, -68.30, "altiplano"),
    "Patacamaya": (-17.24, -67.92, "altiplano"),
    "Achacachi": (-16.05, -68.69, "altiplano"),
    "Copacabana": (-16.17, -69.09, "altiplano"),
    "Desaguadero": (-16.57, -69.04, "altiplano"),
    "Sorata": (-15.77, -68.65, "valle"),
    "Coroico": (-16.19, -67.73, "yungas"),
    "Rurrenabaque": (-14.44, -67.53, "tropico"),
}
# El modelo se entrena solo con los puntos del Altiplano.
FOCO = [nombre for nombre, (_, _, zona) in PUNTOS.items() if zona == "altiplano"]

# --- Descarga (Open-Meteo, reanalisis ERA5) ------------------------------------
URL_API = "https://archive-api.open-meteo.com/v1/archive"
ZONA_HORARIA = "America/La_Paz"
ANIO_INI, ANIO_FIN = 1990, 2025
ANIOS_POR_TRAMO = 5
REINTENTOS = 4
VARIABLES_HORARIAS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "shortwave_radiation",
]

# --- Definicion del problema ---------------------------------------------------
UMBRAL_HELADA = 0.0               # grados C: hay helada si la tmin del dia es menor
VENTANA = 7                       # dias hacia atras que ve el modelo
HORIZONTES = (1, 2, 3)            # dias hacia adelante que se predicen
ANIO_VAL, ANIO_TEST = 2018, 2022  # entrena < 2018, valida 2018-2021, prueba >= 2022

# --- Variables que entran al modelo --------------------------------------------
VARIABLES_METEO = ["tmin", "tmax", "tmean", "hr", "td", "presion", "nubes",
                   "viento", "radiacion", "elevacion"]
ENTRADAS = VARIABLES_METEO + ["doy_sin", "doy_cos"]
COLUMNAS_Y = [f"y{h}" for h in HORIZONTES]

# --- Entrenamiento -------------------------------------------------------------
REDES = ("mlp", "lstm")
SEMILLA = 42                      # primera semilla; las demas son SEMILLA + 1, + 2...
N_SEMILLAS = 5                    # repeticiones del entrenamiento de cada red
EPOCAS, TAMANO_LOTE, PACIENCIA = 60, 256, 8
UMBRAL_PERSISTENCIA = 0.5         # la persistencia es una regla: umbral fijo
