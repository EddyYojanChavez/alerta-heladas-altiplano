"""Dashboard del proyecto: alerta temprana de heladas en el Altiplano paceño.

Uso local:   streamlit run dashboard.py
Requiere haber corrido antes `python entrenar.py` (genera el modelo y las tablas).
"""
import json
from datetime import timedelta

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

import config as cfg
from descargar_datos import a_diario
from predecir import predecir

URL_PRONOSTICO = "https://api.open-meteo.com/v1/forecast"

st.set_page_config(page_title="Alerta de heladas - Altiplano", page_icon="❄️",
                   layout="wide")


# --------------------------------------------------------------------------- #
# Carga de recursos (se guardan en caché para no repetir trabajo)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo():
    """Carga la red guardada, el escalador y los umbrales. None si faltan."""
    if not (cfg.MODELO_KERAS.exists() and cfg.ESCALADOR.exists()
            and cfg.UMBRALES.exists()):
        return None
    import modelos  # TensorFlow solo se importa aquí
    modelo = modelos.cargar_red(cfg.MODELO_KERAS)
    escalador = joblib.load(cfg.ESCALADOR)
    info = json.loads(cfg.UMBRALES.read_text())
    return modelo, escalador, info


@st.cache_data(show_spinner=False)
def leer_csv(ruta):
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data(ttl=3600, show_spinner="Consultando Open-Meteo...")
def ultimos_7_dias(punto):
    """Baja los últimos 7 días completos de un punto desde Open-Meteo."""
    lat, lon, zona = cfg.PUNTOS[punto]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(cfg.VARIABLES_HORARIAS),
        "past_days": cfg.VENTANA + 1,
        "forecast_days": 1,
        "timezone": cfg.ZONA_HORARIA,
    }
    r = requests.get(URL_PRONOSTICO, params=params, timeout=60)
    r.raise_for_status()
    dias = a_diario(r.json(), punto, zona)
    hoy = pd.Timestamp.now(tz=cfg.ZONA_HORARIA).tz_localize(None).normalize()
    dias = dias[dias["fecha"] < hoy].dropna(subset=cfg.VARIABLES_METEO)
    dias = dias.tail(cfg.VENTANA).reset_index(drop=True)
    if len(dias) < cfg.VENTANA:
        raise ValueError(f"Solo llegaron {len(dias)} días completos de {cfg.VENTANA}")
    return dias


# --------------------------------------------------------------------------- #
# Piezas visuales
# --------------------------------------------------------------------------- #
def medidor(prob, umbral, titulo):
    """Medidor de probabilidad; la línea negra marca el umbral de alerta."""
    alerta = prob >= umbral
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "valueformat": ".0f"},
        title={"text": titulo},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#c0392b" if alerta else "#2e86c1"},
            "threshold": {"line": {"color": "black", "width": 4},
                          "thickness": 0.8, "value": umbral * 100},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def mostrar_avisos(ultimos, resultado, umbrales_info):
    """Medidores y mensajes de alerta para 1, 2 y 3 días."""
    ultimo_dia = ultimos["fecha"].max()
    cols = st.columns(len(cfg.HORIZONTES))
    hay_alerta = False
    for col, h in zip(cols, cfg.HORIZONTES):
        prob, alerta = resultado[h]
        umbral = umbrales_info["umbrales"][cfg.HORIZONTES.index(h)]
        fecha_obj = (ultimo_dia + timedelta(days=h)).strftime("%d/%m/%Y")
        with col:
            st.plotly_chart(medidor(prob, umbral, f"Noche del {fecha_obj} (+{h} d)"),
                            use_container_width=True)
            if alerta:
                st.error("ALERTA DE HELADA")
            else:
                st.success("Sin alerta")
            st.caption(f"Umbral de alerta: {umbral:.0%}")
        hay_alerta = hay_alerta or alerta
    return hay_alerta


def grafico_ultimos_dias(ultimos):
    """Temperaturas de los 7 días que alimentan al modelo."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ultimos["fecha"], y=ultimos["tmin"], name="Mínima",
                             mode="lines+markers", line=dict(color="#2e86c1")))
    fig.add_trace(go.Scatter(x=ultimos["fecha"], y=ultimos["tmax"], name="Máxima",
                             mode="lines+markers", line=dict(color="#e67e22")))
    fig.add_hline(y=cfg.UMBRAL_HELADA, line_dash="dash", line_color="red",
                  annotation_text="0 °C")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Temperatura (°C)", legend=dict(orientation="h"))
    return fig


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
st.sidebar.title("❄️ Alerta de heladas")
st.sidebar.caption("Altiplano paceño · Redes neuronales (MLP / LSTM)")
punto = st.sidebar.selectbox("Punto de estudio", cfg.FOCO)
if st.sidebar.button("Actualizar datos"):
    st.cache_data.clear()
st.sidebar.divider()
st.sidebar.caption(
    "Prototipo académico. Usa reanálisis y pronóstico de Open-Meteo, no mediciones "
    "de estación. **No sustituye los avisos oficiales de SENAMHI.**")

st.title("Alerta temprana de heladas en el Altiplano paceño")

paquete = cargar_modelo()
if paquete is None:
    st.warning("Todavía no hay modelo entrenado. Ejecuta `python descargar_datos.py` "
               "y luego `python entrenar.py`, y vuelve a abrir el dashboard.")
    st.stop()
modelo, escalador, info = paquete

tab_alerta, tab_res, tab_geo, tab_csv, tab_info = st.tabs(
    ["🔔 Alerta actual", "📊 Resultados del modelo", "🗺️ Por punto y mes",
     "📁 Probar con mis datos", "ℹ️ Acerca de"])

# --------------------------------------------------------------------------- #
# Pestaña 1: alerta actual
# --------------------------------------------------------------------------- #
with tab_alerta:
    st.subheader(f"Pronóstico de helada en {punto}")
    try:
        ultimos = ultimos_7_dias(punto)
    except Exception as e:  # sin internet, límite de la API, etc.
        st.error(f"No se pudieron obtener los datos recientes: {e}")
        st.info("Puedes probar el modelo subiendo un CSV en la pestaña "
                "«Probar con mis datos».")
    else:
        resultado = predecir(ultimos, modelo, escalador, info["umbrales"])
        hay_alerta = mostrar_avisos(ultimos, resultado, info)
        if hay_alerta:
            st.warning("Hay riesgo de helada en los próximos días: conviene cubrir "
                       "cultivos, regar antes de la noche y resguardar animales.")
        izq, der = st.columns([3, 2])
        with izq:
            st.markdown("**Últimos 7 días medidos (entrada del modelo)**")
            st.plotly_chart(grafico_ultimos_dias(ultimos), use_container_width=True)
        with der:
            lat, lon, _ = cfg.PUNTOS[punto]
            st.markdown("**Ubicación**")
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=8, size=3000)
        with st.expander("Ver datos de entrada"):
            st.dataframe(ultimos[["fecha"] + cfg.VARIABLES_METEO].round(2),
                         hide_index=True)
        st.caption(f"Red usada: {info.get('red', 'desconocida').upper()} · "
                   f"datos hasta el {ultimos['fecha'].max():%d/%m/%Y}. El modelo se "
                   "entrenó con reanálisis ERA5; los datos recientes vienen del "
                   "pronóstico de Open-Meteo, y esa diferencia puede reducir el acierto.")

# --------------------------------------------------------------------------- #
# Pestaña 2: resultados globales
# --------------------------------------------------------------------------- #
with tab_res:
    res = leer_csv(cfg.CSV_RESULTADOS)
    if res is None:
        st.warning("No se encontró resultados.csv. Ejecuta `python entrenar.py`.")
    else:
        st.subheader("Comparación de modelos en el conjunto de prueba (2022-2025)")
        metrica = st.selectbox(
            "Métrica", ["f1_helada", "precision", "exhaustividad", "auc"],
            format_func=lambda m: {"f1_helada": "F1 de la clase helada",
                                   "precision": "Precisión",
                                   "exhaustividad": "Exhaustividad",
                                   "auc": "AUC-ROC"}[m])
        fig = px.bar(res, x="horizonte_dias", y=metrica, color="modelo",
                     barmode="group", labels={"horizonte_dias": "Horizonte (días)"})
        fig.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(res, hide_index=True)
        st.caption("Si una red no supera a la regresión logística o a la persistencia, "
                   "ese es el resultado: se reporta tal cual.")

# --------------------------------------------------------------------------- #
# Pestaña 3: por punto y por mes
# --------------------------------------------------------------------------- #
with tab_geo:
    por_punto = leer_csv(cfg.CSV_POR_PUNTO)
    por_mes = leer_csv(cfg.CSV_POR_MES)
    if por_punto is None or por_mes is None:
        st.warning("Faltan resultados_por_punto.csv o resultados_por_mes.csv. "
                   "Ejecuta `python entrenar.py`.")
    else:
        h_sel = st.radio("Horizonte", list(cfg.HORIZONTES), horizontal=True,
                         format_func=lambda h: f"{h} día(s)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**F1 de la clase helada por punto**")
            d = por_punto[por_punto["horizonte_dias"] == h_sel]
            fig = px.bar(d, x="punto", y="f1_helada", hover_data=["n", "heladas"])
            fig.update_layout(height=350, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**F1 de la clase helada por mes**")
            d = por_mes[por_mes["horizonte_dias"] == h_sel]
            fig = px.line(d, x="mes", y="f1_helada", markers=True,
                          hover_data=["n", "heladas"])
            fig.update_layout(height=350, yaxis_range=[0, 1])
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Noches con helada por mes (conjunto de prueba)**")
        d = por_mes[por_mes["horizonte_dias"] == h_sel]
        fig = px.bar(d, x="mes", y="heladas")
        fig.update_layout(height=280)
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Los meses sin heladas no tienen F1: aparecen vacíos.")

# --------------------------------------------------------------------------- #
# Pestaña 4: probar con un CSV propio
# --------------------------------------------------------------------------- #
with tab_csv:
    st.subheader("Probar el modelo con tus propios datos")
    st.write("Sube un CSV con **7 filas** (una por día, en orden) y las columnas: "
             "`fecha, tmin, tmax, tmean, hr, td, presion, nubes, viento, radiacion, "
             "elevacion`.")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            propios = pd.read_csv(archivo, parse_dates=["fecha"])
            res_propio = predecir(propios, modelo, escalador, info["umbrales"])
        except Exception as e:
            st.error(f"No se pudo usar el archivo: {e}")
        else:
            mostrar_avisos(propios, res_propio, info)
            st.plotly_chart(grafico_ultimos_dias(propios), use_container_width=True)

# --------------------------------------------------------------------------- #
# Pestaña 5: acerca de
# --------------------------------------------------------------------------- #
with tab_info:
    st.subheader("Acerca del proyecto")
    st.markdown(f"""
Proyecto de **Inteligencia Artificial** (UMSA, Carrera de Informática, 2026).

- **Objetivo:** predecir si la temperatura mínima baja de {cfg.UMBRAL_HELADA:g} °C a
  {', '.join(str(h) for h in cfg.HORIZONTES)} días, usando los {cfg.VENTANA} días anteriores.
- **Modelos comparados:** persistencia, regresión logística, MLP y LSTM.
- **Datos:** reanálisis ERA5 de [Open-Meteo](https://open-meteo.com/), {cfg.ANIO_INI}-{cfg.ANIO_FIN}.
- **División temporal:** entrenamiento < {cfg.ANIO_VAL}, validación {cfg.ANIO_VAL}-{cfg.ANIO_TEST - 1},
  prueba ≥ {cfg.ANIO_TEST}.

**Límites:** el reanálisis tiene resolución de 11 a 25 km y no capta microclimas; el umbral de
0 °C es una definición operativa; los resultados de un punto no se extrapolan a otros sin
validar. No sustituye los avisos oficiales de SENAMHI.
""")
