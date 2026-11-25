# Alerta temprana de heladas en el Altiplano paceño con redes neuronales

Proyecto de Inteligencia Artificial — Carrera de Informática, Universidad Mayor de San Andrés (2026).

Predice si habrá helada (temperatura mínima < 0 °C) a 1, 2 y 3 días de anticipación en puntos del
departamento de La Paz. Compara una línea base (regresión logística), la regla de persistencia,
un perceptrón multicapa (MLP) y una red LSTM, usando ventanas de los 7 días anteriores.

## Datos

Reanálisis ERA5 vía [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api), 1990–2025,
9 puntos del departamento de La Paz (6 del Altiplano y 3 de contraste).
Los datos **no** se incluyen en el repositorio: se descargan con `descargar_datos.py`.
Open-Meteo es gratuito solo para uso no comercial y requiere atribución.

## Instalación

```bash
git clone https://github.com/TU_USUARIO/alerta-heladas-altiplano.git
cd alerta-heladas-altiplano
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Se recomienda Python 3.10 o 3.11.

## Uso

```bash
python descargar_datos.py                      # genera datos_heladas_diarios.csv (tarda varios minutos)
python entrenar.py                             # entrena, guarda modelo, escalador, umbrales y tablas de resultados
python predecir.py ultimos_7_dias_ejemplo.csv  # aviso de helada a 1, 2 y 3 días
```

`ultimos_7_dias_ejemplo.csv` es un archivo de ejemplo con valores de muestra; reemplázalo con
7 días reales medidos (columnas: fecha, tmin, tmax, tmean, hr, td, presion, nubes, viento,
radiacion, elevacion).

## Dashboard

```bash
streamlit run dashboard.py
```

Se abre en `http://localhost:8501` con cinco pestañas: alerta actual por punto (datos recientes
de Open-Meteo), resultados de los modelos, análisis por punto y mes, prueba con un CSV propio y
descripción del proyecto. Requiere haber ejecutado antes `python entrenar.py`.

## Estructura

| Archivo | Función |
|---|---|
| `config.py` | Parámetros del proyecto (rutas, puntos, años, ventana, horizontes) |
| `descargar_datos.py` | Descarga datos horarios de Open-Meteo y los resume por día |
| `preparacion.py` | Etiquetas de helada, ventanas de 7 días, escalado y división por fecha |
| `lineas_base.py` | Persistencia y regresión logística |
| `metricas.py` | Umbral de decisión, precisión, exhaustividad, F1 y AUC |
| `modelos.py` | Definición, entrenamiento y carga del MLP y la LSTM (usa TensorFlow) |
| `entrenar.py` | Programa principal: compara los modelos y guarda resultados |
| `predecir.py` | Aviso de helada a partir de los últimos 7 días |
| `dashboard.py` | Dashboard web con Streamlit |

## Uso en Google Colab

```python
!git clone https://github.com/TU_USUARIO/alerta-heladas-altiplano.git
%cd alerta-heladas-altiplano
!pip install -r requirements.txt
!python descargar_datos.py
!python entrenar.py
```

## Límites

- Open-Meteo entrega reanálisis (un modelo del clima), no mediciones de estación; la resolución
  de 11 a 25 km no capta microclimas.
- El umbral de 0 °C es una definición operativa.
- Los resultados de un punto no se extrapolan a otros sin validar.
- No sustituye los avisos oficiales de SENAMHI.

## Autor

TU NOMBRE — Materia: Inteligencia Artificial — Docente: Miguel Toledo Paz

## Licencia

MIT (ver `LICENSE`).
