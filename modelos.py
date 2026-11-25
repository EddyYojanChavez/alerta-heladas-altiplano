"""Redes neuronales (MLP y LSTM) con Keras. Es el unico modulo que usa TensorFlow."""
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config as cfg


def fijar_semilla(semilla):
    """Hace los resultados repetibles."""
    random.seed(semilla)
    np.random.seed(semilla)
    tf.random.set_seed(semilla)


def perdida_ponderada(pesos):
    """Entropia cruzada binaria donde acertar la helada pesa mas (clase rara)."""
    pesos = tf.constant(pesos, dtype=tf.float32)

    def perdida(y, p):
        p = tf.clip_by_value(p, 1e-7, 1 - 1e-7)
        ce = -(pesos * y * tf.math.log(p) + (1 - y) * tf.math.log(1 - p))
        return tf.reduce_mean(ce)

    return perdida


def construir_red(tipo):
    """Arma la red. Entrada: (VENTANA, variables). Salida: un valor por horizonte."""
    entrada = keras.Input(shape=(cfg.VENTANA, len(cfg.ENTRADAS)))
    if tipo == "mlp":
        x = layers.Flatten()(entrada)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
    elif tipo == "lstm":
        x = layers.LSTM(64)(entrada)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation="relu")(x)
    else:
        raise ValueError(f"Tipo de red desconocido: {tipo}")
    salida = layers.Dense(len(cfg.HORIZONTES), activation="sigmoid")(x)
    return keras.Model(entrada, salida, name=tipo)


def entrenar_red(tipo, conjuntos, pesos, semilla):
    """Entrena una red y devuelve (modelo, p_val, p_test).

    Se detiene antes si la perdida de validacion deja de mejorar.
    """
    fijar_semilla(semilla)
    modelo = construir_red(tipo)
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss=perdida_ponderada(pesos))
    parada = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.PACIENCIA, restore_best_weights=True)
    modelo.fit(conjuntos["train"].x, conjuntos["train"].y,
               validation_data=(conjuntos["val"].x, conjuntos["val"].y),
               epochs=cfg.EPOCAS, batch_size=cfg.TAMANO_LOTE,
               callbacks=[parada], verbose=0)
    p_val = modelo.predict(conjuntos["val"].x, verbose=0)
    p_test = modelo.predict(conjuntos["test"].x, verbose=0)
    return modelo, p_val, p_test


def guardar_red(modelo, ruta):
    modelo.save(ruta)


def cargar_red(ruta):
    """compile=False: para predecir no hace falta la perdida personalizada."""
    return keras.models.load_model(ruta, compile=False)
