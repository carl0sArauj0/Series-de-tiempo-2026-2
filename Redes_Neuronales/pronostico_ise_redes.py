# %% [markdown]
# # Redes Neuronales para Series de Tiempo: Indicador de Seguimiento a la Economía (ISE)
# ### MLP y LSTM con rezagos de la serie y variables de Fourier
# 
# **Objetivo del cuaderno.** Pronosticar el Indicador de Seguimiento a la Economía (ISE) mensual de Colombia usando redes neuronales: un **perceptrón multicapa (MLP)** y una **red LSTM**. Construimos el flujo completo con `pandas`, `scikit-learn` y `keras`, pronosticando **1 y 2 meses adelante**.
# 
# **Al terminar este cuaderno podrás:**
# 1. Utilizar la ACF, la PACF y el periodograma para identificar la inercia y los ciclos estacionales (anual y semestral).
# 2. Construir variables de Fourier (senos y cosenos) para capturar ciclos determinísticos en series macroeconómicas.
# 3. Transformar la serie temporal en un problema supervisado tabular (para MLP) y de secuencias 3D (para LSTM).
# 4. Implementar el esquema riguroso: **modelo base lineal/elemental → sintonización de hiperparámetros → validación cruzada secuencial → modelo final en prueba**.
# 5. Comparar las tres estrategias de pronóstico a 2 pasos: **recursiva, directa y *one-shot***.
# 6. Poner el modelo en producción y pronosticar los dos meses siguientes al último dato observado (**julio y agosto de 2026**).
# 
# **Hoja de ruta**
# 
# | Sección | Contenido |
# |---|---|
# | 1 | Datos y exploración (ACF, PACF, periodograma) |
# | 2 | Variables de Fourier para los ciclos estacionales (12 y 6 meses) |
# | 3 | De serie a tabla supervisada: rezagos del ISE y de Fourier |
# | 4 | División temporal (entrenamiento / validación / prueba) y estandarización |
# | 5 | Modelo de referencia: persistencia |
# | 6 | MLP: referencia lineal, elemental, búsqueda Keras-Tuner, validación cruzada y evaluación |
# | 7 | LSTM: ventanas temporales, elemental, Hyperband, validación cruzada y evaluación |
# | 8 | Pronóstico a 2 pasos: recursivo, directo y *one-shot* |
# | 9 | Etapa de producción: pronóstico de los 2 meses siguientes |
# | 10 | Comparación final y conclusiones |
# | Anexo | Compendio detallado de funciones |

# %%
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "NONE"

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import tensorflow as tf
import keras
import keras_tuner as kt
import sklearn

tf.get_logger().setLevel("ERROR")
plt.rcParams["figure.figsize"] = (10, 4)
SEMILLA = 123
keras.utils.set_random_seed(SEMILLA)

print(f"tensorflow: {tf.__version__} | keras: {keras.__version__} | "
      f"keras_tuner: {kt.__version__} | scikit-learn: {sklearn.__version__}")

# %% [markdown]
# ### Rutas y Métrica de Evaluación
# Definimos la ruta de la base de datos y la función de evaluación con las métricas en **puntos del ISE**:
# - **RECM** (Raíz del Error Cuadrático Medio): penaliza fuertemente errores atípicos.
# - **EAM** (Error Absoluto Medio): error promedio esperado del pronóstico.

# %%
DIRECTORIO_ACTUAL = Path.cwd()
RUTA_DATOS = DIRECTORIO_ACTUAL.parent / "Bases_de_datos" / "ISE2005_2026.xlsx"
if not RUTA_DATOS.exists():
    RUTA_DATOS = Path("Bases_de_datos") / "ISE2005_2026.xlsx"

print(f"Ruta de datos localizada: {RUTA_DATOS}")

def evaluar(y_real, y_pred):
    return {"RECM": root_mean_squared_error(y_real, y_pred), 
            "EAM": mean_absolute_error(y_real, y_pred)}

# %% [markdown]
# # 1. Datos y exploración
# 
# Cargamos la serie mensual del **Indicador de Seguimiento a la Economía (ISE)**.

# %%
df = pd.read_excel(RUTA_DATOS)
df["Fecha"] = pd.to_datetime(df["Fecha"])
ise = pd.Series(df["Iset"].values, index=pd.DatetimeIndex(df["Fecha"], freq="MS"), name="ise").sort_index()

print(f"Observaciones: {len(ise)} | Desde: {ise.index[0].strftime('%Y-%m')} hasta {ise.index[-1].strftime('%Y-%m')}")
ise.describe().round(2)

# %%
fig, ax = plt.subplots(2, 1, figsize=(11, 6))
ise.plot(ax=ax[0], lw=1.0, color="black", title="Indicador de Seguimiento a la Economía - ISE (2005–2026)", ylabel="Puntos del índice")
ise["2023-01-01":].plot(ax=ax[1], marker="o", color="black", title="Comportamiento reciente (2023–2026)", ylabel="Puntos del índice")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 1.1 ACF y PACF

# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(ise, lags=36, ax=ax[0], title="ACF (36 meses = 3 años)", fft=True)
plot_pacf(ise, lags=36, method="ywm", alpha=0.01, ax=ax[1], title="PACF (36 meses, bandas al 99%)")
plt.tight_layout(); plt.show()

# %%
T = len(ise)
pacf_valores = pacf(ise, nlags=36, method="ywm")
umbral = 2.58 / np.sqrt(T)

tabla_pacf = pd.DataFrame({"PACF": pacf_valores[1:], "significativo": np.abs(pacf_valores[1:]) > umbral},
                          index=pd.RangeIndex(1, 37, name="rezago"))
print(f"Umbral al 99%: {umbral:.4f}")
print("Rezagos significativos entre 1 y 12:", tabla_pacf.loc[1:12].query("significativo").index.tolist())
tabla_pacf.head(12).round(3)

# %% [markdown]
# ## 1.2 Periodograma: ¿qué ciclos existen?

# %%
frecuencias, potencia = signal.periodogram(ise.values)
frecuencias, potencia = frecuencias[1:], potencia[1:]
periodos = 1.0 / frecuencias

plt.figure(figsize=(10, 4))
plt.loglog(periodos, potencia, lw=0.9)
for P, color, etiqueta in [(12, "tab:red", "Ciclo anual (12 meses)"), (6, "tab:orange", "Ciclo semestral (6 meses)")]:
    plt.axvline(P, color=color, ls="--", label=etiqueta)
plt.xlabel("Periodo (meses, escala log)"); plt.ylabel("Potencia (escala log)")
plt.title("Periodograma en función del periodo (meses)")
plt.legend(); plt.show()

# %% [markdown]
# **Decisión de modelación para el ISE:**
# - Rezagos del ISE: 1 a 12 meses (1 año completo de inercia).
# - Variables de Fourier: Periodos de 12 meses (anual) y 6 meses (semestral).

# %% [markdown]
# # 2. Variables de Fourier para los ciclos estacionales

# %%
PERIODOS = {"12m": 12.0, "6m": 6.0}
ORIGEN = pd.Timestamp("2005-01-01")

def variables_fourier(fechas):
    meses = (fechas.year - ORIGEN.year) * 12 + (fechas.month - ORIGEN.month)
    meses = np.asarray(meses, dtype="float32")
    F = pd.DataFrame(index=fechas)
    for nombre, P in PERIODOS.items():
        F[f"sin_{nombre}"] = np.sin(2 * np.pi * meses / P)
        F[f"cos_{nombre}"] = np.cos(2 * np.pi * meses / P)
    return F

fourier = variables_fourier(ise.index)
fourier.head()

# %%
# Regresión lineal usando exclusivamente las componentes de Fourier
ciclos = LinearRegression().fit(fourier, ise)
ajuste_ciclos = pd.Series(ciclos.predict(fourier), index=ise.index)

ax = ise.plot(lw=0.7, color="gray", label="Observado")
ajuste_ciclos.plot(ax=ax, color="tab:red", lw=1.5, label="Ajuste solo con Fourier")
ax.set_title("Los ciclos de Fourier describen el patrón estacional, no la tendencia")
ax.set_ylabel("Puntos del ISE"); ax.legend(); plt.show()
print(f"RECM usando solo ciclos de Fourier: {root_mean_squared_error(ise, ajuste_ciclos):.2f} puntos")

# %% [markdown]
# # 3. De serie a tabla: rezagos del ISE y de Fourier

# %%
VENTANA = 12
LAGS = list(range(VENTANA, 0, -1))
VARIABLES = ["ise"] + list(fourier.columns)
N_VARS = len(VARIABLES)
serie_y_fourier = pd.concat([ise, fourier], axis=1)

columnas = {}
for k in LAGS:
    for v in VARIABLES:
        nombre = f"lag_{k}" if v == "ise" else f"{v}_lag_{k}"
        columnas[nombre] = serie_y_fourier[v].shift(k)

datos = pd.DataFrame(columnas)
datos["y_h1"] = ise               # t
datos["y_h2"] = ise.shift(-1)     # t+1
datos = datos.iloc[VENTANA:-1]    # Remueve bordes con NaN

X_COLS = [c for c in datos.columns if not c.startswith("y_h")]
print(f"Ejemplos supervisados: {len(datos)} | Columnas explicativas: {len(X_COLS)}")

# %% [markdown]
# # 4. División temporal y estandarización

# %%
n = len(datos)
n_train = int(n * 0.70)
n_val = int(n * 0.15)
n_test = n - n_train - n_val

i_train = slice(0, n_train)
i_val = slice(n_train, n_train + n_val)
i_test = slice(n_train + n_val, n)

print(f"Entrenamiento: {datos.index[i_train][0].strftime('%Y-%m')} a {datos.index[i_train][-1].strftime('%Y-%m')}")
print(f"Validación:    {datos.index[i_val][0].strftime('%Y-%m')} a {datos.index[i_val][-1].strftime('%Y-%m')}")
print(f"Prueba:        {datos.index[i_test][0].strftime('%Y-%m')} a {datos.index[i_test][-1].strftime('%Y-%m')}")

# %%
MEDIA = datos["y_h1"].iloc[i_train].mean()
DE = datos["y_h1"].iloc[i_train].std()

def a_ise(z):
    return np.asarray(z) * DE + MEDIA

cols_ise = [c for c in datos.columns if c.startswith("lag_") or c.startswith("y_h")]
datos_z = datos.copy()
datos_z[cols_ise] = (datos[cols_ise] - MEDIA) / DE

X_todo = datos_z[X_COLS].to_numpy("float32")
Y_todo = datos_z[["y_h1", "y_h2"]].to_numpy("float32")

X_train, X_val, X_test = X_todo[i_train], X_todo[i_val], X_todo[i_test]
Y_train, Y_val, Y_test = Y_todo[i_train], Y_todo[i_val], Y_todo[i_test]
X_train_val = np.concatenate([X_train, X_val])
Y_train_val = np.concatenate([Y_train, Y_val])

y_val_h1, y_val_h2 = datos["y_h1"].iloc[i_val], datos["y_h2"].iloc[i_val]
y_test_h1, y_test_h2 = datos["y_h1"].iloc[i_test], datos["y_h2"].iloc[i_test]
F_t = fourier.loc[datos.index].to_numpy("float32")

# %%
def ventana_siguiente(W, y_nuevo_z, F_nuevo):
    paso_nuevo = np.column_stack([y_nuevo_z, F_nuevo])[:, None, :]
    return np.concatenate([W[:, 1:, :], paso_nuevo], axis=1).astype("float32")

def pronostico_recursivo(modelo, X, F_t_bloque, es_lstm):
    W = X.reshape(len(X), VENTANA, N_VARS)
    entrada = (lambda w: w) if es_lstm else (lambda w: w.reshape(len(w), -1))
    p1 = modelo.predict(entrada(W), verbose=0).ravel()
    W_sig = ventana_siguiente(W, p1, F_t_bloque)
    p2 = modelo.predict(entrada(W_sig), verbose=0).ravel()
    return a_ise(p1), a_ise(p2)

def evaluar_2pasos(y_h1, y_h2, p1, p2):
    m1, m2 = evaluar(y_h1, p1), evaluar(y_h2, p2)
    return {"RECM 1 paso": m1["RECM"], "RECM 2 pasos": m2["RECM"],
            "EAM 1 paso": m1["EAM"], "EAM 2 pasos": m2["EAM"]}

def tabla_metricas(filas, conjunto, estrategia="recursiva"):
    tabla = pd.DataFrame(filas).T
    tabla.columns.name = f"{conjunto} (ISE), 2 pasos estrategia {estrategia}" if estrategia else f"{conjunto} (ISE)"
    return tabla

# %% [markdown]
# # 5. Modelo de referencia: persistencia

# %%
pred_persistencia_val = datos["lag_1"].iloc[i_val]
referencias_val = {"persistencia": evaluar_2pasos(y_val_h1, y_val_h2, pred_persistencia_val, pred_persistencia_val)}
tabla_metricas(referencias_val, "validación").round(4)

# %% [markdown]
# # 6. Perceptrón Multicapa (MLP)

# %%
EPOCAS_MAX = 100
LOTE = 16

def crear_mlp(capas=(8,), activacion="relu", tasa=1e-3, n_salidas=1):
    keras.utils.set_random_seed(SEMILLA)
    modelo = keras.Sequential([keras.Input(shape=(VENTANA * N_VARS,))])
    for unidades in capas:
        modelo.add(keras.layers.Dense(unidades, activation=activacion))
    modelo.add(keras.layers.Dense(n_salidas))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=tasa), loss="mse")
    return modelo

def parada_temprana(paciencia=5):
    return keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)

def entrenar_con_validacion(modelo, X_tr, y_tr, X_va, y_va, paciencia=5):
    historia = modelo.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=EPOCAS_MAX, batch_size=LOTE,
                          callbacks=[parada_temprana(paciencia)], verbose=0)
    historia = pd.DataFrame(historia.history)
    historia.index = historia.index + 1
    historia.index.name = "época"
    return historia

def graficar_historia(historia, titulo):
    recm = np.sqrt(historia[["loss", "val_loss"]]) * DE
    ax = recm.rename(columns={"loss": "entrenamiento", "val_loss": "validación"}).plot(marker=".")
    ax.axvline(historia["val_loss"].idxmin(), color="red", ls="--", label=f"mejor época = {historia['val_loss'].idxmin()}")
    ax.set_ylabel("RECM a 1 paso (puntos ISE)"); ax.set_title(titulo); ax.legend(); plt.show()

# %% [markdown]
# ## 6.2 Referencia: MLP lineal

# %%
mlp_lineal = crear_mlp(capas=(32,), activacion="linear")
historia_mlp_lineal = entrenar_con_validacion(mlp_lineal, X_train, Y_train[:, 0], X_val, Y_val[:, 0], paciencia=20)
EPOCAS_MLP_LINEAL = int(historia_mlp_lineal["val_loss"].idxmin())
graficar_historia(historia_mlp_lineal, "MLP lineal (1x32): curvas de aprendizaje")

pred_val_lineal = pronostico_recursivo(mlp_lineal, X_val, F_t[i_val], es_lstm=False)
referencias_val["MLP lineal 1x32"] = evaluar_2pasos(y_val_h1, y_val_h2, *pred_val_lineal)
tabla_metricas(referencias_val, "validación").round(4)

# %% [markdown]
# ## 6.3 MLP elemental

# %%
mlp_elemental = crear_mlp(capas=(8,))
historia_mlp_elemental = entrenar_con_validacion(mlp_elemental, X_train, Y_train[:, 0], X_val, Y_val[:, 0])
EPOCAS_MLP_ELEMENTAL = int(historia_mlp_elemental["val_loss"].idxmin())
graficar_historia(historia_mlp_elemental, "MLP elemental (1x8 relu): curvas de aprendizaje")

pred_val_mlp_elemental = pronostico_recursivo(mlp_elemental, X_val, F_t[i_val], es_lstm=False)
metricas_val = {**referencias_val, "MLP elemental": evaluar_2pasos(y_val_h1, y_val_h2, *pred_val_mlp_elemental)}
tabla_metricas(metricas_val, "validación").round(4)

# %% [markdown]
# ## 6.4 Búsqueda de hiperparámetros con Keras-Tuner (Carga desde carpeta existente)
# Al configurar `overwrite=False`, Keras-Tuner lee directamente los resultados existentes en la carpeta `busqueda_hiperparametros` sin volver a entrenar.

# %%
def construir_mlp(hp):
    n_capas = hp.Choice("n_capas", [1, 2])
    unidades = hp.Choice("unidades", [8, 16, 32])
    activacion = hp.Choice("activacion", ["relu", "tanh"])
    tasa = hp.Choice("tasa", [1e-3, 1e-2])
    return crear_mlp(capas=(unidades,) * n_capas, activacion=activacion, tasa=tasa)

dir_busqueda = DIRECTORIO_ACTUAL / "busqueda_hiperparametros"
buscador_mlp = kt.GridSearch(construir_mlp, objective="val_loss", seed=SEMILLA,
                             overwrite=False, directory=str(dir_busqueda), project_name="mlp_ise")

# Si ya existen resultados, este search finaliza de inmediato
buscador_mlp.search(X_train, Y_train[:, 0], validation_data=(X_val, Y_val[:, 0]), epochs=EPOCAS_MAX,
                    batch_size=LOTE, callbacks=[parada_temprana(4)], verbose=0)

def tabla_busqueda(buscador):
    filas = []
    for ensayo in buscador.oracle.get_best_trials(num_trials=1000):
        hp = {k: v for k, v in ensayo.hyperparameters.values.items() if not k.startswith("tuner/")}
        filas.append({**hp, "RECM validación 1 paso": np.sqrt(ensayo.score) * DE})
    tabla = pd.DataFrame(filas)
    return tabla.drop_duplicates(subset=list(hp.keys()), keep="first").reset_index(drop=True)

resultados_mlp = tabla_busqueda(buscador_mlp)
resultados_mlp.head(8).round(4)

# %% [markdown]
# ## 6.5 Validación cruzada secuencial MLP

# %%
N_PLIEGUES, TAM_PLIEGUE = 4, 20
cv_secuencial = TimeSeriesSplit(n_splits=N_PLIEGUES, test_size=TAM_PLIEGUE)
pliegues = list(cv_secuencial.split(X_train))

def recm_pliegues(ajustar_y_predecir, X, y):
    recm = []
    for i_tr, i_va in pliegues:
        pred = ajustar_y_predecir(X[i_tr], y[i_tr], X[i_va])
        recm.append(root_mean_squared_error(a_ise(y[i_va]), a_ise(pred)))
    return recm

def red_con_epocas(crear, epocas):
    def ajustar_y_predecir(X_tr, y_tr, X_va):
        modelo = crear()
        modelo.fit(X_tr, y_tr, epochs=epocas, batch_size=LOTE, verbose=0)
        return modelo.predict(X_va, verbose=0).ravel()
    return ajustar_y_predecir

def persistencia_z(X_tr, y_tr, X_va):
    return X_va[:, X_COLS.index("lag_1")]

def tabla_cv(resultados):
    filas = {}
    for nombre, r in resultados.items():
        fila = {f"pliegue {k + 1}": v for k, v in enumerate(r["pliegues"])}
        fila["RECM_cv 1 paso"] = np.sqrt(np.mean(np.square(r["pliegues"])))
        fila.update({k: v for k, v in r.items() if k != "pliegues"})
        filas[nombre] = fila
    tabla = pd.DataFrame(filas).T.astype(float).sort_values("RECM_cv 1 paso")
    tabla.columns.name = "CV secuencial 1 paso (ISE)"
    return tabla

cv_referencias = {
    "persistencia": {"pliegues": recm_pliegues(persistencia_z, X_train, Y_train[:, 0])},
    "MLP lineal 1x32": {"pliegues": recm_pliegues(red_con_epocas(lambda: crear_mlp(capas=(32,), activacion="linear"),
                                                                 EPOCAS_MLP_LINEAL), X_train, Y_train[:, 0]),
                        "épocas": EPOCAS_MLP_LINEAL}
}

def describir_mlp(hp):
    return f"MLP {int(hp['n_capas'])}x{int(hp['unidades'])} {hp['activacion']}, tasa={hp['tasa']}"

candidatos_mlp = {"MLP elemental 1x8 relu, tasa=0.001": dict(capas=(8,), activacion="relu", tasa=1e-3)}
for _, hp in resultados_mlp.head(3).iterrows():
    candidatos_mlp[describir_mlp(hp)] = dict(capas=(int(hp["unidades"]),) * int(hp["n_capas"]),
                                             activacion=hp["activacion"], tasa=hp["tasa"])

cv_mlp = {}
for nombre, config in candidatos_mlp.items():
    modelo = crear_mlp(**config)
    historia = entrenar_con_validacion(modelo, X_train, Y_train[:, 0], X_val, Y_val[:, 0])
    epocas = int(historia["val_loss"].idxmin())
    cv_mlp[nombre] = {"pliegues": recm_pliegues(red_con_epocas(lambda: crear_mlp(**config), epocas), X_train, Y_train[:, 0]),
                      "RECM validación 1 paso": np.sqrt(historia["val_loss"].min()) * DE,
                      "épocas": epocas}

resultados_cv_mlp = tabla_cv({**cv_mlp, **cv_referencias})
resultados_cv_mlp.round(4)

# %%
nombre_mlp = resultados_cv_mlp.drop(index=list(cv_referencias)).index[0]
CONFIG_MLP = candidatos_mlp[nombre_mlp]
EPOCAS_MLP = int(resultados_cv_mlp.loc[nombre_mlp, "épocas"])
print(f"MLP seleccionado: {nombre_mlp} | Épocas óptimas: {EPOCAS_MLP}")

# %% [markdown]
# ## 6.6 MLP: Modelo final en prueba

# %%
mlp_final = crear_mlp(**CONFIG_MLP)
mlp_final.fit(X_train_val, Y_train_val[:, 0], epochs=EPOCAS_MLP, batch_size=LOTE, verbose=0)
pred_test_mlp, pred_test_mlp_h2 = pronostico_recursivo(mlp_final, X_test, F_t[i_test], es_lstm=False)

mlp_lineal_final = crear_mlp(capas=(32,), activacion="linear")
mlp_lineal_final.fit(X_train_val, Y_train_val[:, 0], epochs=EPOCAS_MLP_LINEAL, batch_size=LOTE, verbose=0)
pred_test_lineal = pronostico_recursivo(mlp_lineal_final, X_test, F_t[i_test], es_lstm=False)

pred_test_persistencia = datos["lag_1"].iloc[i_test]
prueba = {
    "persistencia": evaluar_2pasos(y_test_h1, y_test_h2, pred_test_persistencia, pred_test_persistencia),
    "MLP lineal 1x32": evaluar_2pasos(y_test_h1, y_test_h2, *pred_test_lineal),
    f"MLP final ({nombre_mlp})": evaluar_2pasos(y_test_h1, y_test_h2, pred_test_mlp, pred_test_mlp_h2)
}
tabla_metricas(prueba, "prueba").round(4)

# %% [markdown]
# # 7. Red LSTM

# %%
def a_ventanas(X):
    return X.reshape(len(X), VENTANA, N_VARS)

W_train, W_val, W_test = a_ventanas(X_train), a_ventanas(X_val), a_ventanas(X_test)
W_train_val = a_ventanas(X_train_val)
print("Forma de ventanas LSTM:", W_train.shape)

def crear_lstm(capas=(8,), tasa=1e-3, n_salidas=1):
    keras.utils.set_random_seed(SEMILLA)
    modelo = keras.Sequential([keras.Input(shape=(VENTANA, N_VARS))])
    for j, unidades in enumerate(capas):
        modelo.add(keras.layers.LSTM(unidades, return_sequences=(j < len(capas) - 1)))
    modelo.add(keras.layers.Dense(n_salidas))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=tasa), loss="mse")
    return modelo

# %% [markdown]
# ## 7.2 LSTM elemental

# %%
lstm_elemental = crear_lstm(capas=(8,))
historia_lstm_elemental = entrenar_con_validacion(lstm_elemental, W_train, Y_train[:, 0], W_val, Y_val[:, 0])
EPOCAS_LSTM_ELEMENTAL = int(historia_lstm_elemental["val_loss"].idxmin())
graficar_historia(historia_lstm_elemental, "LSTM elemental (1x8): curvas de aprendizaje")

pred_val_lstm_elemental = pronostico_recursivo(lstm_elemental, W_val, F_t[i_val], es_lstm=True)
metricas_val["LSTM elemental"] = evaluar_2pasos(y_val_h1, y_val_h2, *pred_val_lstm_elemental)
tabla_metricas(metricas_val, "validación").round(4)

# %% [markdown]
# ## 7.3 Búsqueda de hiperparámetros LSTM con Hyperband (Carga desde carpeta existente)

# %%
def construir_lstm(hp):
    n_capas = hp.Choice("n_capas", [1, 2])
    unidades = hp.Choice("unidades", [8, 16, 32])
    tasa = hp.Choice("tasa", [1e-3, 1e-2])
    return crear_lstm(capas=(unidades,) * n_capas, tasa=tasa)

buscador_lstm = kt.Hyperband(construir_lstm, objective="val_loss", max_epochs=20, factor=3, seed=SEMILLA,
                             overwrite=False, directory=str(dir_busqueda), project_name="lstm_ise")

buscador_lstm.search(W_train, Y_train[:, 0], validation_data=(W_val, Y_val[:, 0]), batch_size=LOTE,
                     callbacks=[parada_temprana(4)], verbose=0)

resultados_lstm = tabla_busqueda(buscador_lstm)
resultados_lstm.head(6).round(4)

# %% [markdown]
# ## 7.4 Validación cruzada secuencial LSTM

# %%
def describir_lstm(hp):
    return f"LSTM {int(hp['n_capas'])}x{int(hp['unidades'])}, tasa={hp['tasa']}"

candidatos_lstm = {"LSTM elemental 1x8, tasa=0.001": dict(capas=(8,), tasa=1e-3)}
for _, hp in resultados_lstm.head(2).iterrows():
    candidatos_lstm[describir_lstm(hp)] = dict(capas=(int(hp["unidades"]),) * int(hp["n_capas"]), tasa=hp["tasa"])

cv_lstm = {}
for nombre, config in candidatos_lstm.items():
    modelo = crear_lstm(**config)
    historia = entrenar_con_validacion(modelo, W_train, Y_train[:, 0], W_val, Y_val[:, 0])
    epocas = int(historia["val_loss"].idxmin())
    cv_lstm[nombre] = {"pliegues": recm_pliegues(red_con_epocas(lambda: crear_lstm(**config), epocas), W_train, Y_train[:, 0]),
                       "RECM validación 1 paso": np.sqrt(historia["val_loss"].min()) * DE,
                       "épocas": epocas}

resultados_cv_lstm = tabla_cv({**cv_lstm, **cv_referencias})
resultados_cv_lstm.round(4)

# %%
nombre_lstm = resultados_cv_lstm.drop(index=list(cv_referencias)).index[0]
CONFIG_LSTM = candidatos_lstm[nombre_lstm]
EPOCAS_LSTM = int(resultados_cv_lstm.loc[nombre_lstm, "épocas"])
print(f"LSTM seleccionada: {nombre_lstm} | Épocas óptimas: {EPOCAS_LSTM}")

# %% [markdown]
# ## 7.5 LSTM: Modelo final en prueba

# %%
lstm_final = crear_lstm(**CONFIG_LSTM)
lstm_final.fit(W_train_val, Y_train_val[:, 0], epochs=EPOCAS_LSTM, batch_size=LOTE, verbose=0)
pred_test_lstm, pred_test_lstm_h2 = pronostico_recursivo(lstm_final, W_test, F_t[i_test], es_lstm=True)

prueba[f"LSTM final ({nombre_lstm})"] = evaluar_2pasos(y_test_h1, y_test_h2, pred_test_lstm, pred_test_lstm_h2)
tabla_metricas(prueba, "prueba").round(4)

# %% [markdown]
# # 8. Pronóstico 2 pasos adelante: recursivo, directo y *one-shot*

# %%
pronosticos = {}

# 8.1 Estrategia recursiva
pronosticos[("MLP", "recursiva")] = (pred_test_mlp, pred_test_mlp_h2)
pronosticos[("LSTM", "recursiva")] = (pred_test_lstm, pred_test_lstm_h2)

# 8.2 Estrategia directa
mlp_h2 = crear_mlp(**CONFIG_MLP)
mlp_h2.fit(X_train_val, Y_train_val[:, 1], epochs=EPOCAS_MLP, batch_size=LOTE, verbose=0)
pronosticos[("MLP", "directa")] = (pred_test_mlp, a_ise(mlp_h2.predict(X_test, verbose=0).ravel()))

lstm_h2 = crear_lstm(**CONFIG_LSTM)
lstm_h2.fit(W_train_val, Y_train_val[:, 1], epochs=EPOCAS_LSTM, batch_size=LOTE, verbose=0)
pronosticos[("LSTM", "directa")] = (pred_test_lstm, a_ise(lstm_h2.predict(W_test, verbose=0).ravel()))

# 8.3 Estrategia one-shot (MIMO)
mlp_2s = crear_mlp(**CONFIG_MLP, n_salidas=2)
mlp_2s.fit(X_train_val, Y_train_val, epochs=EPOCAS_MLP, batch_size=LOTE, verbose=0)
p_mlp_2s = mlp_2s.predict(X_test, verbose=0)
pronosticos[("MLP", "one-shot")] = (a_ise(p_mlp_2s[:, 0]), a_ise(p_mlp_2s[:, 1]))

lstm_2s = crear_lstm(**CONFIG_LSTM, n_salidas=2)
lstm_2s.fit(W_train_val, Y_train_val, epochs=EPOCAS_LSTM, batch_size=LOTE, verbose=0)
p_lstm_2s = lstm_2s.predict(W_test, verbose=0)
pronosticos[("LSTM", "one-shot")] = (a_ise(p_lstm_2s[:, 0]), a_ise(p_lstm_2s[:, 1]))

# %%
filas_h2 = {("Referencia", "persistencia"): evaluar_2pasos(y_test_h1, y_test_h2, pred_test_persistencia, pred_test_persistencia)}
for (red, est), (p1, p2) in pronosticos.items():
    filas_h2[(red, est)] = evaluar_2pasos(y_test_h1, y_test_h2, p1, p2)

comparacion_2pasos = tabla_metricas(filas_h2, "prueba", estrategia=None)
comparacion_2pasos.index.names = ["red", "estrategia"]
comparacion_2pasos.round(4)

# %%
fechas_h2 = datos.index[i_test] + pd.DateOffset(months=1)
fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for a, red in zip(ax, ["MLP", "LSTM"]):
    pd.Series(y_test_h2.values, index=fechas_h2).plot(ax=a, color="black", marker=".", label="Observado")
    for est, color in [("recursiva", "tab:red"), ("directa", "tab:blue"), ("one-shot", "tab:green")]:
        pd.Series(pronosticos[(red, est)][1], index=fechas_h2).plot(ax=a, color=color, label=est)
    a.set_title(f"{red}: Pronóstico a 2 meses adelante ($y_{{t+1}}$)")
    a.set_ylabel("Puntos del ISE"); a.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# # 9. Etapa de producción: pronóstico de los 2 meses siguientes
# Se elige el modelo con menor RECM en validación cruzada y se reentrena con el 100% de los datos observados para predecir **julio y agosto de 2026**.

# %%
recm_cv_mlp = resultados_cv_mlp.loc[nombre_mlp, "RECM_cv 1 paso"]
recm_cv_lstm = resultados_cv_lstm.loc[nombre_lstm, "RECM_cv 1 paso"]
RED_PROD = "MLP" if recm_cv_mlp <= recm_cv_lstm else "LSTM"
print(f"RECM_cv MLP: {recm_cv_mlp:.4f} | LSTM: {recm_cv_lstm:.4f} -> Red de producción seleccionada: {RED_PROD}")

if RED_PROD == "MLP":
    crear, config, epocas, entrada = crear_mlp, CONFIG_MLP, EPOCAS_MLP, (lambda w: w.reshape(len(w), -1))
else:
    crear, config, epocas, entrada = crear_lstm, CONFIG_LSTM, EPOCAS_LSTM, (lambda w: w)

W_todo = a_ventanas(X_todo)
prod_h1 = crear(**config); prod_h1.fit(entrada(W_todo), Y_todo[:, 0], epochs=epocas, batch_size=LOTE, verbose=0)
prod_h2 = crear(**config); prod_h2.fit(entrada(W_todo), Y_todo[:, 1], epochs=epocas, batch_size=LOTE, verbose=0)
prod_2s = crear(**config, n_salidas=2); prod_2s.fit(entrada(W_todo), Y_todo, epochs=epocas, batch_size=LOTE, verbose=0)

fechas_pronostico = pd.date_range(ise.index[-1] + pd.DateOffset(months=1), periods=2, freq="MS")
print(f"Último dato observado: {ise.index[-1].strftime('%Y-%m')} | Pronosticando: {[f.strftime('%Y-%m') for f in fechas_pronostico]}")

ultimas = serie_y_fourier.iloc[-VENTANA:].copy()
ultimas["ise"] = (ultimas["ise"] - MEDIA) / DE
W_nueva = ultimas[VARIABLES].to_numpy("float32")[None, :, :]
F_nueva = variables_fourier(fechas_pronostico[:1]).to_numpy("float32")

p_rec = pronostico_recursivo(prod_h1, W_nueva, F_nueva, es_lstm=(RED_PROD == "LSTM"))
p_dir = (a_ise(prod_h1.predict(entrada(W_nueva), verbose=0).ravel()),
         a_ise(prod_h2.predict(entrada(W_nueva), verbose=0).ravel()))
p_2s = a_ise(prod_2s.predict(entrada(W_nueva), verbose=0))[0]

pronostico_produccion = pd.DataFrame({
    "recursiva": [p_rec[0][0], p_rec[1][0]],
    "directa": [p_dir[0][0], p_dir[1][0]],
    "one-shot": [p_2s[0], p_2s[1]]
}, index=fechas_pronostico)
pronostico_produccion.index.name = "Mes pronosticado"
pronostico_produccion.round(3)

# %%
fig, ax = plt.subplots(figsize=(11, 4))
ise.iloc[-24:].plot(ax=ax, color="black", marker=".", label="Últimos 24 meses observados")
for estrategia, color, marca in [("recursiva", "tab:red", "o"), ("directa", "tab:blue", "s"), ("one-shot", "tab:green", "^")]:
    serie_p = pd.concat([ise.iloc[-1:], pronostico_produccion[estrategia]])
    serie_p.plot(ax=ax, color=color, marker=marca, ls="--", label=estrategia)
ax.set_title(f"Producción ({RED_PROD}): Pronóstico del ISE para {[f.strftime('%Y-%m') for f in fechas_pronostico]}")
ax.set_ylabel("Puntos del ISE"); ax.legend(fontsize=8); plt.show()

# %% [markdown]
# # 10. Comparación final y conclusiones

# %%
mlp_elemental_final = crear_mlp(capas=(8,))
mlp_elemental_final.fit(X_train_val, Y_train_val[:, 0], epochs=EPOCAS_MLP_ELEMENTAL, batch_size=LOTE, verbose=0)
lstm_elemental_final = crear_lstm(capas=(8,))
lstm_elemental_final.fit(W_train_val, Y_train_val[:, 0], epochs=EPOCAS_LSTM_ELEMENTAL, batch_size=LOTE, verbose=0)

prueba["MLP elemental"] = evaluar_2pasos(y_test_h1, y_test_h2, *pronostico_recursivo(mlp_elemental_final, X_test, F_t[i_test], es_lstm=False))
prueba["LSTM elemental"] = evaluar_2pasos(y_test_h1, y_test_h2, *pronostico_recursivo(lstm_elemental_final, W_test, F_t[i_test], es_lstm=True))

comparacion_modelos = tabla_metricas(prueba, "prueba").sort_values("RECM 1 paso")
for h in ["1 paso", "2 pasos"]:
    comparacion_modelos[f"mejora RECM {h} (%)"] = 100 * (1 - comparacion_modelos[f"RECM {h}"] / comparacion_modelos.loc["persistencia", f"RECM {h}"])
comparacion_modelos.round(4)

# %%
ax = comparacion_modelos[["RECM 1 paso", "RECM 2 pasos"]].iloc[::-1].plot.barh(color=["tab:blue", "tab:orange"])
for barras in ax.containers:
    ax.bar_label(barras, fmt="%.3f", padding=3, fontsize=8)
ax.set_xlabel("RECM en prueba (Puntos ISE)"); ax.set_title("Comparación final en Prueba (menor es mejor)")
ax.set_xlim(0, comparacion_modelos["RECM 2 pasos"].max() * 1.25); ax.legend(loc="lower right"); plt.show()