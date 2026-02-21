# ==============================================================================
# Regresion Lineal y Logistica - Dataset de Peliculas
#
# DESCRIPCION GENERAL:
#   Este script implementa dos tipos de regresion sobre el dataset de peliculas:
#
#   Regresion Lineal   : predice la columna RATING (puntaje de la pelicula)
#                        usando Ridge y Lasso con busqueda aleatoria de
#                        hiperparametros y validacion cruzada.
#
#   Regresion Logistica: clasifica si una pelicula es "exitosa" o no, definido
#                        como tener un RATING superior a la mediana del dataset.
# ==============================================================================


import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from scipy.stats import loguniform

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RUTA_DATASET               = "movies.csv"
CARPETA_SALIDA             = "graficos_regresion"
COLUMNA_OBJETIVO_LINEAL    = "RATING"
COLUMNA_OBJETIVO_LOGISTICA = "exitosa"
COLUMNA_FEATURE_PLOT       = "RunTime"

os.makedirs(CARPETA_SALIDA, exist_ok=True)


# ==============================================================================
# SECCION 0: CARGA Y PREPARACION DE DATOS
# ==============================================================================

print("=" * 60)
print("0. CARGA Y PREPARACION DE DATOS")
print("=" * 60)

df = pd.read_csv(RUTA_DATASET)
print(f"Dimensiones originales: {df.shape[0]} filas x {df.shape[1]} columnas")

# Eliminar filas duplicadas
df.drop_duplicates(inplace=True)

# Convertir VOTES a numerico antes de cualquier otra operacion.
# Esta columna tiene comas como separador de miles (ej. "21,062"), por lo que
# pandas la carga como texto. Se eliminan las comas y se convierte con to_numeric.
# errors="coerce" transforma los valores no parseables en NaN para manejarlos
# despues en la imputacion, en lugar de lanzar un error.
df["VOTES"] = pd.to_numeric(df["VOTES"].str.replace(",", ""), errors="coerce")

# Limpiar espacios y saltos de linea en columnas de texto.
# El dataset tiene celdas con "\n" y espacios extra en GENRE, ONE-LINE y STARS.
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Detectar columnas por tipo de dato DESPUES de la conversion numerica.
# Ahora VOTES ya aparece como numerica y sera cubierta por la imputacion.
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"Columnas numericas   : {num_cols}")
print(f"Columnas categoricas : {cat_cols}")

# Imputar valores nulos en columnas numericas con la mediana.
# La mediana es robusta ante outliers y preferida sobre la media en este caso.
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Imputar valores nulos en columnas categoricas con la moda (valor mas frecuente).
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Verificar que no queden nulos antes de continuar
print(f"Nulos restantes: {df.isnull().sum().sum()}")

# Codificar columnas categoricas con Label Encoding para que el modelo pueda usarlas.
# Se crea una nueva columna con sufijo "_le" para no perder los valores originales.
le = LabelEncoder()
for col in cat_cols:
    df[col + "_le"] = le.fit_transform(df[col].astype(str))

# Crear la variable binaria "exitosa" para la regresion logistica.
# Una pelicula se considera exitosa si su RATING supera la mediana del dataset.
# Esto garantiza clases aproximadamente balanceadas (50% en cada clase).
mediana_rating = df[COLUMNA_OBJETIVO_LINEAL].median()
df[COLUMNA_OBJETIVO_LOGISTICA] = (df[COLUMNA_OBJETIVO_LINEAL] > mediana_rating).astype(int)

print(f"\nVariable '{COLUMNA_OBJETIVO_LOGISTICA}' creada (RATING > mediana={mediana_rating}).")
print(f"Distribucion de clases:\n{df[COLUMNA_OBJETIVO_LOGISTICA].value_counts()}")
print(f"\nDataset preparado: {df.shape[0]} filas x {df.shape[1]} columnas")


# ==============================================================================
# SECCION 1: REGRESION LINEAL (RIDGE Y LASSO)
# ==============================================================================

print("\n" + "=" * 60)
print("1. REGRESION LINEAL (RIDGE Y LASSO)")
print("=" * 60)

# Seleccionar features: columnas numericas excluyendo la objetivo y la binaria
features_excluir = [COLUMNA_OBJETIVO_LINEAL, COLUMNA_OBJETIVO_LOGISTICA]
feature_cols = [
    col for col in df.select_dtypes(include=[np.number]).columns
    if col not in features_excluir
]

X = df[feature_cols].values
y = df[COLUMNA_OBJETIVO_LINEAL].values

print(f"Features utilizadas : {feature_cols}")
print(f"Columna objetivo    : {COLUMNA_OBJETIVO_LINEAL}")
print(f"Tamano del dataset  : {X.shape}")

# Dividir en entrenamiento (80%) y prueba (20%).
# random_state=42 garantiza reproducibilidad de la division.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nEntrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba       : {X_test.shape[0]} muestras")

# Graficar dispersion de entrenamiento vs prueba usando RunTime como eje X.
# Permite verificar visualmente que ambos conjuntos tienen distribucion similar.
if COLUMNA_FEATURE_PLOT in feature_cols:
    idx_feat = feature_cols.index(COLUMNA_FEATURE_PLOT)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train[:, idx_feat], y_train, color="steelblue", alpha=0.5, s=20, label="Entrenamiento")
    plt.scatter(X_test[:, idx_feat], y_test, color="salmon", alpha=0.7, s=20, label="Prueba")
    plt.xlabel(COLUMNA_FEATURE_PLOT)
    plt.ylabel(COLUMNA_OBJETIVO_LINEAL)
    plt.title("Conjunto de entrenamiento vs prueba")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CARPETA_SALIDA}/train_vs_test_lineal.png", bbox_inches="tight")
    plt.close()
    print(f"\n-> Grafico guardado: {CARPETA_SALIDA}/train_vs_test_lineal.png")

# Definir los pipelines para Ridge y Lasso.
# Usar Pipeline garantiza que el scaler se ajuste solo con datos de entrenamiento,
# evitando data leakage (filtracion de informacion del conjunto de prueba).
pipeline_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge()),
])

pipeline_lasso = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=10000)),
])

# Definir la distribucion de hiperparametros para la busqueda aleatoria.
# loguniform genera valores en escala logaritmica, adecuado para alpha porque
# su efecto sobre la regularizacion es multiplicativo, no aditivo.
param_dist = {"model__alpha": loguniform(1e-3, 1e3)}

# Aplicar busqueda aleatoria con validacion cruzada (RandomizedSearchCV).
# n_iter=50  : probar 50 combinaciones aleatorias de hiperparametros.
# cv=5       : usar 5 folds de validacion cruzada en cada combinacion.
# scoring=r2 : usar el coeficiente R2 como metrica de evaluacion.
# n_jobs=-1  : usar todos los nucleos del procesador para paralelizar.
print("\nBuscando mejores hiperparametros para Ridge...")
search_ridge = RandomizedSearchCV(
    pipeline_ridge, param_dist,
    n_iter=50, cv=5, scoring="r2", random_state=42, n_jobs=-1
)
search_ridge.fit(X_train, y_train)

print("Buscando mejores hiperparametros para Lasso...")
search_lasso = RandomizedSearchCV(
    pipeline_lasso, param_dist,
    n_iter=50, cv=5, scoring="r2", random_state=42, n_jobs=-1
)
search_lasso.fit(X_train, y_train)

print(f"\nMejores parametros Ridge : {search_ridge.best_params_}")
print(f"Mejores parametros Lasso : {search_lasso.best_params_}")

# Generar predicciones sobre el conjunto de prueba con los mejores modelos
y_pred_ridge = search_ridge.best_estimator_.predict(X_test)
y_pred_lasso = search_lasso.best_estimator_.predict(X_test)

# Calcular metricas de evaluacion:
#   R2  : que tan bien explica el modelo la varianza total (1.0 es perfecto).
#   MAE : error promedio absoluto en las mismas unidades de RATING (puntos).
r2_ridge  = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_lasso  = r2_score(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print("\nResultados en conjunto de prueba:")
print(f"  Ridge  ->  R2: {r2_ridge:.4f}  |  MAE: {mae_ridge:.4f}")
print(f"  Lasso  ->  R2: {r2_lasso:.4f}  |  MAE: {mae_lasso:.4f}")

# Graficar valores reales vs valores predichos para Ridge y Lasso.
# En un modelo perfecto todos los puntos estarian sobre la linea diagonal y=x.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_pred, nombre, r2, mae in zip(
    axes,
    [y_pred_ridge, y_pred_lasso],
    ["Ridge", "Lasso"],
    [r2_ridge, r2_lasso],
    [mae_ridge, mae_lasso],
):
    ax.scatter(y_test, y_pred, alpha=0.4, s=15, color="steelblue", edgecolors="white")
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Prediccion perfecta")
    ax.set_xlabel(f"Valor real ({COLUMNA_OBJETIVO_LINEAL})")
    ax.set_ylabel("Valor predicho")
    ax.set_title(f"{nombre}  |  R2={r2:.3f}  MAE={mae:.3f}")
    ax.legend()

plt.suptitle("Regresion Lineal: valores reales vs predichos", fontsize=13)
plt.tight_layout()
plt.savefig(f"{CARPETA_SALIDA}/predicciones_lineal.png", bbox_inches="tight")
plt.close()
print(f"-> Grafico guardado: {CARPETA_SALIDA}/predicciones_lineal.png")


# ==============================================================================
# SECCION 2: REGRESION LOGISTICA
# ==============================================================================

print("\n" + "=" * 60)
print("2. REGRESION LOGISTICA")
print("=" * 60)

X_log = df[feature_cols].values
y_log = df[COLUMNA_OBJETIVO_LOGISTICA].values

# Dividir en entrenamiento y prueba.
# stratify=y_log garantiza que la proporcion de clases sea igual en ambos conjuntos,
# lo cual es importante para no sesgar la evaluacion del modelo.
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
)

print(f"Columna objetivo   : {COLUMNA_OBJETIVO_LOGISTICA} (RATING > mediana={mediana_rating})")
print(f"Entrenamiento: {X_train_log.shape[0]} muestras | Prueba: {X_test_log.shape[0]} muestras")

# Definir el pipeline para regresion logistica
pipeline_logistica = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, solver="saga")),
])

# Distribucion de hiperparametros para la busqueda aleatoria.
# Se busca el mejor valor de C y el tipo de penalizacion (l1 o l2).
param_dist_log = {
    "model__C"      : loguniform(1e-3, 1e3),
    "model__penalty": ["l1", "l2"],
}

# Busqueda aleatoria con scoring f1 para balancear precision y recall,
# util cuando ambos tipos de error (falsos positivos y negativos) importan.
print("\nBuscando mejores hiperparametros para regresion logistica...")
search_logistica = RandomizedSearchCV(
    pipeline_logistica, param_dist_log,
    n_iter=50, cv=5, scoring="f1", random_state=42, n_jobs=-1
)
search_logistica.fit(X_train_log, y_train_log)

print(f"Mejores parametros: {search_logistica.best_params_}")

# Predicciones de clase y de probabilidad sobre el conjunto de prueba
y_pred_log = search_logistica.best_estimator_.predict(X_test_log)
y_prob_log = search_logistica.best_estimator_.predict_proba(X_test_log)[:, 1]

# Calcular metricas de clasificacion:
#   Accuracy : proporcion de predicciones correctas sobre el total.
#   F1-score : media armonica de precision y recall.
acc = accuracy_score(y_test_log, y_pred_log)
f1  = f1_score(y_test_log, y_pred_log)

print(f"\nResultados en conjunto de prueba:")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1-score : {f1:.4f}")

# Graficar la distribucion de probabilidades predichas separadas por clase real.
# Permite ver si el modelo separa bien las dos clases o si hay mucha superposicion.
plt.figure(figsize=(8, 5))
for clase, color, etiqueta in zip([0, 1], ["salmon", "steelblue"], ["No exitosa (0)", "Exitosa (1)"]):
    mask = y_test_log == clase
    plt.hist(y_prob_log[mask], bins=30, alpha=0.6, color=color, edgecolor="white", label=etiqueta)
plt.axvline(x=0.5, color="black", linestyle="--", linewidth=1.2, label="Umbral de decision (0.5)")
plt.xlabel("Probabilidad predicha de clase 1 (exitosa)")
plt.ylabel("Frecuencia")
plt.title(f"Distribucion de probabilidades predichas\nAccuracy={acc:.3f}  F1={f1:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CARPETA_SALIDA}/predicciones_logistica.png", bbox_inches="tight")
plt.close()
print(f"-> Grafico guardado: {CARPETA_SALIDA}/predicciones_logistica.png")

# Graficar la matriz de confusion.
# Muestra cuantas muestras de cada clase fueron clasificadas correcta o incorrectamente:
#   Verdaderos Negativos (TN) : real=0, predicho=0 (esquina superior izquierda)
#   Falsos Positivos (FP)     : real=0, predicho=1 (esquina superior derecha)
#   Falsos Negativos (FN)     : real=1, predicho=0 (esquina inferior izquierda)
#   Verdaderos Positivos (TP) : real=1, predicho=1 (esquina inferior derecha)
cm = confusion_matrix(y_test_log, y_pred_log)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No exitosa (0)", "Exitosa (1)"]
).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Matriz de Confusion - Regresion Logistica")
plt.tight_layout()
plt.savefig(f"{CARPETA_SALIDA}/matriz_confusion.png", bbox_inches="tight")
plt.close()
print(f"-> Grafico guardado: {CARPETA_SALIDA}/matriz_confusion.png")


# ==============================================================================
# SECCION 3: RESUMEN FINAL
# ==============================================================================

print("\n" + "=" * 60)
print("3. RESUMEN DE RESULTADOS")
print("=" * 60)

print(f"""
REGRESION LINEAL
  Columna objetivo : {COLUMNA_OBJETIVO_LINEAL}
  Features usadas  : {len(feature_cols)} columnas

  Ridge
    Mejor alpha    : {search_ridge.best_params_["model__alpha"]:.5f}
    R2 en prueba   : {r2_ridge:.4f}
    MAE en prueba  : {mae_ridge:.4f}

  Lasso
    Mejor alpha    : {search_lasso.best_params_["model__alpha"]:.5f}
    R2 en prueba   : {r2_lasso:.4f}
    MAE en prueba  : {mae_lasso:.4f}

REGRESION LOGISTICA
  Columna objetivo : {COLUMNA_OBJETIVO_LOGISTICA} (RATING > mediana)
  Mejores params   : {search_logistica.best_params_}
  Accuracy         : {acc:.4f}
  F1-score         : {f1:.4f}

Graficos guardados en: {CARPETA_SALIDA}
""")

print("=" * 60)
print("Script completado.")
print("=" * 60)