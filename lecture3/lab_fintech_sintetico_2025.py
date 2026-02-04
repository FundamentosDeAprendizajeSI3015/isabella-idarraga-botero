# =============================================================
# LAB FINTECH (SINTÉTICO 2025) — PREPROCESAMIENTO Y EDA
# Datos de entrada fijos para evitar errores de ruta/nombre.
# -------------------------------------------------------------
# Este script está listo para ejecutarse sin argumentos:
#   python lab_fintech_sintetico_2025.py
# 
# Archivos esperados en el mismo directorio:
#   - fintech_top_sintetico_2025.csv
#   - fintech_top_sintetico_dictionary.json
# Salidas (por defecto):
#   ./data_output_finanzas_sintetico/
#       ├─ fintech_train.parquet
#       ├─ fintech_test.parquet
#       ├─ processed_schema.json
#       └─ features_columns.txt
# =============================================================

# Importar módulo json para manipular archivos JSON (diccionario de datos)
import json
# Importar Path de pathlib para manejo multiplataforma de rutas de archivos
from pathlib import Path
# Importar módulo warnings para controlar mensajes de advertencia
import warnings
# Importar módulo sys para interacción con el sistema operativo
import sys
# Suprimir todas las advertencias para mantener la consola limpia durante la ejecución
warnings.filterwarnings("ignore")

# Importar numpy para operaciones numéricas y cálculos matemáticos
import numpy as np
# Importar pandas para manipulación y análisis de datos (DataFrames, Series)
import pandas as pd
# Importar train_test_split para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
# Importar StandardScaler para normalizar características numéricas a media 0 y desv. std 1
from sklearn.preprocessing import StandardScaler
# Importar matplotlib para crear visualizaciones estáticas de alta calidad
import matplotlib.pyplot as plt
# Importar seaborn para gráficos estadísticos mejorados y temas visuales atractivos
import seaborn as sns

# Configurar seaborn con un estilo profesional y bonito
sns.set_style("whitegrid")
# Establecer paleta de colores moderna y armoniosa
sns.set_palette("husl")

# Corregir problemas de codificación en Windows para mostrar caracteres especiales correctamente
if sys.platform == 'win32':
    # Reconfigurar la salida estándar para usar codificación UTF-8
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------
# Constantes de la práctica
# ---------------------------
# Nombre del archivo CSV que contiene los datos del fintech sintético
DATA_CSV = 'fintech_top_sintetico_2025.csv'
# Nombre del archivo JSON que contiene el diccionario/descripción de las columnas
DATA_DICT = 'fintech_top_sintetico_dictionary.json'
# Ruta del directorio donde se guardarán los archivos procesados
OUTDIR = Path('./data_output_finanzas_sintetico')
# Fecha de corte para dividir el dataset en conjuntos de entrenamiento y prueba
SPLIT_DATE = '2025-09-01'  # partición temporal por defecto

# Columnas esperadas por diseño del dataset sintético
# Nombre de la columna que contiene las fechas de los registros
DATE_COL = 'Month'
# Lista de columnas identificadoras (clave única: nombre de la empresa)
ID_COLS = ['Company']
# Lista de columnas categóricas: país, región, tipo de segmento, etc.
CAT_COLS = ['Country', 'Region', 'Segment', 'Subsegment', 'IsPublic', 'Ticker']
# Lista de columnas numéricas: usuarios, TPV, revenue, métricas de negocio, etc.
NUM_COLS = [
    'Users_M','NewUsers_K','TPV_USD_B','TakeRate_pct','Revenue_USD_M',
    'ARPU_USD','Churn_pct','Marketing_Spend_USD_M','CAC_USD','CAC_Total_USD_M',
    'Close_USD','Private_Valuation_USD_B'
]
# Lista específica de columnas de precio para calcular retornos y log-retornos
PRICE_COLS = ['Close_USD']  # para calcular retornos opcionales

# ---------------------------
# 0) Carga de diccionario
# ---------------------------
# Imprimir encabezado del primer paso de procesamiento
print("\n=== 0) Cargando diccionario de datos ===")
# Crear objeto Path para la ruta del diccionario JSON
dict_path = Path(DATA_DICT)
# Verificar si el archivo del diccionario existe en el directorio actual
if not dict_path.exists():
    # Si no existe, lanzar excepción informativa al usuario
    raise FileNotFoundError(f"No se encontró {DATA_DICT}. Asegúrate de tener el archivo en la misma carpeta.")

# Abrir y leer el archivo JSON del diccionario de datos en modo lectura con codificación UTF-8
with open(dict_path, 'r', encoding='utf-8') as f:
    # Cargar el contenido JSON en una variable diccionario de Python
    data_dict = json.load(f)
# Extraer e imprimir la descripción del dataset del diccionario (o valor por defecto si no existe)
print("Descripción:", data_dict.get('description', '(sin descripción)'))
# Extraer e imprimir el período temporal cubierto por el dataset (o valor por defecto si no existe)
print("Periodo:", data_dict.get('period', '(desconocido)'))

# ---------------------------
# 1) Carga del CSV
# ---------------------------
# Imprimir encabezado del paso de carga de datos CSV
print("\n=== 1) Cargando CSV sintético ===")
# Crear objeto Path para la ruta del archivo CSV
csv_path = Path(DATA_CSV)
# Verificar si el archivo CSV existe en el directorio actual
if not csv_path.exists():
    # Si no existe, lanzar excepción informativa al usuario
    raise FileNotFoundError(f"No se encontró {DATA_CSV}. Asegúrate de tener el archivo en la misma carpeta.")

# Leer el archivo CSV usando pandas y almacenar en un DataFrame
df = pd.read_csv(csv_path)
# Imprimir las dimensiones del DataFrame: número de filas y columnas
print("Shape:", df.shape)

# Parseo de fecha y orden temporal
# Verificar si la columna de fecha existe en el DataFrame
if DATE_COL not in df.columns:
    # Si la columna no existe, lanzar excepción indicando el error
    raise KeyError(f"La columna de fecha '{DATE_COL}' no existe en el CSV.")

# Convertir la columna de fecha a formato datetime, coerciendo errores a NaT (Not a Time)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
# Ordenar el DataFrame primero por fecha y luego por empresa (para mantener orden temporal)
df = df.sort_values([DATE_COL] + ID_COLS).reset_index(drop=True)

# Imprimir las primeras 3 filas del DataFrame para verificar carga correcta
print("Primeras filas:")
print(df.head(3))

# ---------------------------
# 2) EDA breve
# ---------------------------
# Imprimir encabezado del análisis exploratorio de datos (EDA)
print("\n=== 2) EDA rápido ===")
# Mostrar información general del DataFrame: tipos de datos, memoria, valores no nulos
print("Info:")
print(df.info())
# Crear una visualización de valores faltantes
print("\nNulos por columna (top 15):")
# Contar valores nulos en cada columna, ordenarlos en forma descendente y mostrar top 15
print(df.isna().sum().sort_values(ascending=False).head(15))

# ---------------------------
# 3) Limpieza básica
# ---------------------------
# Imprimir encabezado del paso de limpieza de datos
print("\n=== 3) Limpieza ===")
# Imputación simple: numéricos con mediana, categóricos con marcador
# Iterar sobre cada columna numérica definida en NUM_COLS
for c in NUM_COLS:
    # Verificar si la columna existe en el DataFrame y tiene valores faltantes (NaN)
    if c in df.columns and df[c].isna().any():
        # Convertir la columna a tipo numérico, coerciendo errores a NaN
        df[c] = pd.to_numeric(df[c], errors='coerce')
        # Rellenar valores faltantes con la mediana de la columna (resistente a outliers)
        df[c] = df[c].fillna(df[c].median())

# Iterar sobre cada columna categórica definida en CAT_COLS
for c in CAT_COLS:
    # Verificar si la columna existe en el DataFrame y tiene valores faltantes (NaN)
    if c in df.columns and df[c].isna().any():
        # Rellenar valores faltantes con marcador especial '__MISSING__' para identificarlos
        df[c] = df[c].fillna('__MISSING__')

# ---------------------------
# 4) Ingeniería ligera: retornos/log-retornos de precio
# ---------------------------
# Imprimir encabezado del paso de ingeniería de características
print("\n=== 4) Ingeniería de rasgos (retornos) ===")
# Verificar si todas las columnas de precio existen en el DataFrame
if all([pc in df.columns for pc in PRICE_COLS]):
    # Iterar sobre cada columna de precio definida en PRICE_COLS
    for pc in PRICE_COLS:
        # Calcular retornos porcentuales agrupados por empresa (cambio porcentual mes a mes)
        df[pc + '_ret'] = (
            # Ordenar por empresa y fecha para asegurar continuidad temporal
            df.sort_values([ID_COLS[0], DATE_COL])
              # Agrupar por empresa (cada empresa tiene su propia serie de precios)
              .groupby(ID_COLS)[pc]
              # Calcular el cambio porcentual respecto al período anterior
              .pct_change()
        )
        # Calcular log-retornos (retornos logarítmicos) usando log(1 + retorno simple)
        df[pc + '_logret'] = np.log1p(df[pc + '_ret'])
        # Rellenar primeros valores NaN (primer período de cada empresa) con 0.0
        df[pc + '_ret'] = df[pc + '_ret'].fillna(0.0)
        # Rellenar primeros valores NaN de log-retornos con 0.0
        df[pc + '_logret'] = df[pc + '_logret'].fillna(0.0)
else:
    # Si no están disponibles las columnas de precio, informar al usuario
    print("[INFO] Columnas de precio no disponibles; se omite cálculo de retornos.")

# Actualizamos lista de numéricos tras ingeniería
# Crear lista de columnas numéricas adicionales creadas por ingeniería de características
extra_num = [c for c in [pc + '_ret' for pc in PRICE_COLS] + [pc + '_logret' for pc in PRICE_COLS] if c in df.columns]
# Crear lista completa de columnas numéricas usadas: originales + ingenieridas
NUM_USED = [c for c in NUM_COLS if c in df.columns] + extra_num

# ---------------------------
# 5) Separación X / y (sin y por defecto) + codificación
# ---------------------------
# Imprimir encabezado del paso de preparación de características
print("\n=== 5) Preparación de X: codificación one-hot y escalado ===")
# Crear matriz de características X eliminando identificadores y columna de fecha
# (fecha se usa para split pero no como predictor)
X = df.drop(columns=[DATE_COL] + ID_COLS, errors='ignore').copy()

# One-hot en categóricas
# Filtrar solo las columnas categóricas que existen en X (matriz de características)
cat_in_X = [c for c in CAT_COLS if c in X.columns]
# Aplicar codificación one-hot: convertir variables categóricas en variables binarias
# drop_first=True evita multicolinealidad perfecta eliminando una categoría base
X = pd.get_dummies(X, columns=cat_in_X, drop_first=True)

# Partición temporal por defecto utilizando la fecha de corte
# Convertir la fecha de corte a formato datetime
cutoff = pd.to_datetime(SPLIT_DATE)
# Crear máscara booleana para datos de entrenamiento (antes de la fecha de corte)
idx_train = df[DATE_COL] < cutoff
# Crear máscara booleana para datos de prueba (desde la fecha de corte en adelante)
idx_test = df[DATE_COL] >= cutoff

# Dividir X en conjuntos de entrenamiento y prueba usando las máscaras
X_train, X_test = X.loc[idx_train].copy(), X.loc[idx_test].copy()

# Escalado de numéricos (solo columnas presentes en X)
# Filtrar columnas numéricas que existen en X_train (después de one-hot encoding)
num_in_X = [c for c in NUM_USED if c in X_train.columns]
# Instanciar escalador estándar que normaliza características a media 0 y desv. std 1
scaler = StandardScaler()
# Aplicar escalado si hay columnas numéricas
if num_in_X:
    # Ajustar el escalador con datos de entrenamiento (fit) y transformar
    # IMPORTANTE: no usar transform() en test para evitar data leakage
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
    # Aplicar el escalador ya ajustado a datos de prueba (sin reajustar)
    X_test[num_in_X] = scaler.transform(X_test[num_in_X])
else:
    # Informar si no hay columnas numéricas para escalar
    print("[INFO] No se encontraron columnas numéricas para escalar.")

# Imprimir dimensiones de los conjuntos de entrenamiento y prueba
print("Shapes -> X_train:", X_train.shape, " X_test:", X_test.shape)

# ---------------------------
# 6) Exportación
# ---------------------------
# Imprimir encabezado del paso de exportación de datos procesados
print("\n=== 6) Exportación ===")
# Crear directorio de salida si no existe (parents=True crea directorios padre si es necesario)
OUTDIR.mkdir(parents=True, exist_ok=True)
# Definir ruta del archivo parquet para datos de entrenamiento
train_path = OUTDIR / 'fintech_train.parquet'
# Definir ruta del archivo parquet para datos de prueba
test_path = OUTDIR / 'fintech_test.parquet'

# Guardamos sólo X (sin objetivo/target)
# Guardar X_train en formato parquet (comprimido y eficiente para datos tabulares)
X_train.to_parquet(train_path, index=False)
# Guardar X_test en formato parquet
X_test.to_parquet(test_path, index=False)

# Guardar esquema procesado
# Crear diccionario con metadatos del procesamiento para referencia futura
processed_schema = {
    # Ruta absoluta del archivo CSV original
    'source_csv': str(csv_path.resolve()),
    # Ruta absoluta del archivo diccionario original
    'source_dict': str(dict_path.resolve()),
    # Nombre de la columna de fecha
    'date_col': DATE_COL,
    # Columnas identificadoras utilizadas
    'id_cols': ID_COLS,
    # Columnas categóricas usadas en el análisis
    'categorical_cols_used': cat_in_X,
    # Columnas numéricas usadas en el análisis
    'numeric_cols_used': num_in_X,
    # Columnas creadas por ingeniería de características
    'engineered_cols': extra_num,
    # Información sobre la división temporal
    'split': {
        'type': 'time_split',  # Tipo de split: basado en tiempo
        'cutoff': SPLIT_DATE,  # Fecha de corte
        'train_rows': int(idx_train.sum()),  # Número de filas en entrenamiento
        'test_rows': int(idx_test.sum()),  # Número de filas en prueba
    },
    # Dimensiones del conjunto de entrenamiento
    'X_train_shape': list(X_train.shape),
    # Dimensiones del conjunto de prueba
    'X_test_shape': list(X_test.shape),
    # Notas importantes sobre el dataset y procesamiento
    'notes': [
        'Dataset 100% SINTÉTICO con fines académicos; no refleja métricas reales.',
        'Evitar fuga de datos: el escalador se ajusta en TRAIN y se aplica a TEST.'
    ]
}

# Guardar el esquema procesado en formato JSON para referencia futura
with open(OUTDIR / 'processed_schema.json', 'w', encoding='utf-8') as f:
    # Usar indent=2 para formato legible, ensure_ascii=False para caracteres especiales
    json.dump(processed_schema, f, ensure_ascii=False, indent=2)

# Lista de columnas finales para referencia de modelado
# Abrir archivo de texto para guardar los nombres de todas las características
with open(OUTDIR / 'features_columns.txt', 'w', encoding='utf-8') as f:
    # Escribir un nombre de columna por línea (sin índices)
    f.write("\n".join(X_train.columns))

print("\nArchivos exportados:")
print(" -", train_path)
print(" -", test_path)
print(" -", OUTDIR / 'processed_schema.json')
print(" -", OUTDIR / 'features_columns.txt')

print("\nListo. Recuerda: este dataset es sintetico para practica academica.")

# =============================================================
# 7) VISUALIZACIONES INTERESANTES - EDA GRÁFICO
# =============================================================
print("\n=== 7) Generando visualizaciones ===")

# Crear figura con múltiples subgráficos para visualización integral
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Análisis Exploratorio de Datos (EDA) - Fintech Sintético 2025', 
             fontsize=18, fontweight='bold', y=0.995)

# -------- SUBPLOT 1: Distribución de valores nulos --------
# Ubicar el primer gráfico en posición 3x3, índice 1
ax1 = plt.subplot(3, 3, 1)
# Seleccionar columnas numéricas del DataFrame original
numeric_cols_for_viz = [c for c in NUM_COLS if c in df.columns]
# Contar valores nulos en columnas numéricas
null_counts = df[numeric_cols_for_viz].isna().sum().sort_values(ascending=True)
# Crear gráfico de barras horizontal mostrando columnas con valores nulos
if null_counts.sum() > 0:
    null_counts.plot(kind='barh', ax=ax1, color='coral')
else:
    null_counts.plot(kind='barh', ax=ax1, color='green')
# Personalizar el gráfico
ax1.set_xlabel('Cantidad de Nulos')
ax1.set_title('Valores Faltantes por Columna', fontweight='bold', fontsize=11)
ax1.grid(axis='x', alpha=0.3)

# -------- SUBPLOT 2: Distribución temporal del dataset --------
# Ubicar el segundo gráfico en posición 3x3, índice 2
ax2 = plt.subplot(3, 3, 2)
# Contar el número de registros por mes
records_per_month = df.groupby(DATE_COL).size()
# Crear gráfico de línea mostrando la evolución temporal
records_per_month.plot(ax=ax2, color='steelblue', linewidth=2, marker='o', markersize=4)
# Marcar la fecha de corte con una línea vertical roja
ax2.axvline(x=cutoff, color='red', linestyle='--', linewidth=2, label='Fecha de Split')
# Personalizar el gráfico
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Cantidad de Registros')
ax2.set_title('Distribución Temporal de Datos', fontweight='bold', fontsize=11)
ax2.legend()
ax2.grid(alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# -------- SUBPLOT 3: Distribución de empresas --------
# Ubicar el tercer gráfico en posición 3x3, índice 3
ax3 = plt.subplot(3, 3, 3)
# Contar cuántos registros tiene cada empresa
company_counts = df[ID_COLS[0]].value_counts().head(10)
# Crear gráfico de barras mostrando top 10 empresas
company_counts.plot(kind='barh', ax=ax3, color='mediumpurple')
# Personalizar el gráfico
ax3.set_xlabel('Cantidad de Registros')
ax3.set_title('Top 10 Empresas por Registros', fontweight='bold', fontsize=11)
ax3.grid(axis='x', alpha=0.3)

# -------- SUBPLOT 4: Distribución por País --------
# Ubicar el cuarto gráfico en posición 3x3, índice 4
ax4 = plt.subplot(3, 3, 4)
# Contar el número de registros por país
country_counts = df['Country'].value_counts().head(8)
# Crear gráfico de barras circular (pie) mostrando distribución por país
colors_pie = sns.color_palette("husl", len(country_counts))
ax4.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', 
        colors=colors_pie, startangle=90)
# Personalizar el gráfico
ax4.set_title('Distribución por País', fontweight='bold', fontsize=11)

# -------- SUBPLOT 5: Distribución por Región --------
# Ubicar el quinto gráfico en posición 3x3, índice 5
ax5 = plt.subplot(3, 3, 5)
# Contar el número de registros por región
region_counts = df['Region'].value_counts()
# Crear gráfico de barras mostrando distribución por región
region_counts.plot(kind='bar', ax=ax5, color='lightseagreen')
# Personalizar el gráfico
ax5.set_xlabel('Región')
ax5.set_ylabel('Cantidad')
ax5.set_title('Distribución por Región', fontweight='bold', fontsize=11)
ax5.grid(axis='y', alpha=0.3)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

# -------- SUBPLOT 6: Distribución de tipos de empresa (Público/Privado) --------
# Ubicar el sexto gráfico en posición 3x3, índice 6
ax6 = plt.subplot(3, 3, 6)
# Contar empresas públicas vs privadas
is_public_counts = df['IsPublic'].value_counts()
# Crear gráfico de barras comparando entidades públicas vs privadas
colors_public = ['#FF6B6B', '#4ECDC4']
is_public_counts.plot(kind='bar', ax=ax6, color=colors_public)
# Personalizar el gráfico
ax6.set_xlabel('Tipo de Empresa')
ax6.set_ylabel('Cantidad')
ax6.set_title('Empresas Públicas vs Privadas', fontweight='bold', fontsize=11)
ax6.grid(axis='y', alpha=0.3)
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=0)

# -------- SUBPLOT 7: Distribución de Revenue (ingresos) --------
# Ubicar el séptimo gráfico en posición 3x3, índice 7
ax7 = plt.subplot(3, 3, 7)
# Seleccionar datos de revenue válidos (sin NaN)
revenue_data = df['Revenue_USD_M'].dropna()
# Crear histograma mostrando la distribución de ingresos
ax7.hist(revenue_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
# Añadir línea vertical para la media
ax7.axvline(revenue_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: ${revenue_data.mean():.1f}M')
# Añadir línea vertical para la mediana
ax7.axvline(revenue_data.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: ${revenue_data.median():.1f}M')
# Personalizar el gráfico
ax7.set_xlabel('Ingresos (USD Millones)')
ax7.set_ylabel('Frecuencia')
ax7.set_title('Distribución de Ingresos (Revenue)', fontweight='bold', fontsize=11)
ax7.legend()
ax7.grid(axis='y', alpha=0.3)

# -------- SUBPLOT 8: Distribución de Users (usuarios) --------
# Ubicar el octavo gráfico en posición 3x3, índice 8
ax8 = plt.subplot(3, 3, 8)
# Seleccionar datos de usuarios válidos (sin NaN)
users_data = df['Users_M'].dropna()
# Crear histograma mostrando la distribución de usuarios
ax8.hist(users_data, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
# Añadir línea vertical para la media
ax8.axvline(users_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {users_data.mean():.1f}M')
# Personalizar el gráfico
ax8.set_xlabel('Usuarios (Millones)')
ax8.set_ylabel('Frecuencia')
ax8.set_title('Distribución de Usuarios', fontweight='bold', fontsize=11)
ax8.legend()
ax8.grid(axis='y', alpha=0.3)

# -------- SUBPLOT 9: Matriz de correlación de variables numéricas --------
# Ubicar el noveno gráfico en posición 3x3, índice 9
ax9 = plt.subplot(3, 3, 9)
# Seleccionar columnas numéricas para calcular correlaciones
numeric_subset = df[numeric_cols_for_viz].select_dtypes(include=[np.number])
# Calcular matriz de correlación de Pearson
correlation_matrix = numeric_subset.corr()
# Crear mapa de calor (heatmap) mostrando correlaciones entre variables
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, ax=ax9, 
            cbar_kws={'label': 'Correlación'}, square=True, 
            annot=False, fmt='.2f', linewidths=0.5)
# Personalizar el gráfico
ax9.set_title('Matriz de Correlación - Variables Numéricas', fontweight='bold', fontsize=11)
plt.setp(ax9.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax9.yaxis.get_majorticklabels(), rotation=0, fontsize=8)

# Ajustar espaciado entre subgráficos
plt.tight_layout()

# Guardar figura en alta resolución
viz_output_path = OUTDIR / 'eda_visualization.png'
plt.savefig(viz_output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualización principal guardada: {viz_output_path}")

# Mostrar la figura en pantalla
plt.show()

# ====== GRÁFICO ADICIONAL: Análisis de Retornos de Precio (si existen) ======
if 'Close_USD_ret' in df.columns and 'Close_USD_logret' in df.columns:
    # Crear nueva figura para análisis de retornos
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Análisis de Retornos de Precio (Close_USD)', fontsize=16, fontweight='bold')
    
    # -------- Subplot 1: Distribución de retornos simples --------
    # Graficar histograma de retornos simples (porcentuales)
    axes[0, 0].hist(df['Close_USD_ret'].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['Close_USD_ret'].mean(), color='red', linestyle='--', linewidth=2, label='Media')
    axes[0, 0].set_xlabel('Retorno Simple (%)')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución de Retornos Simples', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # -------- Subplot 2: Distribución de log-retornos --------
    # Graficar histograma de log-retornos
    axes[0, 1].hist(df['Close_USD_logret'].dropna(), bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['Close_USD_logret'].mean(), color='darkred', linestyle='--', linewidth=2, label='Media')
    axes[0, 1].set_xlabel('Log-Retorno')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Log-Retornos', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # -------- Subplot 3: Series temporal de retornos --------
    # Graficar serie temporal de retornos simples
    axes[1, 0].plot(df.groupby(DATE_COL)['Close_USD_ret'].mean(), color='steelblue', linewidth=1.5, marker='o', markersize=3)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Fecha')
    axes[1, 0].set_ylabel('Retorno Promedio (%)')
    axes[1, 0].set_title('Evolución Temporal de Retornos Simples', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # -------- Subplot 4: Q-Q Plot para verificar normalidad --------
    # Crear Q-Q plot para comparar distribución de retornos con distribución normal
    from scipy import stats
    stats.probplot(df['Close_USD_ret'].dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Retornos vs Distribución Normal', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # Ajustar espaciado
    plt.tight_layout()
    
    # Guardar figura de retornos
    returns_viz_path = OUTDIR / 'returns_analysis.png'
    plt.savefig(returns_viz_path, dpi=300, bbox_inches='tight')
    print(f"✓ Análisis de retornos guardado: {returns_viz_path}")
    
    plt.show()

# ====== GRÁFICO ADICIONAL: Relación entre características numéricas clave ======
# Crear figura para análisis de relaciones entre pares de variables importantes
fig3, axes = plt.subplots(2, 3, figsize=(16, 10))
fig3.suptitle('Relaciones entre Características Clave del Negocio', fontsize=16, fontweight='bold')

# -------- Scatter 1: Users vs Revenue --------
# Graficar relación entre número de usuarios y ingresos
ax = axes[0, 0]
ax.scatter(df['Users_M'].dropna(), df['Revenue_USD_M'].dropna(), alpha=0.5, c='steelblue', s=30)
ax.set_xlabel('Usuarios (Millones)')
ax.set_ylabel('Ingresos (USD Millones)')
ax.set_title('Usuarios vs Ingresos', fontweight='bold')
ax.grid(alpha=0.3)

# -------- Scatter 2: Revenue vs Valuation --------
# Graficar relación entre ingresos y valuación privada
ax = axes[0, 1]
ax.scatter(df['Revenue_USD_M'].dropna(), df['Private_Valuation_USD_B'].dropna(), alpha=0.5, c='coral', s=30)
ax.set_xlabel('Ingresos (USD Millones)')
ax.set_ylabel('Valuación Privada (USD Billones)')
ax.set_title('Ingresos vs Valuación', fontweight='bold')
ax.grid(alpha=0.3)

# -------- Scatter 3: Marketing Spend vs NewUsers --------
# Graficar relación entre gasto en marketing y nuevos usuarios
ax = axes[0, 2]
ax.scatter(df['Marketing_Spend_USD_M'].dropna(), df['NewUsers_K'].dropna(), alpha=0.5, c='mediumseagreen', s=30)
ax.set_xlabel('Gasto Marketing (USD Millones)')
ax.set_ylabel('Nuevos Usuarios (Miles)')
ax.set_title('Marketing Spend vs Nuevos Usuarios', fontweight='bold')
ax.grid(alpha=0.3)

# -------- Scatter 4: Churn vs Revenue --------
# Graficar relación entre tasa de abandono (churn) e ingresos
ax = axes[1, 0]
ax.scatter(df['Churn_pct'].dropna(), df['Revenue_USD_M'].dropna(), alpha=0.5, c='mediumpurple', s=30)
ax.set_xlabel('Churn (%)')
ax.set_ylabel('Ingresos (USD Millones)')
ax.set_title('Churn vs Ingresos', fontweight='bold')
ax.grid(alpha=0.3)

# -------- Scatter 5: ARPU vs TPV --------
# Graficar relación entre ingresos por usuario y volumen de transacciones
ax = axes[1, 1]
ax.scatter(df['ARPU_USD'].dropna(), df['TPV_USD_B'].dropna(), alpha=0.5, c='orange', s=30)
ax.set_xlabel('ARPU (USD)')
ax.set_ylabel('TPV (USD Billones)')
ax.set_title('ARPU vs TPV', fontweight='bold')
ax.grid(alpha=0.3)

# -------- Scatter 6: CAC vs Revenue --------
# Graficar relación entre costo de adquisición de cliente y ingresos
ax = axes[1, 2]
ax.scatter(df['CAC_USD'].dropna(), df['Revenue_USD_M'].dropna(), alpha=0.5, c='crimson', s=30)
ax.set_xlabel('CAC (USD)')
ax.set_ylabel('Ingresos (USD Millones)')
ax.set_title('CAC vs Ingresos', fontweight='bold')
ax.grid(alpha=0.3)

# Ajustar espaciado
plt.tight_layout()

# Guardar figura de relaciones
relations_viz_path = OUTDIR / 'features_relationships.png'
plt.savefig(relations_viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Análisis de relaciones guardado: {relations_viz_path}")

plt.show()

# Mensaje final resumen
print("\n" + "="*70)
print("✓ PROCESAMIENTO COMPLETADO CON ÉXITO")
print("="*70)
print(f"✓ Datos de entrenamiento: {X_train.shape[0]} registros, {X_train.shape[1]} características")
print(f"✓ Datos de prueba: {X_test.shape[0]} registros, {X_test.shape[1]} características")
print(f"✓ Todas las visualizaciones se encuentran en: {OUTDIR}")
print("="*70)