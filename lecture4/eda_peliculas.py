# EDA - Dataset de Peliculas
# SI3015 - Fundamentos de Aprendizaje Automatico

# DESCRIPCION GENERAL:
#   Este script realiza un Analisis Exploratorio de Datos (EDA) completo sobre
#   el dataset de peliculas. Incluye medidas estadisticas, visualizaciones,
#   deteccion y eliminacion de outliers, y transformaciones de columnas listas
#   para ser usadas en modelos de aprendizaje automatico.
#
# LIBRERIAS REQUERIDAS:
#   pip install pandas numpy matplotlib seaborn scikit-learn

import os           # Para manejo de directorios del sistema operativo
import warnings     # Para suprimir advertencias no criticas durante la ejecucion
import numpy as np  # Operaciones numericas y matematicas
import pandas as pd # Carga, manipulacion y analisis de datos tabulares
import matplotlib.pyplot as plt  # Generacion de graficos estaticos
import seaborn as sns            # Visualizaciones estadisticas de alto nivel

# Escaladores para normalizar y estandarizar variables numericas
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Suprimir advertencias de pandas y sklearn que no afectan la logica del script
warnings.filterwarnings("ignore")

# Estilo visual global para todos los graficos generados con seaborn
sns.set_theme(style="whitegrid")

# Ruta al archivo CSV del dataset de peliculas.
RUTA_DATASET = "movies.csv"

# Carpeta donde se guardaran todos los graficos generados.
# Se crea automaticamente si no existe.
CARPETA_SALIDA = "graficos"
os.makedirs(CARPETA_SALIDA, exist_ok=True)


# ==============================================================================
# SECCION 0: CARGA DE DATOS

# Se carga el archivo CSV en un DataFrame de pandas.
# Se muestra una vista previa, los tipos de datos y los valores nulos
# para tener un panorama inicial del conjunto de datos.
# ==============================================================================

print("=" * 60)
print("0. CARGA DE DATOS")
print("=" * 60)

# Leer el archivo CSV y cargarlo en un DataFrame
df = pd.read_csv(RUTA_DATASET)

# Mostrar dimensiones del dataset
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas\n")

# Mostrar las primeras 5 filas para inspeccionar visualmente el contenido
print("Primeras 5 filas del dataset:")
print(df.head())

# Mostrar el tipo de dato de cada columna (int64, float64, object, etc.)
print("\nTipos de datos por columna:")
print(df.dtypes)

# Contar cuantos valores nulos hay por columna.
# Esto indica donde hay datos faltantes que deben tratarse antes del modelado.
print("\nValores nulos por columna:")
print(df.isnull().sum())


# ==============================================================================
# SECCION 1: LIMPIEZA BASICA

# Antes de analizar, se realiza una limpieza inicial:
#   - Eliminar filas completamente duplicadas.
#   - Identificar columnas numericas y categoricas automaticamente.
#   - Imputar (rellenar) valores nulos con estadisticos representativos:
#       * Numericas   -> mediana (robusta ante outliers)
#       * Categoricas -> moda    (valor mas frecuente)
# ==============================================================================

print("\n" + "=" * 60)
print("1. LIMPIEZA BASICA")
print("=" * 60)

# Guardar numero de filas antes de eliminar duplicados para poder comparar
filas_antes_dup = len(df)

# Eliminar filas duplicadas. inplace=True modifica el DataFrame directamente.
df.drop_duplicates(inplace=True)
print(f"Duplicados eliminados: {filas_antes_dup - len(df)}")

# Detectar automaticamente las columnas segun su tipo de dato:
#   - num_cols: columnas con valores enteros o decimales (int64, float64)
#   - cat_cols: columnas con texto o categorias (object, category)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\nColumnas numericas   : {num_cols}")
print(f"Columnas categoricas : {cat_cols}")

# Imputar valores nulos en columnas numericas usando la mediana.
# La mediana es preferida sobre la media porque no se ve afectada por outliers.
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Imputar valores nulos en columnas categoricas usando la moda.
# mode() devuelve una Serie; .iloc[0] extrae el valor mas frecuente.
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Verificar que ya no queden valores nulos en el DataFrame
print("\nTotal de valores nulos tras la limpieza:", df.isnull().sum().sum())


# ==============================================================================
# SECCION 2: MEDIDAS DE TENDENCIA CENTRAL
#
#   Media   : promedio aritmetico de todos los valores. Es sensible a outliers;
#             un valor muy extremo puede desplazarla significativamente.
#
#   Mediana : valor que divide los datos ordenados por la mitad. Es robusta
#             ante outliers, por eso se usa tambien para imputacion de nulos.
#
#   Moda    : valor que aparece con mayor frecuencia en los datos. Util en
#             variables discretas o categoricas.
# ==============================================================================

print("\n" + "=" * 60)
print("2. MEDIDAS DE TENDENCIA CENTRAL")
print("=" * 60)

# Construir una tabla resumen con las tres medidas para cada columna numerica
tendencia = pd.DataFrame({
    "Media"   : df[num_cols].mean(),           # Promedio de todos los valores
    "Mediana" : df[num_cols].median(),         # Valor central de la distribucion
    "Moda"    : df[num_cols].mode().iloc[0],   # Valor que mas se repite
})

# to_string() muestra la tabla completa sin truncar columnas
print(tendencia.to_string())


# ==============================================================================
# SECCION 3: MEDIDAS DE DISPERSION

# Las medidas de dispersion describen que tan "esparcidos" estan los datos
# alrededor de su centro. Un valor alto indica mayor variabilidad.
#
#   Desviacion Estandar : raiz cuadrada de la varianza. Mide la dispersion
#                         promedio de los valores respecto a la media.
#
#   Varianza            : promedio de las desviaciones al cuadrado respecto a la media.
#
#   Rango               : diferencia entre el valor maximo y el minimo.
#                         Muy sensible a la presencia de outliers.
#
#   IQR                 : diferencia entre el percentil 75 (Q3) y el percentil 25 (Q1).
#                         Mide la dispersion del 50% central de los datos.
#                         Es robusto ante outliers y base del metodo de deteccion IQR.
# ==============================================================================

print("\n" + "=" * 60)
print("3. MEDIDAS DE DISPERSION")
print("=" * 60)

dispersion = pd.DataFrame({
    "Desv. Estandar" : df[num_cols].std(),
    "Varianza"       : df[num_cols].var(),
    "Rango"          : df[num_cols].max() - df[num_cols].min(),
    # IQR = Q3 - Q1; cuantifica la amplitud del 50% central de los datos
    "IQR"            : df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25),
})

print(dispersion.to_string())


# ==============================================================================
# SECCION 4: MEDIDAS DE POSICION Y ELIMINACION DE OUTLIERS

# Los percentiles dividen los datos ordenados en partes porcentuales iguales:
#   - Q1 (percentil 25): el 25% de los datos estan por debajo de este valor.
#   - Q2 (percentil 50): equivale a la mediana; divide los datos en dos mitades.
#   - Q3 (percentil 75): el 75% de los datos estan por debajo de este valor.
#
# Metodo IQR para deteccion de outliers:
#   Limite inferior = Q1 - 1.5 * IQR
#   Limite superior = Q3 + 1.5 * IQR
#   Cualquier valor fuera de estos limites se considera un outlier y se elimina.
#   Este metodo es robusto porque no asume ninguna distribucion de los datos.
# ==============================================================================

print("\n" + "=" * 60)
print("4. MEDIDAS DE POSICION Y OUTLIERS")
print("=" * 60)

# Calcular y mostrar los tres percentiles principales para cada columna numerica
print("\nPercentiles (Q1=25%, Mediana=50%, Q3=75%):")
print(df[num_cols].quantile([0.25, 0.50, 0.75]).to_string())

# --- Boxplots antes de eliminar outliers ---
# El boxplot (diagrama de caja) muestra visualmente la distribucion de una variable:
#   - Caja central  : va de Q1 a Q3 (representa el IQR).
#   - Linea interior: muestra la mediana.
#   - Bigotes       : se extienden hasta 1.5 * IQR desde cada extremo de la caja.
#   - Puntos fuera  : valores candidatos a outliers que superan los bigotes.
fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 5))

# Si solo hay una columna numerica, axes no es lista; se convierte para poder iterar
if len(num_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, num_cols):
    sns.boxplot(y=df[col], ax=ax, color="steelblue")
    ax.set_title(f"Boxplot: {col}")

plt.suptitle("Boxplots - Antes de eliminar outliers", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{CARPETA_SALIDA}/boxplots_antes.png", bbox_inches="tight")
plt.close()
print(f"\n-> Grafico guardado: {CARPETA_SALIDA}/boxplots_antes.png")

# --- Eliminacion de outliers con metodo IQR ---
# Se recorre cada columna numerica y se eliminan las filas cuyos valores
# esten fuera de los limites calculados con el IQR.
df_limpio = df.copy()        # Copia del DataFrame para no modificar el original
filas_antes = len(df_limpio) # Guardar cantidad de filas para comparar al final

for col in num_cols:
    Q1  = df_limpio[col].quantile(0.25)     # Primer cuartil
    Q3  = df_limpio[col].quantile(0.75)     # Tercer cuartil
    IQR = Q3 - Q1                           # Rango intercuartilico

    limite_inf = Q1 - 1.5 * IQR            # Limite inferior aceptable
    limite_sup = Q3 + 1.5 * IQR            # Limite superior aceptable

    # Conservar solo las filas cuyos valores esten dentro de los limites
    df_limpio = df_limpio[
        (df_limpio[col] >= limite_inf) & (df_limpio[col] <= limite_sup)
    ]

print(f"\nFilas antes de eliminar outliers  : {filas_antes}")
print(f"Filas despues de eliminar outliers: {len(df_limpio)}")
print(f"Outliers eliminados               : {filas_antes - len(df_limpio)}")

# Actualizar el DataFrame principal con los datos ya limpios de outliers
df = df_limpio.copy()


# ==============================================================================
# SECCION 5: HISTOGRAMAS

# El histograma divide el rango de valores de una variable en intervalos (bins)
# y cuenta cuantos datos caen en cada uno. Permite identificar:
#   - Si los datos siguen una distribucion normal (forma de campana simetrica).
#   - Si hay sesgo positivo (cola hacia la derecha) o negativo (hacia la izquierda).
#   - Si existen picos multiples que podrian sugerir subpoblaciones en los datos.
# ==============================================================================

print("\n" + "=" * 60)
print("5. HISTOGRAMAS - DISTRIBUCION DE COLUMNAS NUMERICAS")
print("=" * 60)

# Calcular numero de filas y columnas para la cuadricula de subplots.
# Se limita a 3 graficos por fila para mantener la legibilidad.
n         = len(num_cols)
cols_grid = min(n, 3)
rows_grid = (n + cols_grid - 1) // cols_grid  # Division entera redondeada hacia arriba

fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(6 * cols_grid, 5 * rows_grid))

# flatten() convierte la matriz de ejes en un arreglo 1D para facilitar la iteracion
axes = np.array(axes).flatten()

for i, col in enumerate(num_cols):
    # bins=30: divide el rango de valores en 30 intervalos
    # alpha=0.85: leve transparencia entre barras para mejor presentacion visual
    axes[i].hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[i].set_title(f"Distribucion: {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frecuencia")

# Ocultar subplots sobrantes si el numero de columnas no llena la cuadricula
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Histogramas de variables numericas", fontsize=14)
plt.tight_layout()
plt.savefig(f"{CARPETA_SALIDA}/histogramas.png", bbox_inches="tight")
plt.close()
print(f"-> Grafico guardado: {CARPETA_SALIDA}/histogramas.png")


# ==============================================================================
# SECCION 6: GRAFICOS DE DISPERSION

# El grafico de dispersion (scatter plot) representa cada observacion como un
# punto en un plano bidimensional, con una variable en cada eje. Permite:
#   - Detectar correlaciones lineales positivas o negativas entre dos variables.
#   - Identificar relaciones no lineales o patrones curvilineos.
#   - Visualizar grupos o clusters naturales en los datos.
#
# El pairplot genera automaticamente todos los pares posibles de columnas
# numericas. En la diagonal principal muestra la distribucion individual de
# cada variable mediante una curva de densidad KDE (Kernel Density Estimation).
# ==============================================================================

print("\n" + "=" * 60)
print("6. GRAFICOS DE DISPERSION ENTRE COLUMNAS NUMERICAS")
print("=" * 60)

if len(num_cols) >= 2:

    # Tomar hasta 500 filas para agilizar el renderizado del pairplot.
    # random_state=42 garantiza que siempre se tome la misma muestra (reproducibilidad).
    pair_df   = df[num_cols].sample(min(500, len(df)), random_state=42)
    pair_plot = sns.pairplot(pair_df, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 20})
    pair_plot.fig.suptitle("Graficos de dispersion entre variables numericas", y=1.02, fontsize=14)
    pair_plot.fig.savefig(f"{CARPETA_SALIDA}/dispersion_pairplot.png", bbox_inches="tight")
    plt.close()
    print(f"-> Grafico guardado: {CARPETA_SALIDA}/dispersion_pairplot.png")

    # Grafico de dispersion individual entre las dos primeras columnas numericas.
    # Sirve para analizar en detalle la relacion entre un par particular de variables.
    col_x, col_y = num_cols[0], num_cols[1]
    plt.figure(figsize=(7, 5))
    plt.scatter(df[col_x], df[col_y], alpha=0.5, color="steelblue", edgecolors="white", s=30)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f"Dispersion: {col_x} vs {col_y}")
    plt.tight_layout()
    plt.savefig(f"{CARPETA_SALIDA}/dispersion_{col_x}_vs_{col_y}.png", bbox_inches="tight")
    plt.close()
    print(f"-> Grafico guardado: {CARPETA_SALIDA}/dispersion_{col_x}_vs_{col_y}.png")


# ==============================================================================
# SECCION 7: TRANSFORMACIONES DE COLUMNAS

# Para que los modelos de ML procesen los datos correctamente, se aplican:
#
#   7.1 Label Encoding    : asigna un entero unico a cada categoria de texto.
#                           Introduce orden implicito, por lo que se usa solo
#                           en columnas de alta cardinalidad (> 10 valores).
#
#   7.2 One Hot Encoding  : crea una columna binaria (0/1) por cada categoria.
#                           No introduce orden. Ideal para baja cardinalidad (<=10).
#
#   7.3 Binary Encoding   : representa el entero del Label Encoding en binario.
#                           Genera menos columnas que OHE para alta cardinalidad.
#
#   7.4 Correlacion       : detecta pares de variables con relacion lineal fuerte.
#                           Columnas con |r| > 0.9 son redundantes (multicolinealidad)
#                           y se recomienda eliminar una de ellas.
#
#   7.5 Escalamiento      : ajusta la escala de las variables numericas.
#                           Min-Max -> rango [0, 1].
#                           StandardScaler -> media=0 y desviacion estandar=1.
#
#   7.6 Transformacion    : aplica log1p a columnas con sesgo positivo (skewness > 1)
#       logaritmica         para aproximar su distribucion a una forma normal.
# ==============================================================================

print("\n" + "=" * 60)
print("7. TRANSFORMACIONES DE COLUMNAS")
print("=" * 60)

# Trabajar sobre una copia del DataFrame para conservar los datos originales intactos
df_transform = df.copy()


# --- 7.1 Label Encoding -------------------------------------------------------
# Reemplaza cada categoria con un entero unico (0, 1, 2, ...).
# Ejemplo: ["Drama", "Comedia", "Terror"] -> [0, 1, 2]
# Se aplica solo a columnas de alta cardinalidad (> 10 categorias) porque One Hot
# Encoding generaria demasiadas columnas nuevas para estas variables.

print("\n-> 7.1 Label Encoding (columnas con mas de 10 categorias):")
le = LabelEncoder()
label_encoded_cols = []  # Lista para recordar que columnas fueron procesadas (usadas en 7.3)

for col in cat_cols:
    if df_transform[col].nunique() > 10:
        # Transformar la columna y guardar el resultado con sufijo "_le"
        df_transform[col + "_le"] = le.fit_transform(df_transform[col].astype(str))
        label_encoded_cols.append(col)
        print(f"   {col} ({df_transform[col].nunique()} categorias) -> {col}_le")


# --- 7.2 One Hot Encoding -----------------------------------------------------
# Crea una columna binaria por cada valor unico de la columna original.
# Ejemplo: columna "genero" con ["Accion", "Drama"] genera:
#           genero_Accion (0 o 1) y genero_Drama (0 o 1)
# drop_first=False conserva todas las categorias generadas sin eliminar ninguna.

print("\n-> 7.2 One Hot Encoding (columnas con 10 o menos categorias):")
ohe_cols = [col for col in cat_cols if df_transform[col].nunique() <= 10]

if ohe_cols:
    # pd.get_dummies genera automaticamente todas las columnas binarias
    dummies = pd.get_dummies(df_transform[ohe_cols], prefix=ohe_cols, drop_first=False)
    # Agregar las columnas generadas al DataFrame de transformaciones
    df_transform = pd.concat([df_transform, dummies], axis=1)
    print(f"   Columnas procesadas        : {ohe_cols}")
    print(f"   Nuevas columnas generadas  : {list(dummies.columns)[:10]} ...")
else:
    print("   No se encontraron columnas con cardinalidad <= 10.")


# --- 7.3 Binary Encoding ------------------------------------------------------
# Toma el valor entero asignado por Label Encoding y lo representa en bits.
# Ejemplo: valor 6 en binario = 110 -> genera columnas _bin_0=0, _bin_1=1, _bin_2=1
# Con n categorias se necesitan ceil(log2(n)) columnas, mucho menos que One Hot Encoding.

print("\n-> 7.3 Binary Encoding (sobre las columnas ya label-encoded):")

# Se limita a las primeras 2 columnas procesadas para no saturar el DataFrame
for col in label_encoded_cols[:2]:
    max_val = df_transform[col + "_le"].max()    # Valor entero maximo de la columna
    # Calcular cuantos bits son necesarios para representar todos los valores enteros
    n_bits = max(1, int(np.ceil(np.log2(max_val + 1))))

    for bit in range(n_bits):
        # >> bit : desplaza los bits del numero 'bit' posiciones hacia la derecha
        # & 1    : extrae el bit menos significativo del resultado (siempre 0 o 1)
        df_transform[f"{col}_bin_{bit}"] = (df_transform[col + "_le"] >> bit) & 1

    print(f"   {col}_le -> {n_bits} columnas binarias ({col}_bin_0 ... {col}_bin_{n_bits-1})")
