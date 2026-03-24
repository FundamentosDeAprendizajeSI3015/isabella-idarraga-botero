# # Agrupamiento (Clustering) - Dataset Sintético FIRE UdeA
# 
# **SI3015 - Fundamentos de Aprendizaje Automático**
# 
# Aplicamos K-Means y DBSCAN sobre el dataset `dataset_sintetico_para_modelado.csv` (variables financieras sintéticas, 500 observaciones).
# 
# **Proceso:**
# 1. Carga e inspección del dataset
# 2. K-Means con K=2
# 3. Método del codo para hallar el mejor K
# 4. K-Means con el K óptimo
# 5. DBSCAN
# 6. UMAP
# 7. Comparación con el label real

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

plt.rc('font', family='serif', size=12)
sns.set_theme(style='whitegrid', palette='Set2')

random_state = 42

# ## 1. Carga e inspección del dataset

from pathlib import Path

candidate_paths = [
    Path('lecture8/dataset_sintetico_para_modelado.csv'),
    Path('../lecture8/dataset_sintetico_para_modelado.csv'),
    Path('dataset_sintetico_para_modelado.csv'),
]

for path in candidate_paths:
    if path.exists():
        df = pd.read_csv(path)
        print(f'Archivo cargado desde: {path}')
        break
else:
    raise FileNotFoundError(
        f'No se encontró dataset_sintetico_para_modelado.csv. '
        f'Ejecuta primero lecture8/02_pipeline_analisis_sintetico.py'
    )

print(f'Shape: {df.shape}')
df.head()

# Separamos las features (excluimos 'label' — es la variable supervisada)
feature_cols = [c for c in df.columns if c != 'label']

data = df[feature_cols].values
print(f'Features: {feature_cols}')
print(f'Shape datos: {data.shape}')

# Distribución de las features
df[feature_cols].describe()

# ## 1.5. Análisis del Labeling
# 
# Antes de clusterizar, analizamos la calidad del labeling original:
# - Distribución de clases (¿están balanceadas?)
# - Correlación de features con el label
# - Separabilidad visual de las clases
# - Luego compararemos con clustering para validar si el labeling es correcto

# Distribución del label
print('=== DISTRIBUCIÓN DEL LABEL ===')
print(df['label'].value_counts())
print(f'\nProporción:')
print(df['label'].value_counts(normalize=True).round(3))

# Visualizar balance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico de barras
vc = df['label'].value_counts()
axes[0].bar(vc.index.astype(str), vc.values, color=['#3498db', '#e74c3c'], alpha=0.8, width=0.6)
axes[0].set_title('Frecuencia del Label', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Conteo')
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Gráfico de torta
axes[1].pie(vc.values, labels=[f'Label {k}' for k in vc.index], autopct='%1.1f%%',
            colors=['#3498db', '#e74c3c'], startangle=90)
axes[1].set_title('Proporción del Label', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Balance
balance_ratio = df['label'].value_counts().min() / df['label'].value_counts().max()
print(f'\nBalance: {balance_ratio:.2%} (mín/máx)')
if balance_ratio > 0.4:
    print('✓ Clases relativamente balanceadas')
else:
    print('⚠ Clases desbalanceadas')

# Correlación de features con el label
print('=== CORRELACIÓN DE FEATURES CON LABEL ===')
corr_label = df[feature_cols + ['label']].corr()['label'][:-1].sort_values(ascending=False)
print(corr_label.round(3))

# Visualizar
fig, ax = plt.subplots(figsize=(9, 5))
colores = ['#27ae60' if x > 0 else '#e74c3c' for x in corr_label.values]
ax.barh(corr_label.index, corr_label.values, color=colores, alpha=0.8)
ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_title('Correlación de Features con el Label (Pearson)', fontsize=12, fontweight='bold')
ax.set_xlabel('Correlación')
for i, v in enumerate(corr_label.values):
    ax.text(v + 0.02 if v > 0 else v - 0.02, i, f'{v:.3f}', va='center',
            ha='left' if v > 0 else 'right', fontsize=9)
plt.tight_layout()
plt.show()

# Cuáles features son más discriminativas
strong_corr = corr_label[abs(corr_label) > 0.3].index.tolist()
print(f'\nFeatures con correlación fuerte (|r| > 0.3): {strong_corr}')

# Estadísticas por clase
print('=== ESTADÍSTICAS POR CLASE ===')
for label in sorted(df['label'].unique()):
    print(f'\nLabel = {label}:')
    print(df[df['label'] == label][feature_cols].describe().loc[['mean', 'std', 'min', 'max']].round(3))

# Boxplots para visualizar separabilidad
n_features = len(feature_cols)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten()

for i, feat in enumerate(feature_cols):
    ax = axes[i]
    grupos = [df[df['label'] == k][feat].dropna() for k in sorted(df['label'].unique())]
    bp = ax.boxplot(grupos, labels=[f'Label {k}' for k in sorted(df['label'].unique())],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f'{feat}', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribución de Features por Clase (Boxplots)', fontsize=13, y=1.00)
plt.tight_layout()
plt.show()

print('\n💡 Si hay separación clara entre cajas → el label es discriminativo')

# Índice de separabilidad
print('=== ÍNDICE DE SEPARABILIDAD (Distancias teóricas) ===')
from sklearn.preprocessing import StandardScaler as SS
from scipy.spatial.distance import cdist

scaler_temp = SS()
data_scaled_temp = scaler_temp.fit_transform(df[feature_cols])

# Distancia intra-clase vs inter-clase
intra_distances = []
inter_distances = []

for label_val in df['label'].unique():
    mask = df['label'].values == label_val
    intra_distances.append(cdist(data_scaled_temp[mask], data_scaled_temp[mask]).mean())

for l1 in df['label'].unique():
    for l2 in df['label'].unique():
        if l1 < l2:
            mask1 = df['label'].values == l1
            mask2 = df['label'].values == l2
            inter_distances.append(cdist(data_scaled_temp[mask1], data_scaled_temp[mask2]).mean())

intra_mean = np.mean(intra_distances)
inter_mean = np.mean(inter_distances)
separabilidad = (inter_mean - intra_mean) / inter_mean if inter_mean > 0 else 0

print(f'Distancia intra-clase promedio: {intra_mean:.4f}')
print(f'Distancia inter-clase promedio: {inter_mean:.4f}')
print(f'Índice de separabilidad: {separabilidad:.4f}')
if separabilidad > 0.3:
    print('✓ Buena separabilidad entre clases')
else:
    print('⚠ Separabilidad moderada o baja')

# ## 2. Pipeline de preprocesamiento
# 
# Escalamos con `StandardScaler` dado que las variables tienen escalas muy distintas (ej. `cfo` en millones vs `liquidez` entre 0 y 3).

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, np.arange(data.shape[1])),
])

# También escalamos los datos una vez para reusar en DBSCAN y UMAP
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# ## 3. K-Means con K = 2

clu_kmeans_2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clustering', KMeans(n_clusters=2, random_state=random_state))
])

clu_kmeans_2.fit(data)
labels_k2 = clu_kmeans_2['clustering'].labels_
print(f'Inercia con K=2: {clu_kmeans_2["clustering"].inertia_:.2f}')
print(f'Distribución de clusters: {np.unique(labels_k2, return_counts=True)}')

# Visualizamos con los pares de variables disponibles
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df['liquidez'], df['dias_efectivo'], c=labels_k2, cmap='Set1', alpha=0.6, s=15)
axes[0].set_xlabel('Liquidez')
axes[0].set_ylabel('Días de Efectivo')
axes[0].set_title('K-Means K=2: Liquidez vs Días de Efectivo')

axes[1].scatter(df['liquidez'], df['cfo'], c=labels_k2, cmap='Set1', alpha=0.6, s=15)
axes[1].set_xlabel('Liquidez')
axes[1].set_ylabel('CFO')
axes[1].set_title('K-Means K=2: Liquidez vs CFO')

plt.tight_layout()
plt.show()

# ## 4. Método del codo — Hallar el K óptimo

inert = []
k_range = list(range(1, 11))

for k in k_range:
    clu = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clustering', KMeans(n_clusters=k, random_state=random_state))
    ])
    clu.fit(data)
    inert.append(clu['clustering'].inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_range, inert, marker='o')
ax.set_xlabel('Número de clusters K')
ax.set_ylabel('Inercia')
ax.set_title('Método del Codo - Dataset Sintético FIRE UdeA')
ax.set_xticks(k_range)
plt.tight_layout()
plt.show()

# ## 5. K-Means con el K óptimo
# 
# Observa el gráfico anterior y ajusta `K_OPTIMO` al valor donde se encuentra el "codo".

K_OPTIMO = 3  # <-- ajusta según el gráfico del codo

clu_kmeans_opt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clustering', KMeans(n_clusters=K_OPTIMO, random_state=random_state))
])

clu_kmeans_opt.fit(data)
labels_opt = clu_kmeans_opt['clustering'].labels_
print(f'Inercia con K={K_OPTIMO}: {clu_kmeans_opt["clustering"].inertia_:.2f}')
print(f'Distribución de clusters: {np.unique(labels_opt, return_counts=True)}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df['liquidez'], df['dias_efectivo'], c=labels_opt, cmap='Set1', alpha=0.6, s=15)
axes[0].set_xlabel('Liquidez')
axes[0].set_ylabel('Días de Efectivo')
axes[0].set_title(f'K-Means K={K_OPTIMO}: Liquidez vs Días de Efectivo')

axes[1].scatter(df['liquidez'], df['cfo'], c=labels_opt, cmap='Set1', alpha=0.6, s=15)
axes[1].set_xlabel('Liquidez')
axes[1].set_ylabel('CFO')
axes[1].set_title(f'K-Means K={K_OPTIMO}: Liquidez vs CFO')

plt.tight_layout()
plt.show()

# ## 6. DBSCAN
# 
# DBSCAN no requiere especificar K y detecta outliers (etiqueta `-1`).

clu_dbscan = DBSCAN(eps=0.5, min_samples=10)
clu_dbscan.fit(data_scaled)

labels_dbscan = clu_dbscan.labels_
clusters_unicos, conteos = np.unique(labels_dbscan, return_counts=True)
print('Clusters encontrados (label=-1 son outliers):')
for c, n in zip(clusters_unicos, conteos):
    nombre = 'Outliers' if c == -1 else f'Cluster {c}'
    print(f'  {nombre}: {n} puntos')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df['liquidez'], df['dias_efectivo'], c=labels_dbscan, cmap='Set1', alpha=0.6, s=15)
axes[0].set_xlabel('Liquidez')
axes[0].set_ylabel('Días de Efectivo')
axes[0].set_title('DBSCAN: Liquidez vs Días de Efectivo')

axes[1].scatter(df['liquidez'], df['cfo'], c=labels_dbscan, cmap='Set1', alpha=0.6, s=15)
axes[1].set_xlabel('Liquidez')
axes[1].set_ylabel('CFO')
axes[1].set_title('DBSCAN: Liquidez vs CFO')

plt.tight_layout()
plt.show()

# ## 7. UMAP
# 
# UMAP reduce las dimensiones del dataset a 2D para visualizar la estructura real de los datos.
# Coloreamos el embedding con el **label real**, **K-Means** y **DBSCAN** para comparar.
# 
# > Si no tienes instalado umap: `pip install umap-learn`

import umap

reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=random_state)
embedding = reducer.fit_transform(data_scaled)

print(f'Shape embedding UMAP: {embedding.shape}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

configs = [
    (df['label'].values, 'Label real'),
    (labels_k2,          'K-Means K=2'),
    (labels_opt,         f'K-Means K={K_OPTIMO}'),
]

for ax, (labels, titulo) in zip(axes, configs):
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=labels, cmap='Set1', alpha=0.6, s=10)
    ax.set_title(titulo)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(sc, ax=ax)

plt.suptitle('UMAP - Dataset Sintético FIRE UdeA', fontsize=13)
plt.tight_layout()
plt.show()

# UMAP coloreado con DBSCAN
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                c=labels_dbscan, cmap='Set1', alpha=0.6, s=10)
ax.set_title('UMAP - DBSCAN (gris = outliers)')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
plt.colorbar(sc, ax=ax)
plt.tight_layout()
plt.show()

# ## 8. Comparación con el label real
# 
# Comparamos los clusters encontrados con la variable `label` original del dataset (no se usó en el clustering).

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (labels, titulo) in zip(axes, [
    (df['label'].values, 'Label real'),
    (labels_k2,          'K-Means K=2'),
    (labels_opt,         f'K-Means K={K_OPTIMO}'),
]):
    ax.scatter(df['liquidez'], df['dias_efectivo'], c=labels, cmap='Set1', alpha=0.6, s=15)
    ax.set_xlabel('Liquidez')
    ax.set_ylabel('Días de Efectivo')
    ax.set_title(titulo)

plt.suptitle('Comparación: label real vs clusters encontrados', fontsize=13)
plt.tight_layout()
plt.show()