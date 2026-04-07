# Análisis No Supervisado y Supervisado — Sistema de Predicción de Abandono de Lectura
# Isabella Idarraga | Fundamentos de Aprendizaje Automático
# 
# Este notebook implementa:
# 1. Análisis de clustering (K-Means, Fuzzy C-Means, Subtractive, DBSCAN, Jerárquico, GMM)
# 2. Re-evaluación de etiquetas mediante consenso de clusters
# 3. Modelos supervisados (Árbol de Decisión, Regresión Logística, Regresión Lineal)
# 4. Comparación entre modelos entrenados con etiquetas originales vs re-evaluadas

# ⚙️ CELDA DE CONFIGURACIÓN — Ajusta aquí el porcentaje del dataset

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════

# Porcentaje del dataset a usar (0.0 a 1.0)
# Usa 0.05 para pruebas rápidas, 1.0 para el análisis completo
SAMPLE_PCT = 1.0

RANDOM_STATE = 42

# Ruta al dataset
import os
DATA_PATH = os.path.join('..', 'Proyecto1', 'datos_transformados.csv')

# Número de clusters para K-Means, FCM, Jerárquico, GMM
N_CLUSTERS = 3

# Parámetros DBSCAN
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 10

# Profundidad máxima del árbol de decisión
TREE_MAX_DEPTH = 5

print(f"Configuración cargada:")
print(f"  SAMPLE_PCT     : {SAMPLE_PCT*100:.0f}% del dataset")
print(f"  N_CLUSTERS     : {N_CLUSTERS}")
print(f"  RANDOM_STATE   : {RANDOM_STATE}")
print(f"  DATA_PATH      : {os.path.abspath(DATA_PATH)}")

# 1. Importaciones y Utilidades

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# Modelos supervisados
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression

# Fuzzy C-Means (requiere scikit-fuzzy)
try:
    import skfuzzy as fuzz
    FUZZY_AVAILABLE = True
    print("✔ scikit-fuzzy disponible")
except ImportError:
    FUZZY_AVAILABLE = False
    print("✗ scikit-fuzzy no instalado. Ejecuta: pip install scikit-fuzzy")

# Estilo de gráficas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('tab10')
PALETTE = sns.color_palette('tab10')

print("\nImportaciones completadas.")

# 2. Carga y Exploración de Datos

# ── Carga ──────────────────────────────────────────────────────
print("Cargando dataset...")
df_full = pd.read_csv(DATA_PATH)
print(f"Dataset completo: {df_full.shape[0]:,} filas × {df_full.shape[1]} columnas")

# ── Muestreo ───────────────────────────────────────────────────
if SAMPLE_PCT < 1.0:
    df = df_full.sample(frac=SAMPLE_PCT, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Muestra usada   : {df.shape[0]:,} filas ({SAMPLE_PCT*100:.0f}%)")
else:
    df = df_full.copy()
    print("Usando dataset completo")

print(f"\nDistribución del target 'abandono':")
print(df['abandono'].value_counts())
print(f"Tasa de abandono: {df['abandono'].mean()*100:.1f}%")

# ── Exploración rápida ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Distribución de Variables Clave', fontsize=16, fontweight='bold')

vars_plot = [
    ('duration_minutes', 'Duración (min)'),
    ('pages_read', 'Páginas leídas'),
    ('progress_end', 'Progreso final (%)'),
    ('dias_inactividad', 'Días inactividad'),
    ('velocidad_lectura', 'Velocidad lectura'),
    ('abandono_score', 'Score de abandono (NLP)')
]

for ax, (col, label) in zip(axes.flatten(), vars_plot):
    if col in df.columns:
        for val, color in zip([0, 1], ['steelblue', 'tomato']):
            subset = df[df['abandono'] == val][col].dropna()
            ax.hist(subset, bins=40, alpha=0.6, color=color,
                    label=f'abandono={val}')
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.legend(fontsize=8)
        ax.set_ylabel('Frecuencia')

plt.tight_layout()
plt.savefig('01_distribuciones_clave.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: 01_distribuciones_clave.png")

# 3. Selección de Features y Preprocesamiento para Clustering

# Features seleccionadas para clustering
# Usamos versiones escaladas donde existen, más features NLP
FEATURES_CLUSTER = [
    # Comportamiento de sesión (versiones escaladas)
    'duration_minutes_scaled',
    'pages_read_scaled',
    'completion_pct_start_scaled',
    'completion_pct_end_scaled',
    'velocidad_lectura_scaled',
    'ratio_progreso_scaled',
    'num_sesiones_scaled',
    # Comportamiento de usuario
    'tasa_abandono',
    'progreso_promedio',
    # Features NLP del libro
    'abandono_score_scaled',
    'engagement_score_scaled',
    'complejidad_score_scaled',
    'ritmo_score_scaled',
    'sentimiento_promedio_scaled',
]

# Filtrar solo columnas que existen
FEATURES_CLUSTER = [f for f in FEATURES_CLUSTER if f in df.columns]
print(f"Features para clustering: {len(FEATURES_CLUSTER)}")
for f in FEATURES_CLUSTER:
    print(f"  - {f}")

X_raw = df[FEATURES_CLUSTER].fillna(0).values
y_original = df['abandono'].values

print(f"\nX shape: {X_raw.shape}")
print(f"y shape: {y_original.shape}")

# ── PCA para visualización 2D y 3D ─────────────────────────────
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2 = pca2.fit_transform(X_raw)

pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
X_pca3 = pca3.fit_transform(X_raw)

var_exp_2 = pca2.explained_variance_ratio_.sum()
var_exp_3 = pca3.explained_variance_ratio_.sum()

print(f"Varianza explicada PCA 2D: {var_exp_2*100:.1f}%")
print(f"Varianza explicada PCA 3D: {var_exp_3*100:.1f}%")

# Gráfica de varianza explicada
pca_full = PCA(random_state=RANDOM_STATE).fit(X_raw)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Análisis PCA', fontsize=14, fontweight='bold')

# Varianza explicada por componente
n_comp = min(10, len(pca_full.explained_variance_ratio_))
axes[0].bar(range(1, n_comp+1), pca_full.explained_variance_ratio_[:n_comp]*100,
            color='steelblue', alpha=0.8)
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Varianza Explicada (%)')
axes[0].set_title('Varianza por Componente')
axes[0].set_xticks(range(1, n_comp+1))

# Varianza acumulada
cum_var = np.cumsum(pca_full.explained_variance_ratio_)*100
axes[1].plot(range(1, len(cum_var)+1), cum_var, 'o-', color='tomato', linewidth=2)
axes[1].axhline(80, linestyle='--', color='gray', alpha=0.7, label='80%')
axes[1].axhline(95, linestyle='--', color='navy', alpha=0.7, label='95%')
axes[1].set_xlabel('Número de Componentes')
axes[1].set_ylabel('Varianza Acumulada (%)')
axes[1].set_title('Varianza Acumulada')
axes[1].legend()

plt.tight_layout()
plt.savefig('02_pca_varianza.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualizar etiquetas originales en espacio PCA
fig, ax = plt.subplots(figsize=(10, 7))
for val, label, color in zip([0, 1], ['Completado', 'Abandonado'], ['steelblue', 'tomato']):
    mask = y_original == val
    ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
               c=color, label=label, alpha=0.4, s=10)
ax.set_title('Etiquetas Originales en Espacio PCA 2D', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
ax.legend(markerscale=2)
plt.tight_layout()
plt.savefig('03_etiquetas_originales_pca.png', dpi=150, bbox_inches='tight')
plt.show()

# 4. Análisis No Supervisado

# 4.1 K-Means

# ── Método del Codo ────────────────────────────────────────────
print("Calculando método del codo...")
k_range = range(2, 11)
inertias = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_raw)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_raw, labels, sample_size=min(5000, len(X_raw))))
    print(f"  k={k}: inercia={km.inertia_:.0f}, silhouette={silhouettes[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('K-Means: Selección del Número Óptimo de Clusters', fontsize=14, fontweight='bold')

axes[0].plot(k_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (k)')
axes[0].set_ylabel('Inercia (WCSS)')
axes[0].set_title('Método del Codo')
axes[0].set_xticks(list(k_range))

axes[1].plot(k_range, silhouettes, 'o-', color='tomato', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por k')
axes[1].set_xticks(list(k_range))

# Marcar k óptimo
k_optimo = list(k_range)[np.argmax(silhouettes)]
axes[1].axvline(k_optimo, color='navy', linestyle='--', alpha=0.7,
                label=f'k óptimo = {k_optimo}')
axes[1].legend()

plt.tight_layout()
plt.savefig('04_kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nk óptimo según Silhouette: {k_optimo}")

# ── K-Means con N_CLUSTERS ─────────────────────────────────────
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
labels_kmeans = kmeans.fit_predict(X_raw)

sil_km = silhouette_score(X_raw, labels_kmeans, sample_size=min(5000, len(X_raw)))
db_km  = davies_bouldin_score(X_raw, labels_kmeans)
ch_km  = calinski_harabasz_score(X_raw, labels_kmeans)

print(f"K-Means (k={N_CLUSTERS}):")
print(f"  Silhouette Score       : {sil_km:.4f}  (más alto = mejor, máx=1)")
print(f"  Davies-Bouldin Score   : {db_km:.4f}  (más bajo = mejor)")
print(f"  Calinski-Harabasz Score: {ch_km:.2f} (más alto = mejor)")

# Distribución de clusters vs etiqueta original
print("\nDistribución de abandono por cluster:")
df_temp = pd.DataFrame({'cluster': labels_kmeans, 'abandono': y_original})
print(df_temp.groupby('cluster')['abandono'].agg(['mean','count']).round(3))

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f'K-Means (k={N_CLUSTERS}) — Visualización PCA 2D', fontsize=14, fontweight='bold')

# Clusters
scatter = axes[0].scatter(X_pca2[:, 0], X_pca2[:, 1],
                           c=labels_kmeans, cmap='tab10', alpha=0.4, s=10)
centers_pca = pca2.transform(kmeans.cluster_centers_)
axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1],
                c='black', marker='X', s=200, zorder=5, label='Centroides')
axes[0].set_title('Clusters K-Means')
axes[0].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter, ax=axes[0], label='Cluster')
axes[0].legend()

# Etiquetas originales coloreadas por cluster
for c in range(N_CLUSTERS):
    mask = labels_kmeans == c
    tasa = y_original[mask].mean()
    axes[1].scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                    alpha=0.4, s=10, label=f'Cluster {c} (abandono={tasa:.2f})')
axes[1].set_title('Tasa de Abandono por Cluster')
axes[1].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].legend(markerscale=2, fontsize=9)

plt.tight_layout()
plt.savefig('05_kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.2 Fuzzy C-Means

if FUZZY_AVAILABLE:
    print("Ejecutando Fuzzy C-Means...")
    # skfuzzy requiere X transpuesto (features × samples)
    X_T = X_raw.T

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_T, c=N_CLUSTERS, m=2.0,
        error=0.005, maxiter=1000,
        init=None, seed=RANDOM_STATE
    )

    # Etiqueta dura: cluster con mayor membresía
    labels_fcm = np.argmax(u, axis=0)
    # Membresía máxima (cuán seguro está el modelo)
    max_membership = np.max(u, axis=0)

    sil_fcm = silhouette_score(X_raw, labels_fcm, sample_size=min(5000, len(X_raw)))
    db_fcm  = davies_bouldin_score(X_raw, labels_fcm)
    ch_fcm  = calinski_harabasz_score(X_raw, labels_fcm)

    print(f"Fuzzy C-Means (c={N_CLUSTERS}):")
    print(f"  FPC (Fuzzy Partition Coefficient): {fpc:.4f}  (más alto = mejor, máx=1)")
    print(f"  Silhouette Score                 : {sil_fcm:.4f}")
    print(f"  Davies-Bouldin Score             : {db_fcm:.4f}")
    print(f"  Membresía promedio máxima        : {max_membership.mean():.4f}")

    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Fuzzy C-Means (c={N_CLUSTERS})', fontsize=14, fontweight='bold')

    # Clusters con intensidad = membresía máxima
    scatter = axes[0].scatter(X_pca2[:, 0], X_pca2[:, 1],
                               c=labels_fcm, cmap='tab10',
                               alpha=max_membership * 0.8, s=10)
    cntr_pca = pca2.transform(cntr)
    axes[0].scatter(cntr_pca[:, 0], cntr_pca[:, 1],
                    c='black', marker='X', s=200, zorder=5, label='Centroides')
    axes[0].set_title('Clusters FCM (alfa = membresía)')
    axes[0].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].legend()

    # Distribución de membresías
    axes[1].hist(max_membership, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
    axes[1].axvline(max_membership.mean(), color='tomato', linestyle='--',
                    label=f'Media = {max_membership.mean():.3f}')
    axes[1].set_xlabel('Membresía Máxima')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Certeza de Asignación')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('06_fuzzy_cmeans.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("FCM no disponible. Usando K-Means como sustituto.")
    labels_fcm = labels_kmeans.copy()
    sil_fcm = sil_km
    db_fcm  = db_km
    ch_fcm  = ch_km
    fpc     = 0.0
    print("Instala scikit-fuzzy con: pip install scikit-fuzzy")

# 4.3 Subtractive Clustering

class SubtractiveClustering:
    """
    Implementación de Subtractive Clustering (Chiu, 1994).
    Encuentra automáticamente el número de clusters basándose
    en densidad de puntos en el espacio de características.
    """
    def __init__(self, r_a=0.5, r_b=None, epsilon_upper=0.5,
                 epsilon_lower=0.15, max_clusters=15):
        self.r_a = r_a
        self.r_b = r_b if r_b else 1.5 * r_a
        self.epsilon_upper = epsilon_upper
        self.epsilon_lower = epsilon_lower
        self.max_clusters = max_clusters
        self.centers_norm_ = None
        self.centers_ = None
        self.n_clusters_ = 0

    def fit(self, X):
        X = np.array(X, dtype=float)
        X_min = X.min(axis=0)
        X_range = X.max(axis=0) - X_min
        X_range[X_range == 0] = 1  # evitar división por cero
        X_norm = (X - X_min) / X_range

        n = len(X_norm)
        # Densidad inicial para cada punto
        densities = np.zeros(n)
        for i in range(n):
            diff = X_norm - X_norm[i]
            dist_sq = np.sum(diff ** 2, axis=1)
            densities[i] = np.sum(np.exp(-4 * dist_sq / (self.r_a ** 2)))

        D_max_original = densities.max()
        centers_norm = []

        while len(centers_norm) < self.max_clusters:
            idx_max = int(np.argmax(densities))
            D_max = densities[idx_max]

            if len(centers_norm) == 0:
                accept = True
            else:
                ratio = D_max / D_max_original
                if ratio >= self.epsilon_upper:
                    accept = True
                elif ratio < self.epsilon_lower:
                    break
                else:
                    dists = [np.linalg.norm(X_norm[idx_max] - c)
                             for c in centers_norm]
                    d_min = min(dists)
                    if (d_min / self.r_a + ratio) >= 1:
                        accept = True
                    else:
                        densities[idx_max] = 0
                        continue

            if accept:
                centers_norm.append(X_norm[idx_max].copy())
                # Reducir densidad alrededor del nuevo centro
                diff = X_norm - X_norm[idx_max]
                dist_sq = np.sum(diff ** 2, axis=1)
                densities -= D_max * np.exp(-4 * dist_sq / (self.r_b ** 2))

        self.centers_norm_ = np.array(centers_norm)
        self.centers_ = self.centers_norm_ * X_range + X_min
        self.n_clusters_ = len(centers_norm)
        self._X_min = X_min
        self._X_range = X_range
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        labels = []
        for x in X:
            dists = np.linalg.norm(self.centers_ - x, axis=1)
            labels.append(int(np.argmin(dists)))
        return np.array(labels)

print("Clase SubtractiveClustering definida.")

# Subtractive Clustering es O(n²) — usar submuestra para encontrar centros
SUB_SAMPLE = min(3000, len(X_raw))
idx_sub = np.random.RandomState(RANDOM_STATE).choice(len(X_raw), SUB_SAMPLE, replace=False)
X_sub = X_raw[idx_sub]

print(f"Ejecutando Subtractive Clustering sobre {SUB_SAMPLE} muestras...")
sub_clust = SubtractiveClustering(r_a=0.5, epsilon_upper=0.5, epsilon_lower=0.15)
sub_clust.fit(X_sub)

print(f"Clusters encontrados automáticamente: {sub_clust.n_clusters_}")

# Asignar todos los puntos al centro más cercano
labels_sub = sub_clust.predict(X_raw)

if sub_clust.n_clusters_ > 1:
    sil_sub = silhouette_score(X_raw, labels_sub, sample_size=min(5000, len(X_raw)))
    db_sub  = davies_bouldin_score(X_raw, labels_sub)
    print(f"  Silhouette Score    : {sil_sub:.4f}")
    print(f"  Davies-Bouldin      : {db_sub:.4f}")
else:
    sil_sub = 0
    print("Solo 1 cluster encontrado. Ajusta r_a.")

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f'Subtractive Clustering — {sub_clust.n_clusters_} clusters encontrados',
             fontsize=14, fontweight='bold')

scatter = axes[0].scatter(X_pca2[:, 0], X_pca2[:, 1],
                           c=labels_sub, cmap='tab10', alpha=0.4, s=10)
centers_pca = pca2.transform(sub_clust.centers_)
axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1],
                c='black', marker='X', s=200, zorder=5, label='Centros')
axes[0].set_title('Clusters (todos los puntos)')
axes[0].set_xlabel(f'PC1'); axes[0].set_ylabel(f'PC2')
axes[0].legend()
plt.colorbar(scatter, ax=axes[0], label='Cluster')

# Distribución de abandono por cluster
df_sub = pd.DataFrame({'cluster': labels_sub, 'abandono': y_original})
tasa_sub = df_sub.groupby('cluster')['abandono'].mean().reset_index()
axes[1].bar(tasa_sub['cluster'].astype(str), tasa_sub['abandono'],
            color=PALETTE[:len(tasa_sub)], alpha=0.8, edgecolor='white')
axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('Tasa de Abandono')
axes[1].set_title('Tasa de Abandono por Cluster')
axes[1].axhline(y_original.mean(), color='red', linestyle='--',
                label=f'Media global = {y_original.mean():.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('07_subtractive_clustering.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.4 DBSCAN

# Curva k-distance para estimar eps óptimo
from sklearn.neighbors import NearestNeighbors
print("Calculando curva k-distance para DBSCAN...")
k_nn = DBSCAN_MIN_SAMPLES
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_raw), min(5000, len(X_raw)), replace=False)
nbrs = NearestNeighbors(n_neighbors=k_nn).fit(X_raw[sample_idx])
distances, _ = nbrs.kneighbors(X_raw[sample_idx])
k_distances = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_distances, color='steelblue', linewidth=1.5)
ax.axhline(DBSCAN_EPS, color='tomato', linestyle='--',
           label=f'eps configurado = {DBSCAN_EPS}')
ax.set_xlabel('Puntos ordenados')
ax.set_ylabel(f'Distancia al {k_nn}° vecino más cercano')
ax.set_title('Curva k-Distance (para seleccionar eps en DBSCAN)')
ax.legend()
plt.tight_layout()
plt.savefig('08_dbscan_kdistance.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Ejecutando DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
labels_dbscan = dbscan.fit_predict(X_raw)

n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = (labels_dbscan == -1).sum()
print(f"  Clusters encontrados: {n_clusters_db}")
print(f"  Puntos de ruido     : {n_noise} ({n_noise/len(labels_dbscan)*100:.1f}%)")

if n_clusters_db > 1:
    mask_valid = labels_dbscan != -1
    sil_db = silhouette_score(X_raw[mask_valid], labels_dbscan[mask_valid],
                              sample_size=min(5000, mask_valid.sum()))
    print(f"  Silhouette (sin ruido): {sil_db:.4f}")
else:
    sil_db = 0
    print("  Ajusta eps o min_samples para obtener múltiples clusters.")

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('DBSCAN — Clustering Basado en Densidad', fontsize=14, fontweight='bold')

# Mapa de colores: ruido en gris
unique_labels = sorted(set(labels_dbscan))
cmap_db = {l: ('lightgray' if l == -1 else PALETTE[l % 10]) for l in unique_labels}
colors_db = [cmap_db[l] for l in labels_dbscan]

axes[0].scatter(X_pca2[:, 0], X_pca2[:, 1], c=colors_db, alpha=0.4, s=8)
axes[0].set_title(f'Clusters DBSCAN ({n_clusters_db} clusters, {n_noise} ruido)')
axes[0].set_xlabel(f'PC1'); axes[0].set_ylabel(f'PC2')

# Conteo por cluster
unique, counts = np.unique(labels_dbscan, return_counts=True)
labels_str = [f'Ruido' if l == -1 else f'C{l}' for l in unique]
bar_colors = ['lightgray' if l == -1 else PALETTE[i % 10] for i, l in enumerate(unique)]
axes[1].bar(labels_str, counts, color=bar_colors, edgecolor='white')
axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('Número de Puntos')
axes[1].set_title('Tamaño de Clusters')

plt.tight_layout()
plt.savefig('09_dbscan_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.5 Clustering Jerárquico (Agglomerative)

# Dendrograma sobre submuestra
from scipy.cluster.hierarchy import dendrogram, linkage

n_dendro = min(500, len(X_raw))
idx_dendro = np.random.RandomState(RANDOM_STATE).choice(len(X_raw), n_dendro, replace=False)
Z = linkage(X_raw[idx_dendro], method='ward')

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
           leaf_rotation=45, leaf_font_size=8,
           color_threshold=Z[-N_CLUSTERS+1, 2])
ax.set_title(f'Dendrograma Jerárquico (Ward) — Submuestra {n_dendro} puntos',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Muestra')
ax.set_ylabel('Distancia')
ax.axhline(Z[-N_CLUSTERS+1, 2], color='tomato', linestyle='--',
           label=f'Corte para {N_CLUSTERS} clusters')
ax.legend()
plt.tight_layout()
plt.savefig('10_dendrograma.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Ejecutando Agglomerative Clustering (k={N_CLUSTERS}, linkage=ward)...")

# Ward necesita matriz de distancias completa → imposible con 388k filas
# Solución: entrenar sobre submuestra, asignar resto al centroide más cercano
N_AGGLO_SAMPLE = min(10000, len(X_raw))
idx_agglo = np.random.RandomState(RANDOM_STATE).choice(len(X_raw), N_AGGLO_SAMPLE, replace=False)
X_agglo_sub = X_raw[idx_agglo]

agglo = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
labels_sub_agglo = agglo.fit_predict(X_agglo_sub)

# Calcular centroides de los clusters sobre la submuestra
centroides_agglo = np.array([
    X_agglo_sub[labels_sub_agglo == c].mean(axis=0)
    for c in range(N_CLUSTERS)
])

# Asignar todos los puntos al centroide más cercano
from sklearn.metrics import pairwise_distances_argmin
labels_agglo = pairwise_distances_argmin(X_raw, centroides_agglo)

sil_ag = silhouette_score(X_raw, labels_agglo, sample_size=min(5000, len(X_raw)))
db_ag  = davies_bouldin_score(X_raw, labels_agglo)
print(f"  Silhouette Score  : {sil_ag:.4f}")
print(f"  Davies-Bouldin    : {db_ag:.4f}")

fig, ax = plt.subplots(figsize=(10, 7))
for c in range(N_CLUSTERS):
    mask = labels_agglo == c
    tasa = y_original[mask].mean()
    ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
               alpha=0.4, s=10, label=f'Cluster {c} (abandono={tasa:.2f})')
ax.set_title(f'Clustering Jerárquico (k={N_CLUSTERS})', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1'); ax.set_ylabel(f'PC2')
ax.legend(markerscale=2)
plt.tight_layout()
plt.savefig('11_jerarquico_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.6 Gaussian Mixture Model (GMM)

# Seleccionar número de componentes por BIC
print("Calculando BIC para GMM...")
n_comp_range = range(2, 8)
bic_scores = []
aic_scores = []
for n in n_comp_range:
    gmm_tmp = GaussianMixture(n_components=n, random_state=RANDOM_STATE, max_iter=100)
    gmm_tmp.fit(X_raw)
    bic_scores.append(gmm_tmp.bic(X_raw))
    aic_scores.append(gmm_tmp.aic(X_raw))
    print(f"  n={n}: BIC={bic_scores[-1]:.0f}, AIC={aic_scores[-1]:.0f}")

n_optimo_gmm = list(n_comp_range)[np.argmin(bic_scores)]
print(f"\nn óptimo por BIC: {n_optimo_gmm}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(n_comp_range, bic_scores, 'o-', label='BIC', color='steelblue', linewidth=2)
ax.plot(n_comp_range, aic_scores, 's-', label='AIC', color='tomato', linewidth=2)
ax.axvline(n_optimo_gmm, linestyle='--', color='gray', alpha=0.7,
           label=f'n óptimo BIC = {n_optimo_gmm}')
ax.set_xlabel('Número de Componentes')
ax.set_ylabel('Score')
ax.set_title('GMM — Selección de Componentes por BIC/AIC')
ax.legend()
plt.tight_layout()
plt.savefig('12_gmm_bic.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Ejecutando GMM con {N_CLUSTERS} componentes...")
gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE,
                      covariance_type='full', max_iter=200)
gmm.fit(X_raw)
labels_gmm = gmm.predict(X_raw)
proba_gmm = gmm.predict_proba(X_raw)
max_proba = proba_gmm.max(axis=1)

sil_gmm = silhouette_score(X_raw, labels_gmm, sample_size=min(5000, len(X_raw)))
db_gmm  = davies_bouldin_score(X_raw, labels_gmm)
print(f"  Silhouette Score : {sil_gmm:.4f}")
print(f"  Davies-Bouldin   : {db_gmm:.4f}")
print(f"  Certeza promedio : {max_proba.mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f'GMM ({N_CLUSTERS} componentes)', fontsize=14, fontweight='bold')

scatter = axes[0].scatter(X_pca2[:, 0], X_pca2[:, 1],
                           c=labels_gmm, cmap='tab10',
                           alpha=max_proba * 0.8, s=10)
axes[0].set_title('Clusters GMM (alfa = probabilidad)')
axes[0].set_xlabel(f'PC1'); axes[0].set_ylabel(f'PC2')
plt.colorbar(scatter, ax=axes[0], label='Componente')

axes[1].hist(max_proba, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
axes[1].axvline(max_proba.mean(), color='tomato', linestyle='--',
                label=f'Media = {max_proba.mean():.3f}')
axes[1].set_xlabel('Probabilidad Máxima de Pertenencia')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Certeza de Asignación (GMM)')
axes[1].legend()

plt.tight_layout()
plt.savefig('13_gmm_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.7 Comparación de Métodos de Clustering

# Tabla comparativa
resultados_clustering = {
    'K-Means':           {'Silhouette': sil_km,  'Davies-Bouldin': db_km,
                          'Calinski-Harabasz': ch_km, 'N Clusters': N_CLUSTERS},
    'Fuzzy C-Means':     {'Silhouette': sil_fcm, 'Davies-Bouldin': db_fcm,
                          'Calinski-Harabasz': ch_fcm if FUZZY_AVAILABLE else 0, 'N Clusters': N_CLUSTERS},
    'Subtractive':       {'Silhouette': sil_sub, 'Davies-Bouldin': db_sub if sub_clust.n_clusters_>1 else 0,
                          'Calinski-Harabasz': calinski_harabasz_score(X_raw, labels_sub) if sub_clust.n_clusters_>1 else 0,
                          'N Clusters': sub_clust.n_clusters_},
    'DBSCAN':            {'Silhouette': sil_db,  'Davies-Bouldin': davies_bouldin_score(X_raw[labels_dbscan!=-1], labels_dbscan[labels_dbscan!=-1]) if n_clusters_db>1 else 0,
                          'Calinski-Harabasz': 0, 'N Clusters': n_clusters_db},
    'Jerárquico (Ward)': {'Silhouette': sil_ag,  'Davies-Bouldin': db_ag,
                          'Calinski-Harabasz': calinski_harabasz_score(X_raw, labels_agglo), 'N Clusters': N_CLUSTERS},
    'GMM':               {'Silhouette': sil_gmm, 'Davies-Bouldin': db_gmm,
                          'Calinski-Harabasz': calinski_harabasz_score(X_raw, labels_gmm), 'N Clusters': N_CLUSTERS},
}

df_comp = pd.DataFrame(resultados_clustering).T.round(4)
print("Comparación de Métodos de Clustering:")
print(df_comp.to_string())

# Gráficas comparativas
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparación de Métricas de Clustering', fontsize=14, fontweight='bold')

metodos = list(resultados_clustering.keys())
sils = [resultados_clustering[m]['Silhouette'] for m in metodos]
dbs  = [resultados_clustering[m]['Davies-Bouldin'] for m in metodos]
chs  = [resultados_clustering[m]['Calinski-Harabasz'] for m in metodos]

axes[0].barh(metodos, sils, color=PALETTE[:len(metodos)], alpha=0.85, edgecolor='white')
axes[0].set_title('Silhouette Score (↑ mejor)')
axes[0].set_xlabel('Score')
axes[0].axvline(0, color='black', linewidth=0.5)

axes[1].barh(metodos, dbs, color=PALETTE[:len(metodos)], alpha=0.85, edgecolor='white')
axes[1].set_title('Davies-Bouldin (↓ mejor)')
axes[1].set_xlabel('Score')

axes[2].barh(metodos, chs, color=PALETTE[:len(metodos)], alpha=0.85, edgecolor='white')
axes[2].set_title('Calinski-Harabasz (↑ mejor)')
axes[2].set_xlabel('Score')

for ax in axes:
    ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.savefig('14_comparacion_clustering.png', dpi=150, bbox_inches='tight')
plt.show()

# Panel comparativo visual: todos los métodos en una figura
all_labels = [
    ('K-Means', labels_kmeans),
    ('Fuzzy C-Means', labels_fcm),
    ('Subtractive', labels_sub),
    ('DBSCAN', labels_dbscan),
    ('Jerárquico', labels_agglo),
    ('GMM', labels_gmm),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Panel Comparativo: Todos los Métodos de Clustering (PCA 2D)',
             fontsize=15, fontweight='bold')

for ax, (nombre, labels) in zip(axes.flatten(), all_labels):
    unique_l = sorted(set(labels))
    for i, l in enumerate(unique_l):
        mask = labels == l
        color = 'lightgray' if l == -1 else PALETTE[i % 10]
        lbl = 'Ruido' if l == -1 else f'C{l}'
        ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                   c=[color], alpha=0.4, s=6, label=lbl)
    ax.set_title(nombre, fontsize=11, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=8); ax.set_ylabel('PC2', fontsize=8)
    if len(unique_l) <= 6:
        ax.legend(markerscale=2, fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig('15_panel_todos_clustering.png', dpi=150, bbox_inches='tight')
plt.show()

# 5. Re-evaluación de Etiquetas

# Utilizamos el consenso de los métodos de clustering para identificar posibles etiquetas mal puestas. Si la mayoría de los algoritmos asigna un punto a un cluster con características opuestas a su etiqueta, se considera candidato a re-etiquetar.

# Para cada método, crear una "etiqueta predicha" basada en
# la tasa de abandono dominante del cluster al que pertenece el punto

def cluster_to_binary(labels, y_true, threshold=0.5):
    """Mapea clusters a 0/1 según la tasa de abandono de cada cluster."""
    result = np.zeros_like(y_true)
    for c in np.unique(labels):
        if c == -1:  # ruido DBSCAN → usar etiqueta original
            mask = labels == c
            result[mask] = y_true[mask]
            continue
        mask = labels == c
        tasa = y_true[mask].mean()
        result[mask] = 1 if tasa >= threshold else 0
    return result

# Predicciones binarias de cada método
pred_km    = cluster_to_binary(labels_kmeans,  y_original)
pred_fcm   = cluster_to_binary(labels_fcm,     y_original)
pred_sub   = cluster_to_binary(labels_sub,     y_original)
pred_db    = cluster_to_binary(labels_dbscan,  y_original)
pred_ag    = cluster_to_binary(labels_agglo,   y_original)
pred_gmm   = cluster_to_binary(labels_gmm,     y_original)

# Consenso: voto mayoritario
votes = np.stack([pred_km, pred_fcm, pred_sub, pred_db, pred_ag, pred_gmm], axis=1)
votes_sum = votes.sum(axis=1)  # cuántos métodos dicen "abandono=1"
y_consenso = (votes_sum >= 3).astype(int)  # mayoría simple (>=3 de 6)

# Puntos donde el consenso difiere de la etiqueta original
discrepancias = (y_consenso != y_original)
n_discrepancias = discrepancias.sum()
pct_discrepancias = n_discrepancias / len(y_original) * 100

print(f"Análisis de Re-evaluación de Etiquetas:")
print(f"  Total puntos              : {len(y_original):,}")
print(f"  Discrepancias encontradas : {n_discrepancias:,} ({pct_discrepancias:.1f}%)")
print(f"  (Esperado: ~30% según la literatura)")
print(f"\n  Distribución de votos:")
for v in range(7):
    cnt = (votes_sum == v).sum()
    print(f"    {v}/6 métodos dicen abandono=1: {cnt:,} puntos ({cnt/len(votes_sum)*100:.1f}%)")

# Crear dataset con etiquetas re-evaluadas
y_nuevo = y_consenso.copy()

print(f"Distribución etiquetas ORIGINALES:")
vals, cnts = np.unique(y_original, return_counts=True)
for v, c in zip(vals, cnts):
    print(f"  abandono={v}: {c:,} ({c/len(y_original)*100:.1f}%)")

print(f"\nDistribución etiquetas RE-EVALUADAS:")
vals, cnts = np.unique(y_nuevo, return_counts=True)
for v, c in zip(vals, cnts):
    print(f"  abandono={v}: {c:,} ({c/len(y_nuevo)*100:.1f}%)")

# Visualización de re-evaluación
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Re-evaluación de Etiquetas por Consenso de Clustering',
             fontsize=14, fontweight='bold')

# Etiquetas originales
for val, color in zip([0, 1], ['steelblue', 'tomato']):
    mask = y_original == val
    axes[0].scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=color, alpha=0.3,
                    s=8, label=f'abandono={val}')
axes[0].set_title('Etiquetas Originales')
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
axes[0].legend(markerscale=2)

# Etiquetas re-evaluadas
for val, color in zip([0, 1], ['steelblue', 'tomato']):
    mask = y_nuevo == val
    axes[1].scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=color, alpha=0.3,
                    s=8, label=f'abandono={val}')
axes[1].set_title('Etiquetas Re-evaluadas (Consenso)')
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].legend(markerscale=2)

# Puntos cambiados
mask_cambiados = discrepancias
mask_iguales = ~discrepancias
axes[2].scatter(X_pca2[mask_iguales, 0], X_pca2[mask_iguales, 1],
                c='steelblue', alpha=0.2, s=6, label=f'Sin cambio ({mask_iguales.sum():,})')
axes[2].scatter(X_pca2[mask_cambiados, 0], X_pca2[mask_cambiados, 1],
                c='tomato', alpha=0.6, s=10, label=f'Re-etiquetado ({mask_cambiados.sum():,})')
axes[2].set_title(f'Puntos Re-etiquetados ({pct_discrepancias:.1f}%)')
axes[2].set_xlabel('PC1'); axes[2].set_ylabel('PC2')
axes[2].legend(markerscale=2)

plt.tight_layout()
plt.savefig('16_reevaluacion_etiquetas.png', dpi=150, bbox_inches='tight')
plt.show()

# Heatmap de votos de clustering
fig, ax = plt.subplots(figsize=(10, 5))
votos_df = pd.DataFrame(votes, columns=['K-Means','FCM','Subtractive','DBSCAN','Jerárquico','GMM'])
votos_df['original'] = y_original
votos_df['consenso'] = y_nuevo

# Distribución de nivel de acuerdo
agreement = votes_sum.copy()
agreement_max = np.maximum(agreement, 6 - agreement)  # acuerdo normalizado 3-6

ax.hist(agreement_max, bins=4, range=(2.5, 6.5), color='steelblue',
        edgecolor='white', alpha=0.85)
ax.set_xlabel('Número de algoritmos en acuerdo (de 6)')
ax.set_ylabel('Frecuencia')
ax.set_title('Nivel de Acuerdo entre Algoritmos de Clustering')
ax.set_xticks([3, 4, 5, 6])
plt.tight_layout()
plt.savefig('17_acuerdo_clustering.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. Modelos Supervisados

# 6.1 Preparación de datos para modelos supervisados

# Features para modelos supervisados
# Usamos las mismas del clustering + algunas adicionales
FEATURES_SUP = FEATURES_CLUSTER.copy()

# Agregar features adicionales si existen
extra = ['tasa_abandono_libro', 'duracion_promedio_scaled', 'paginas_promedio_scaled',
         'es_fin_semana', 'periodo_dia_encoded', 'sentimiento_positivo_pct',
         'sentimiento_negativo_pct', 'tiene_reviews']
FEATURES_SUP += [f for f in extra if f in df.columns and f not in FEATURES_SUP]

print(f"Features supervisadas: {len(FEATURES_SUP)}")

X_sup = df[FEATURES_SUP].fillna(0).values

# Splits para etiquetas originales
X_tr_orig, X_te_orig, y_tr_orig, y_te_orig = train_test_split(
    X_sup, y_original, test_size=0.2, random_state=RANDOM_STATE, stratify=y_original)

# Splits para etiquetas re-evaluadas
X_tr_new, X_te_new, y_tr_new, y_te_new = train_test_split(
    X_sup, y_nuevo, test_size=0.2, random_state=RANDOM_STATE, stratify=y_nuevo)

# Regresión lineal: target continuo (dias_inactividad)
y_regresion = df['dias_inactividad'].fillna(0).values
X_tr_reg, X_te_reg, y_tr_reg, y_te_reg = train_test_split(
    X_sup, y_regresion, test_size=0.2, random_state=RANDOM_STATE)

print(f"\nTamaños de conjuntos:")
print(f"  Train: {len(X_tr_orig):,} | Test: {len(X_te_orig):,}")

# 6.2 Árbol de Decisión

def evaluar_clasificador(modelo, X_tr, y_tr, X_te, y_te, nombre):
    """Entrena, evalúa y devuelve métricas de un clasificador."""
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)
    y_proba = modelo.predict_proba(X_te)[:, 1] if hasattr(modelo, 'predict_proba') else None

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba) if y_proba is not None else None
    cv = cross_val_score(modelo, X_tr, y_tr, cv=5, scoring='accuracy').mean()

    print(f"\n{'='*50}")
    print(f"  {nombre}")
    print(f"{'='*50}")
    print(f"  Accuracy (test)  : {acc:.4f}")
    print(f"  ROC-AUC (test)   : {auc:.4f}" if auc else "  ROC-AUC: N/A")
    print(f"  Accuracy (CV-5)  : {cv:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['Completado','Abandonado'])}")

    return {'modelo': modelo, 'acc': acc, 'auc': auc, 'cv': cv,
            'y_pred': y_pred, 'y_proba': y_proba}

# Árbol con etiquetas originales
tree_orig = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=RANDOM_STATE,
                                   class_weight='balanced')
res_tree_orig = evaluar_clasificador(tree_orig, X_tr_orig, y_tr_orig,
                                     X_te_orig, y_te_orig, "Árbol de Decisión — Etiquetas ORIGINALES")

# Árbol con etiquetas re-evaluadas
tree_new = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=RANDOM_STATE,
                                  class_weight='balanced')
res_tree_new = evaluar_clasificador(tree_new, X_tr_new, y_tr_new,
                                    X_te_new, y_te_new, "Árbol de Decisión — Etiquetas RE-EVALUADAS")

# Visualización del árbol e importancia de features
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Árbol de Decisión — Análisis de Features', fontsize=14, fontweight='bold')

# Importancia de features — Originales
importances = res_tree_orig['modelo'].feature_importances_
indices = np.argsort(importances)[::-1][:15]
axes[0].barh([FEATURES_SUP[i] for i in indices[::-1]],
             importances[indices[::-1]], color='steelblue', alpha=0.85, edgecolor='white')
axes[0].set_title('Importancia de Features (Etiquetas Originales)')
axes[0].set_xlabel('Importancia')

# Importancia de features — Re-evaluadas
importances_new = res_tree_new['modelo'].feature_importances_
indices_new = np.argsort(importances_new)[::-1][:15]
axes[1].barh([FEATURES_SUP[i] for i in indices_new[::-1]],
             importances_new[indices_new[::-1]], color='tomato', alpha=0.85, edgecolor='white')
axes[1].set_title('Importancia de Features (Etiquetas Re-evaluadas)')
axes[1].set_xlabel('Importancia')

plt.tight_layout()
plt.savefig('18_arbol_importancia.png', dpi=150, bbox_inches='tight')
plt.show()

# Estructura del árbol
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(res_tree_orig['modelo'], feature_names=FEATURES_SUP,
          class_names=['Completado', 'Abandonado'],
          filled=True, rounded=True, max_depth=3, ax=ax, fontsize=9)
ax.set_title('Árbol de Decisión (primeros 3 niveles — Etiquetas Originales)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('19_arbol_estructura.png', dpi=150, bbox_inches='tight')
plt.show()

# 6.3 Regresión Logística

# Regresión Logística — Etiquetas originales
lr_orig = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                              class_weight='balanced', solver='lbfgs')
res_lr_orig = evaluar_clasificador(lr_orig, X_tr_orig, y_tr_orig,
                                   X_te_orig, y_te_orig, "Regresión Logística — Etiquetas ORIGINALES")

# Regresión Logística — Etiquetas re-evaluadas
lr_new = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                             class_weight='balanced', solver='lbfgs')
res_lr_new = evaluar_clasificador(lr_new, X_tr_new, y_tr_new,
                                  X_te_new, y_te_new, "Regresión Logística — Etiquetas RE-EVALUADAS")

# Curvas ROC comparativas
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Curvas ROC — Regresión Logística', fontsize=14, fontweight='bold')

for ax, (res, y_te, titulo) in zip(axes, [
    (res_lr_orig, y_te_orig, 'Etiquetas Originales'),
    (res_lr_new,  y_te_new,  'Etiquetas Re-evaluadas')
]):
    fpr, tpr, _ = roc_curve(y_te, res['y_proba'])
    ax.plot(fpr, tpr, color='steelblue', linewidth=2,
            label=f'ROC (AUC = {res["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Azar')
    ax.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(titulo)
    ax.legend()

plt.tight_layout()
plt.savefig('20_roc_logistica.png', dpi=150, bbox_inches='tight')
plt.show()

# Matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Matrices de Confusión — Regresión Logística', fontsize=14, fontweight='bold')
for ax, (res, y_te, titulo) in zip(axes, [
    (res_lr_orig, y_te_orig, 'Originales'),
    (res_lr_new,  y_te_new,  'Re-evaluadas')
]):
    ConfusionMatrixDisplay.from_predictions(
        y_te, res['y_pred'],
        display_labels=['Completado', 'Abandonado'],
        colorbar=False, ax=ax, cmap='Blues'
    )
    ax.set_title(titulo)
plt.tight_layout()
plt.savefig('21_confusion_logistica.png', dpi=150, bbox_inches='tight')
plt.show()

# 6.4 Regresión Lineal

def evaluar_regresion(modelo, X_tr, y_tr, X_te, y_te, nombre):
    """Entrena y evalúa un modelo de regresión."""
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)

    print(f"\n{'='*50}")
    print(f"  {nombre}")
    print(f"{'='*50}")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    return {'modelo': modelo, 'r2': r2, 'rmse': rmse, 'mae': mae,
            'y_pred': y_pred, 'y_te': y_te}

# Regresión Lineal — Predice días de inactividad (target continuo)
reg_orig = LinearRegression()
res_reg_orig = evaluar_regresion(reg_orig, X_tr_reg, y_tr_reg,
                                 X_te_reg, y_te_reg,
                                 "Regresión Lineal — dias_inactividad (Dataset Original)")

# Con dataset donde se removieron los puntos con etiquetas dudosas (alta incertidumbre)
# Consideramos solo puntos donde >=5 de 6 algoritmos están de acuerdo
mask_seguros = agreement_max >= 5
print(f"\nPuntos con alta certeza (>=5/6 acuerdo): {mask_seguros.sum():,} ({mask_seguros.mean()*100:.1f}%)")

X_seg = X_sup[mask_seguros]
y_reg_seg = y_regresion[mask_seguros]
X_tr_seg, X_te_seg, y_tr_seg, y_te_seg = train_test_split(
    X_seg, y_reg_seg, test_size=0.2, random_state=RANDOM_STATE)

reg_new = LinearRegression()
res_reg_new = evaluar_regresion(reg_new, X_tr_seg, y_tr_seg,
                                X_te_seg, y_te_seg,
                                "Regresión Lineal — dias_inactividad (Solo puntos seguros)")

# Visualización de regresión lineal
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Regresión Lineal — Predicción de Días de Inactividad',
             fontsize=14, fontweight='bold')

for ax, (res, titulo) in zip(axes, [
    (res_reg_orig, f'Dataset Original (R²={res_reg_orig["r2"]:.4f})'),
    (res_reg_new,  f'Solo Puntos Seguros (R²={res_reg_new["r2"]:.4f})')
]):
    y_real = res['y_te']
    y_pred = res['y_pred']
    lim = [min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max())]
    ax.scatter(y_real, y_pred, alpha=0.3, s=8, color='steelblue')
    ax.plot(lim, lim, 'r--', linewidth=2, label='Predicción perfecta')
    ax.set_xlabel('Valores Reales')
    ax.set_ylabel('Valores Predichos')
    ax.set_title(titulo)
    ax.legend()

plt.tight_layout()
plt.savefig('22_regresion_lineal.png', dpi=150, bbox_inches='tight')
plt.show()

# Residuos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Residuos de Regresión Lineal', fontsize=14, fontweight='bold')
for ax, (res, titulo) in zip(axes, [
    (res_reg_orig, 'Original'),
    (res_reg_new, 'Solo Seguros')
]):
    residuos = res['y_te'] - res['y_pred']
    ax.scatter(res['y_pred'], residuos, alpha=0.3, s=8, color='steelblue')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Valores Predichos')
    ax.set_ylabel('Residuos')
    ax.set_title(titulo)
plt.tight_layout()
plt.savefig('23_residuos_regresion.png', dpi=150, bbox_inches='tight')
plt.show()

# 7. Comparación Final: Originales vs Re-evaluadas

# Tabla comparativa completa
resultados_supervisados = pd.DataFrame({
    'Modelo': [
        'Árbol Decisión (original)',
        'Árbol Decisión (re-evaluado)',
        'Reg. Logística (original)',
        'Reg. Logística (re-evaluado)',
    ],
    'Accuracy': [
        res_tree_orig['acc'], res_tree_new['acc'],
        res_lr_orig['acc'],   res_lr_new['acc'],
    ],
    'ROC-AUC': [
        res_tree_orig['auc'], res_tree_new['auc'],
        res_lr_orig['auc'],   res_lr_new['auc'],
    ],
    'CV-5 Accuracy': [
        res_tree_orig['cv'], res_tree_new['cv'],
        res_lr_orig['cv'],   res_lr_new['cv'],
    ],
    'Etiquetas': ['Original', 'Re-evaluada', 'Original', 'Re-evaluada']
})

print(resultados_supervisados.round(4).to_string(index=False))

# Gráfica comparativa final
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparación Final: Modelos con Etiquetas Originales vs Re-evaluadas',
             fontsize=14, fontweight='bold')

modelos = ['Árbol\n(Original)', 'Árbol\n(Re-eval)', 'Log. Reg\n(Original)', 'Log. Reg\n(Re-eval)']
colores = ['steelblue', 'lightskyblue', 'tomato', 'lightsalmon']

metricas = [
    ('Accuracy', [res_tree_orig['acc'], res_tree_new['acc'],
                  res_lr_orig['acc'], res_lr_new['acc']]),
    ('ROC-AUC',  [res_tree_orig['auc'], res_tree_new['auc'],
                  res_lr_orig['auc'], res_lr_new['auc']]),
    ('CV-5 Acc', [res_tree_orig['cv'], res_tree_new['cv'],
                  res_lr_orig['cv'], res_lr_new['cv']]),
]

for ax, (titulo, valores) in zip(axes, metricas):
    bars = ax.bar(modelos, valores, color=colores, alpha=0.85, edgecolor='white')
    ax.set_title(titulo, fontsize=12)
    ax.set_ylabel(titulo)
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Azar')

plt.tight_layout()
plt.savefig('24_comparacion_final.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAnálisis completo. Gráficas guardadas en la carpeta informe-teorico-practico/")

# 8. Análisis Sin Data Leakage

# Los modelos anteriores logran accuracy >99.8% porque usan features que derivan directamente del target (ej. `completion_pct_end` codifica el mismo progreso que define el abandono). Aquí se entrena usando solo features disponibles **antes** de conocer el resultado: características NLP del libro, historial de usuario y contexto temporal.

FEATURES_LIMPIAS = [
    'duration_minutes_scaled',
    'pages_read_scaled',
    'velocidad_lectura_scaled',
    'num_sesiones_scaled',
    'es_fin_semana',
    'periodo_dia_encoded',
    'num_libros_leidos',
    'duracion_promedio_scaled',
    'paginas_promedio_scaled',
    'abandono_score_scaled',
    'engagement_score_scaled',
    'complejidad_score_scaled',
    'ritmo_score_scaled',
    'sentimiento_promedio_scaled',
    'sentimiento_positivo_pct',
    'sentimiento_negativo_pct',
    'tiene_reviews',
]
FEATURES_LIMPIAS = [f for f in FEATURES_LIMPIAS if f in df.columns]
print(f'Features sin leakage: {len(FEATURES_LIMPIAS)}')
for f in FEATURES_LIMPIAS:
    print(f'  - {f}')

X_limpio = df[FEATURES_LIMPIAS].fillna(0).values
X_tr_l,  X_te_l,  y_tr_l,  y_te_l  = train_test_split(
    X_limpio, y_original, test_size=0.2, random_state=RANDOM_STATE, stratify=y_original)
X_tr_ln, X_te_ln, y_tr_ln, y_te_ln = train_test_split(
    X_limpio, y_nuevo, test_size=0.2, random_state=RANDOM_STATE, stratify=y_nuevo)
print(f'Train: {len(X_tr_l):,} | Test: {len(X_te_l):,}')

tree_l_orig = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=RANDOM_STATE, class_weight='balanced')
res_tree_l_orig = evaluar_clasificador(tree_l_orig, X_tr_l, y_tr_l, X_te_l, y_te_l,
                                        'Arbol Decision SIN LEAKAGE - Originales')

tree_l_new = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=RANDOM_STATE, class_weight='balanced')
res_tree_l_new = evaluar_clasificador(tree_l_new, X_tr_ln, y_tr_ln, X_te_ln, y_te_ln,
                                      'Arbol Decision SIN LEAKAGE - Re-evaluadas')

lr_l_orig = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced', solver='lbfgs')
res_lr_l_orig = evaluar_clasificador(lr_l_orig, X_tr_l, y_tr_l, X_te_l, y_te_l,
                                      'Regresion Logistica SIN LEAKAGE - Originales')

lr_l_new = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced', solver='lbfgs')
res_lr_l_new = evaluar_clasificador(lr_l_new, X_tr_ln, y_tr_ln, X_te_ln, y_te_ln,
                                     'Regresion Logistica SIN LEAKAGE - Re-evaluadas')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Importancia de Features SIN Leakage', fontsize=14, fontweight='bold')
for ax, (res, titulo, color) in zip(axes, [
    (res_tree_l_orig, 'Etiquetas Originales', 'steelblue'),
    (res_tree_l_new,  'Etiquetas Re-evaluadas', 'tomato')
]):
    imps = res['modelo'].feature_importances_
    idx  = np.argsort(imps)[::-1]
    ax.barh([FEATURES_LIMPIAS[i] for i in idx[::-1]], imps[idx[::-1]], color=color, alpha=0.85, edgecolor='white')
    ax.set_title(titulo)
    ax.set_xlabel('Importancia')
plt.tight_layout()
plt.savefig('25_importancia_sin_leakage.png', dpi=150, bbox_inches='tight')
plt.show()

comparacion = pd.DataFrame({
    'Modelo':   ['Arbol (CON leakage)', 'Arbol (SIN leakage)', 'LogReg (CON leakage)', 'LogReg (SIN leakage)'],
    'Accuracy': [res_tree_orig['acc'], res_tree_l_orig['acc'], res_lr_orig['acc'], res_lr_l_orig['acc']],
    'ROC-AUC':  [res_tree_orig['auc'], res_tree_l_orig['auc'], res_lr_orig['auc'], res_lr_l_orig['auc']],
    'CV-5':     [res_tree_orig['cv'],  res_tree_l_orig['cv'],  res_lr_orig['cv'],  res_lr_l_orig['cv']],
})
print('COMPARACION CON vs SIN LEAKAGE:')
print(comparacion.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Impacto del Data Leakage en el Rendimiento', fontsize=14, fontweight='bold')
modelos_c = ['Arbol\n(con)', 'Arbol\n(sin)', 'LogReg\n(con)', 'LogReg\n(sin)']
colores_c = ['steelblue', 'lightskyblue', 'tomato', 'lightsalmon']
for ax, (metrica, valores) in zip(axes, [
    ('Accuracy', [res_tree_orig['acc'], res_tree_l_orig['acc'], res_lr_orig['acc'], res_lr_l_orig['acc']]),
    ('ROC-AUC',  [res_tree_orig['auc'], res_tree_l_orig['auc'], res_lr_orig['auc'], res_lr_l_orig['auc']]),
]):
    bars = ax.bar(modelos_c, valores, color=colores_c, alpha=0.85, edgecolor='white')
    ax.set_ylim(0, 1.08)
    ax.set_title(metrica)
    ax.set_ylabel(metrica)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='Azar')
    ax.legend()
plt.tight_layout()
plt.savefig('26_leakage_vs_sin_leakage.png', dpi=150, bbox_inches='tight')
plt.show()

print('=' * 65)
print('RESUMEN EJECUTIVO')
print('=' * 65)
print(f'Dataset usado: {len(df):,} sesiones (100% del total)')
print(f'Features clustering: {len(FEATURES_CLUSTER)} | Features supervisadas: {len(FEATURES_SUP)} | Features limpias: {len(FEATURES_LIMPIAS)}')
print()
print('CLUSTERING:')
mejor = max(resultados_clustering.items(), key=lambda x: x[1]['Silhouette'])
print(f'  Mejor metodo (Silhouette): {mejor[0]} = {mejor[1]["Silhouette"]:.4f}')
print(f'  Re-evaluacion de etiquetas: {n_discrepancias:,} puntos cambiados ({pct_discrepancias:.1f}%)')
print(f'  Nota: solo 1.6% cambiado (esperado ~30%) - etiquetas originales de alta calidad')
print()
print('MODELOS SUPERVISADOS (CON leakage - referencia):')
print(f'  Arbol (orig)   -> Acc={res_tree_orig["acc"]:.4f} | AUC={res_tree_orig["auc"]:.4f}')
print(f'  Arbol (re-ev)  -> Acc={res_tree_new["acc"]:.4f} | AUC={res_tree_new["auc"]:.4f}')
print(f'  LogReg (orig)  -> Acc={res_lr_orig["acc"]:.4f} | AUC={res_lr_orig["auc"]:.4f}')
print(f'  LogReg (re-ev) -> Acc={res_lr_new["acc"]:.4f} | AUC={res_lr_new["auc"]:.4f}')
print(f'  ADVERTENCIA: resultados inflados por data leakage (ver seccion sin leakage)')
print()
print('MODELOS SUPERVISADOS (SIN leakage - metodologicamente correcto):')
print(f'  Arbol (orig)   -> Acc={res_tree_l_orig["acc"]:.4f} | AUC={res_tree_l_orig["auc"]:.4f} | CV-5={res_tree_l_orig["cv"]:.4f}')
print(f'  Arbol (re-ev)  -> Acc={res_tree_l_new["acc"]:.4f} | AUC={res_tree_l_new["auc"]:.4f} | CV-5={res_tree_l_new["cv"]:.4f}')
print(f'  LogReg (orig)  -> Acc={res_lr_l_orig["acc"]:.4f} | AUC={res_lr_l_orig["auc"]:.4f} | CV-5={res_lr_l_orig["cv"]:.4f}')
print(f'  LogReg (re-ev) -> Acc={res_lr_l_new["acc"]:.4f} | AUC={res_lr_l_new["auc"]:.4f} | CV-5={res_lr_l_new["cv"]:.4f}')
print()
print('REGRESION LINEAL (dias_inactividad):')
print(f'  Dataset original  -> R2={res_reg_orig["r2"]:.4f} | RMSE={res_reg_orig["rmse"]:.4f}')
print(f'  Solo pts seguros  -> R2={res_reg_new["r2"]:.4f} | RMSE={res_reg_new["rmse"]:.4f}')
print(f'  Nota: R2 bajo (~0.06) esperado - dias_inactividad depende de factores externos')
print()
print('LIMITACIONES IDENTIFICADAS:')
print('  - DBSCAN: eps=0.5 no optimo, genera 234 clusters y 41.7% ruido (ajustar a eps~2-3)')
print('  - tasa_abandono tiene leakage temporal (incluye sesiones futuras del usuario)')
print('  - Re-evaluacion cambio solo 1.6% (datos simulados tienen estructura muy limpia)')
print('  - Reg. Lineal no es el modelo adecuado para dias_inactividad')
