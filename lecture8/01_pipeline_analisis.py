# # Pipeline de Análisis Exploratorio - Dataset FIRE UdeA
#
# **Objetivo:** Entender el dataset, identificar variables relevantes y preparar el terreno para modelado con árboles de decisión.
#
# ## Etapas del pipeline
# 1. Carga e inspección inicial
# 2. Calidad de datos (nulos, duplicados, tipos)
# 3. Análisis univariado
# 4. Análisis bivariado (relación con `label`)
# 5. Correlaciones y selección de variables
# 6. Conclusiones para el modelado

# ## 1. Carga e inspección inicial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11
sns.set_theme(style='whitegrid', palette='Set2')

# Ruta al dataset
DATA_PATH = '../dataset_sintetico_FIRE_UdeA_realista.csv'

df = pd.read_csv(DATA_PATH)
print(f'Shape: {df.shape}')
df.head()

print('=== TIPOS DE DATOS ===')
print(df.dtypes)
print(f'\nPeriodo: {df["anio"].min()} - {df["anio"].max()}')
print(f'Unidades: {df["unidad"].nunique()}')
print(df['unidad'].value_counts())

# ## 2. Calidad de datos

print('=== VALORES NULOS POR COLUMNA ===')
nulos = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(2)
calidad = pd.DataFrame({'nulos': nulos, 'pct_nulos': pct_nulos})
print(calidad[calidad['nulos'] > 0])

print(f'\nDuplicados: {df.duplicated().sum()}')

# Visualizar nulos
if nulos.sum() > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    calidad['pct_nulos'].plot(kind='bar', ax=ax, color='salmon')
    ax.set_title('Porcentaje de valores nulos por columna')
    ax.set_ylabel('% nulos')
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Separar columnas numéricas y categóricas
TARGET = 'label'
COLS_EXCLUIR = ['anio', 'unidad', TARGET]
cols_num = [c for c in df.select_dtypes(include=np.number).columns if c not in COLS_EXCLUIR]
cols_cat = [c for c in df.select_dtypes(include='object').columns if c not in COLS_EXCLUIR]

print(f'Variables numéricas ({len(cols_num)}): {cols_num}')
print(f'Variables categóricas ({len(cols_cat)}): {cols_cat}')
print(f'\nDistribución del target:')
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True).round(3))

# Imputar nulos con mediana (para análisis exploratorio)
df_clean = df.copy()
for col in cols_num:
    if df_clean[col].isnull().sum() > 0:
        mediana = df_clean[col].median()
        df_clean[col].fillna(mediana, inplace=True)
        print(f'  {col}: imputado con mediana={mediana:.4f}')

print(f'\nNulos restantes: {df_clean[cols_num].isnull().sum().sum()}')

# ## 3. Análisis Univariado

print('=== ESTADÍSTICAS DESCRIPTIVAS ===')
df_clean[cols_num].describe().T.style.format('{:.4f}').background_gradient(cmap='Blues', axis=1)

# Distribuciones de variables numéricas
n_cols = 3
n_rows = (len(cols_num) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(cols_num):
    ax = axes[i]
    df_clean[col].hist(ax=ax, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(df_clean[col].mean(), color='red', linestyle='--', linewidth=1.5, label='media')
    ax.axvline(df_clean[col].median(), color='orange', linestyle='--', linewidth=1.5, label='mediana')
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribuciones de variables numéricas', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# Detección de outliers con IQR
print('=== OUTLIERS (método IQR) ===')
outlier_reporte = []
for col in cols_num:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df_clean[col] < Q1 - 1.5 * IQR) | (df_clean[col] > Q3 + 1.5 * IQR)).sum()
    outlier_reporte.append({'variable': col, 'n_outliers': n_out, 'pct': round(n_out/len(df_clean)*100, 1)})

out_df = pd.DataFrame(outlier_reporte).sort_values('n_outliers', ascending=False)
print(out_df.to_string(index=False))

# ## 4. Análisis Bivariado — relación con `label`

# Distribución del target
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

vc = df_clean[TARGET].value_counts()
axes[0].bar(vc.index.astype(str), vc.values, color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Distribución del target (label)')
axes[0].set_xlabel('label')
axes[0].set_ylabel('Frecuencia')
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

axes[1].pie(vc.values, labels=[f'label={k}' for k in vc.index], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Proporción del target')

plt.tight_layout()
plt.show()

# Boxplots por clase del target
n_rows = (len(cols_num) + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(cols_num):
    ax = axes[i]
    grupos = [df_clean.loc[df_clean[TARGET] == k, col] for k in sorted(df_clean[TARGET].unique())]
    bp = ax.boxplot(grupos, labels=[f'label={k}' for k in sorted(df_clean[TARGET].unique())],
                    patch_artist=True)
    colores = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(col, fontsize=10)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribución por clase de target', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# Test de Mann-Whitney U para significancia estadística
print('=== SIGNIFICANCIA ESTADÍSTICA (Mann-Whitney U) ===')
print(f'{"Variable":<30} {"p-valor":<12} {"Significativa"}')
print('-' * 55)

resultados = []
for col in cols_num:
    g0 = df_clean.loc[df_clean[TARGET] == 0, col].dropna()
    g1 = df_clean.loc[df_clean[TARGET] == 1, col].dropna()
    if len(g0) > 0 and len(g1) > 0:
        stat, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
        sig = '*** SI' if p < 0.05 else 'no'
        resultados.append({'variable': col, 'p_valor': p, 'significativa': p < 0.05})
        print(f'{col:<30} {p:<12.4f} {sig}')

resultados_df = pd.DataFrame(resultados).sort_values('p_valor')
vars_significativas = resultados_df[resultados_df['significativa']]['variable'].tolist()
print(f'\nVariables con diferencia significativa entre clases: {len(vars_significativas)}')
print(vars_significativas)

# ## 5. Correlaciones y selección de variables

# Correlación con el target
corr_target = df_clean[cols_num + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
colores = ['#e74c3c' if v > 0 else '#3498db' for v in corr_target.values]
bars = ax.barh(corr_target.index, corr_target.values, color=colores, alpha=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Correlación de Pearson con el target (label)')
ax.set_xlabel('Correlación')
for bar, val in zip(bars, corr_target.values):
    ax.text(val + (0.005 if val >= 0 else -0.005), bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
plt.tight_layout()
plt.show()

print('\nTop correlaciones (absoluto):')
print(corr_target.head(8))

# Matriz de correlación entre features
corr_matrix = df_clean[cols_num].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, annot_kws={'size': 8})
ax.set_title('Matriz de correlación entre variables numéricas', fontsize=12)
plt.tight_layout()
plt.show()

# Detectar pares con alta correlación (multicolinealidad)
umbral_corr = 0.80
alta_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        c = corr_matrix.iloc[i, j]
        if abs(c) >= umbral_corr:
            alta_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], round(c, 3)))

print(f'Pares con correlación >= {umbral_corr}:')
if alta_corr:
    for a, b, c in alta_corr:
        print(f'  {a}  <-->  {b}  :  {c}')
else:
    print('  Ninguno — no hay multicolinealidad severa')

# Importancia de variables con Random Forest (sin tuning, solo exploratoria)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

X = df_clean[cols_num].copy()
y = df_clean[TARGET].copy()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importancias = pd.Series(rf.feature_importances_, index=cols_num).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
importancias.plot(kind='barh', ax=ax, color='steelblue', alpha=0.85)
ax.set_title('Importancia de variables (Random Forest — exploratoria)')
ax.set_xlabel('Importancia (mean decrease impurity)')
plt.tight_layout()
plt.show()

print('\nTop 8 variables más importantes:')
print(importancias.sort_values(ascending=False).head(8))

# ## 6. Análisis temporal y por unidad

# Distribución de labels por año
pivot_anio = df_clean.groupby(['anio', TARGET]).size().unstack(fill_value=0)
pivot_anio.plot(kind='bar', stacked=True, figsize=(10, 4),
                color=['#2ecc71', '#e74c3c'], alpha=0.85)
plt.title('Distribución de label por año')
plt.xlabel('Año')
plt.ylabel('Conteo')
plt.legend(title='label')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Distribución de labels por unidad
pivot_unidad = df_clean.groupby(['unidad', TARGET]).size().unstack(fill_value=0)
pivot_unidad.plot(kind='bar', stacked=True, figsize=(12, 4),
                  color=['#2ecc71', '#e74c3c'], alpha=0.85)
plt.title('Distribución de label por unidad académica')
plt.xlabel('')
plt.ylabel('Conteo')
plt.legend(title='label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ## 7. Conclusiones y variables seleccionadas para el modelado

# Selección final de variables basada en:
#  1) Correlación con target (|corr| > 0.15)
#  2) Significancia estadística (Mann-Whitney p < 0.05)
#  3) Importancia en Random Forest exploratoria

corr_abs = corr_target.abs()
vars_corr = corr_abs[corr_abs > 0.15].index.tolist()
vars_mw   = vars_significativas
vars_rf   = importancias.sort_values(ascending=False).head(6).index.tolist()

# Union de candidatas
candidatas = set(vars_corr) | set(vars_mw) | set(vars_rf)

# Score de consenso
score = {}
for v in cols_num:
    s = 0
    if v in vars_corr: s += 1
    if v in vars_mw:   s += 1
    if v in vars_rf:   s += 1
    score[v] = s

score_df = pd.DataFrame({
    'variable': list(score.keys()),
    'score_consenso': list(score.values()),
    'corr_target': [corr_target.get(v, 0) for v in score.keys()],
    'p_valor_mw': [resultados_df.set_index('variable')['p_valor'].get(v, 1.0) for v in score.keys()],
    'importancia_rf': [importancias.get(v, 0) for v in score.keys()]
}).sort_values('score_consenso', ascending=False)

print('=== RANKING DE VARIABLES PARA MODELADO ===')
print(score_df.to_string(index=False))

vars_finales = score_df[score_df['score_consenso'] >= 2]['variable'].tolist()
print(f'\n>>> Variables seleccionadas (score >= 2): {vars_finales}')

# Guardar dataset limpio con variables seleccionadas
cols_guardar = vars_finales + [TARGET]
df_clean[cols_guardar].to_csv('dataset_para_modelado.csv', index=False)
print(f'Dataset para modelado guardado: {len(df_clean)} filas x {len(cols_guardar)} columnas')
print(f'Columnas: {cols_guardar}')

# Resumen ejecutivo
print('=' * 60)
print('RESUMEN EJECUTIVO — Pipeline de Análisis')
print('=' * 60)
print(f'  Registros totales           : {len(df)}')
print(f'  Variables numéricas         : {len(cols_num)}')
print(f'  Columnas con nulos          : {(df[cols_num].isnull().sum() > 0).sum()}')
print(f'  Balance del target          : {dict(df[TARGET].value_counts())}')
print(f'  Variables significativas MW : {len(vars_significativas)}')
print(f'  Variables seleccionadas     : {len(vars_finales)}')
print(f'  -> {vars_finales}')
print('='*60)
print('Siguiente paso: 02_modelado_arbol_decision.ipynb')
