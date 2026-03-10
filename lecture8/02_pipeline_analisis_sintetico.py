# # Pipeline de Análisis Exploratorio — Dataset Sintético FIRE UdeA
#
# **Dataset:** `dataset_sintetico_FIRE_UdeA.csv`
# **Diferencias con el dataset realista:** 500 observaciones, sin dimensión temporal ni por unidad, 7 variables numéricas puras.
#
# ## Etapas
# 1. Carga e inspección inicial
# 2. Calidad de datos
# 3. Análisis univariado
# 4. Análisis bivariado (relación con `label`)
# 5. Correlaciones y multicolinealidad
# 6. Selección de variables para modelado

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

DATA_PATH = '../dataset_sintetico_FIRE_UdeA.csv'
TARGET = 'label'

df = pd.read_csv(DATA_PATH)
print(f'Shape: {df.shape}')
df.head(10)

print('=== TIPOS DE DATOS ===')
print(df.dtypes)
print(f'\nDistribución del target:')
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True).round(3))

# Variables predictoras
cols_num = [c for c in df.columns if c != TARGET]
print(f'Variables predictoras ({len(cols_num)}): {cols_num}')

# ## 2. Calidad de datos

print('=== VALORES NULOS ===')
nulos = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(2)
calidad = pd.DataFrame({'nulos': nulos, 'pct_nulos': pct_nulos})
print(calidad)

print(f'\nDuplicados exactos: {df.duplicated().sum()}')

# Graficar solo si hay nulos
if nulos.sum() > 0:
    calidad['pct_nulos'].plot(kind='bar', color='salmon', figsize=(8, 3))
    plt.title('% de valores nulos por columna')
    plt.ylabel('% nulos')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print('No hay valores nulos — el dataset está completo.')

# Revisar rangos y posibles valores anómalos
print('=== ESTADÍSTICAS DESCRIPTIVAS ===')
df[cols_num].describe().T.style.format('{:.4f}').background_gradient(cmap='Blues', axis=1)

# ## 3. Análisis Univariado

# Histogramas con curva KDE
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for i, col in enumerate(cols_num):
    ax = axes[i]
    df[col].plot(kind='hist', bins=20, density=True, ax=ax,
                 color='steelblue', alpha=0.7, edgecolor='white')
    df[col].plot(kind='kde', ax=ax, color='darkblue', linewidth=2)
    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5, label='media')
    ax.axvline(df[col].median(), color='orange', linestyle='--', linewidth=1.5, label='mediana')
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)

axes[-1].set_visible(False)
plt.suptitle('Distribuciones univariadas con KDE', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# Asimetría y curtosis
print('=== ASIMETRÍA Y CURTOSIS ===')
shape_df = pd.DataFrame({
    'skewness': df[cols_num].skew().round(3),
    'kurtosis': df[cols_num].kurt().round(3)
})
shape_df['dist_normal'] = shape_df.apply(
    lambda r: 'SI' if abs(r['skewness']) < 1 and abs(r['kurtosis']) < 3 else 'no', axis=1
)
print(shape_df.to_string())

# Test de normalidad Shapiro-Wilk
print('=== NORMALIDAD (Shapiro-Wilk, muestra de 200) ===')
print(f'{"Variable":<28} {"estadístico":<14} {"p-valor":<12} {"Normal?"}')
print('-' * 62)
muestra = df[cols_num].sample(min(200, len(df)), random_state=42)
for col in cols_num:
    stat, p = stats.shapiro(muestra[col].dropna())
    normal = 'SI' if p > 0.05 else 'no'
    print(f'{col:<28} {stat:<14.4f} {p:<12.4f} {normal}')

# Detección de outliers con IQR
print('=== OUTLIERS (método IQR) ===')
outlier_reporte = []
for col in cols_num:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    outlier_reporte.append({'variable': col, 'n_outliers': n_out, 'pct': round(n_out/len(df)*100, 1)})

out_df = pd.DataFrame(outlier_reporte).sort_values('n_outliers', ascending=False)
print(out_df.to_string(index=False))

# Boxplots de outliers
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(cols_num):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='navy'),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(col, fontsize=9)
axes[-1].set_visible(False)
plt.suptitle('Boxplots — detección de outliers', fontsize=12)
plt.tight_layout()
plt.show()

# ## 4. Análisis Bivariado — relación con `label`

# Distribución del target
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
vc = df[TARGET].value_counts()
axes[0].bar(vc.index.astype(str), vc.values, color=['#2ecc71', '#e74c3c'], alpha=0.85)
axes[0].set_title('Frecuencia del target (label)')
axes[0].set_xlabel('label')
axes[0].set_ylabel('Conteo')
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

axes[1].pie(vc.values, labels=[f'label={k}' for k in vc.index],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Proporción del target')
plt.tight_layout()
plt.show()

# KDE por clase del target — ver separación entre clases
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
colores_clase = {0: '#2ecc71', 1: '#e74c3c'}

for i, col in enumerate(cols_num):
    ax = axes[i]
    for clase, color in colores_clase.items():
        grupo = df.loc[df[TARGET] == clase, col].dropna()
        grupo.plot(kind='kde', ax=ax, color=color, linewidth=2, label=f'label={clase}')
        ax.axvline(grupo.mean(), color=color, linestyle=':', linewidth=1.2)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)

axes[-1].set_visible(False)
plt.suptitle('KDE por clase — separación entre label=0 y label=1', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# Boxplots por clase
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for i, col in enumerate(cols_num):
    ax = axes[i]
    grupos = [df.loc[df[TARGET] == k, col].dropna() for k in sorted(df[TARGET].unique())]
    bp = ax.boxplot(grupos, labels=[f'label={k}' for k in sorted(df[TARGET].unique())],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(col, fontsize=10)

axes[-1].set_visible(False)
plt.suptitle('Boxplots por clase del target', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# Test Mann-Whitney U para cada variable
print('=== SIGNIFICANCIA ESTADÍSTICA (Mann-Whitney U) ===')
print(f'{"Variable":<28} {"mediana_0":<14} {"mediana_1":<14} {"p-valor":<12} {"Significativa"}')
print('-' * 80)

resultados = []
for col in cols_num:
    g0 = df.loc[df[TARGET] == 0, col].dropna()
    g1 = df.loc[df[TARGET] == 1, col].dropna()
    stat, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
    sig = '*** SI' if p < 0.05 else 'no'
    resultados.append({'variable': col, 'mediana_0': g0.median(), 'mediana_1': g1.median(),
                       'p_valor': p, 'significativa': p < 0.05})
    print(f'{col:<28} {g0.median():<14.4f} {g1.median():<14.4f} {p:<12.4f} {sig}')

resultados_df = pd.DataFrame(resultados).sort_values('p_valor')
vars_significativas = resultados_df[resultados_df['significativa']]['variable'].tolist()
print(f'\nVariables significativas: {vars_significativas}')

# ## 5. Correlaciones y multicolinealidad

# Correlación de cada variable con el target
corr_target = df[cols_num + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
colores = ['#e74c3c' if v > 0 else '#3498db' for v in corr_target.values]
bars = ax.barh(corr_target.index, corr_target.values, color=colores, alpha=0.85)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Correlación de Pearson con label')
ax.set_xlabel('Correlación')
for bar, val in zip(bars, corr_target.values):
    offset = 0.005 if val >= 0 else -0.005
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
plt.tight_layout()
plt.show()

# Matriz de correlación completa
corr_matrix = df[cols_num].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, annot_kws={'size': 9})
ax.set_title('Matriz de correlación entre variables', fontsize=12)
plt.tight_layout()
plt.show()

# Pares con alta correlación (posible multicolinealidad)
umbral = 0.75
alta_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        c = corr_matrix.iloc[i, j]
        if abs(c) >= umbral:
            alta_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], round(c, 3)))

print(f'Pares con |correlación| >= {umbral}:')
if alta_corr:
    for a, b, c in alta_corr:
        print(f'  {a}  <-->  {b}  :  {c}')
else:
    print('  Ninguno — no hay multicolinealidad severa')

# Pairplot coloreado por clase (solo variables más relevantes)
top_vars = corr_target.abs().head(4).index.tolist()
cols_pairplot = top_vars + [TARGET]

g = sns.pairplot(df[cols_pairplot], hue=TARGET, palette={0: '#2ecc71', 1: '#e74c3c'},
                 plot_kws={'alpha': 0.5, 's': 20}, diag_kind='kde')
g.fig.suptitle('Pairplot — top 4 variables vs label', y=1.01, fontsize=12)
plt.show()

# Importancia con Random Forest exploratorio
from sklearn.ensemble import RandomForestClassifier

X = df[cols_num]
y = df[TARGET]

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

importancias = pd.Series(rf.feature_importances_, index=cols_num).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
importancias.plot(kind='barh', ax=ax, color='steelblue', alpha=0.85)
ax.set_title('Importancia de variables (Random Forest exploratorio)')
ax.set_xlabel('Importancia (mean decrease impurity)')
for i, (idx, val) in enumerate(importancias.items()):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.show()

# ## 6. Selección de variables para modelado

# Score de consenso: correlación + Mann-Whitney + RF
corr_abs = corr_target.abs()
vars_corr = corr_abs[corr_abs > 0.10].index.tolist()
vars_mw   = vars_significativas
vars_rf   = importancias.sort_values(ascending=False).head(5).index.tolist()

score_df = pd.DataFrame({
    'variable': cols_num,
    'corr_target': [corr_target.get(v, 0) for v in cols_num],
    'en_corr':     [v in vars_corr for v in cols_num],
    'en_mw':       [v in vars_mw for v in cols_num],
    'en_rf':       [v in vars_rf for v in cols_num],
    'p_valor_mw':  [resultados_df.set_index('variable')['p_valor'].get(v, 1.0) for v in cols_num],
    'importancia_rf': [importancias.get(v, 0) for v in cols_num]
})
score_df['score_consenso'] = score_df[['en_corr', 'en_mw', 'en_rf']].sum(axis=1)
score_df = score_df.sort_values('score_consenso', ascending=False)

print('=== RANKING DE VARIABLES PARA MODELADO ===')
cols_show = ['variable', 'score_consenso', 'corr_target', 'p_valor_mw', 'importancia_rf']
print(score_df[cols_show].to_string(index=False))

vars_finales = score_df[score_df['score_consenso'] >= 2]['variable'].tolist()
print(f'\n>>> Variables seleccionadas (score >= 2): {vars_finales}')

# Visualización del score de consenso
fig, ax = plt.subplots(figsize=(8, 4))
colores_score = ['#e74c3c' if v == 3 else '#f39c12' if v == 2 else '#95a5a6'
                 for v in score_df['score_consenso']]
ax.barh(score_df['variable'], score_df['score_consenso'], color=colores_score, alpha=0.85)
ax.axvline(2, color='black', linestyle='--', linewidth=1.2, label='umbral selección (2)')
ax.set_title('Score de consenso por variable')
ax.set_xlabel('Score (0-3)')
ax.set_xlim(0, 3.5)
ax.legend()
from matplotlib.patches import Patch
leyenda = [Patch(color='#e74c3c', label='score=3 (alta prioridad)'),
           Patch(color='#f39c12', label='score=2 (incluir)'),
           Patch(color='#95a5a6', label='score<=1 (descartar)')]
ax.legend(handles=leyenda, loc='lower right', fontsize=9)
plt.tight_layout()
plt.show()

# Guardar dataset listo para modelado
cols_guardar = vars_finales + [TARGET]
df[cols_guardar].to_csv('dataset_sintetico_para_modelado.csv', index=False)
print(f'Guardado: {len(df)} filas x {len(cols_guardar)} columnas')
print(f'Columnas: {cols_guardar}')

# Resumen ejecutivo
print('=' * 60)
print('RESUMEN EJECUTIVO — Dataset Sintético FIRE UdeA')
print('=' * 60)
print(f'  Registros totales           : {len(df)}')
print(f'  Variables predictoras       : {len(cols_num)}')
print(f'  Nulos totales               : {df[cols_num].isnull().sum().sum()}')
print(f'  Balance del target          : {dict(df[TARGET].value_counts())}')
print(f'  Clases balanceadas          : {"SI" if df[TARGET].value_counts(normalize=True).min() > 0.4 else "no — revisar"}')
print(f'  Variables significativas MW : {len(vars_significativas)}')
print(f'  Variables seleccionadas     : {len(vars_finales)}')
print(f'  -> {vars_finales}')
print(f'\n  Multicolinealidad severa    : {"SI" if alta_corr else "no detectada"}')
print('=' * 60)
print('Siguiente paso: modelado con árbol de decisión')
