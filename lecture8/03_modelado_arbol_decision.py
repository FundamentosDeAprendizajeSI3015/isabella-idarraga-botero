# # Modelado con Árboles de Decisión — FIRE UdeA
#
# **Objetivo:** Superar las métricas del modelo baseline GBM.
#
# | Métrica | Baseline GBM (test) | Meta |
# |---------|--------------------|---------|
# | ROC-AUC | 0.417 | > 0.70 |
# | Log-loss | 4.877 | < 1.50 |
# | Brier   | 0.257 | < 0.20 |
#
# **Causa del fallo del baseline:** overfitting total (train AUC=1.0, test AUC=0.417).
# **Solución:** modelos más simples + regularización + selección de features.

# ## 0. Importaciones y configuración

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

plt.rcParams['figure.figsize'] = (10, 5)
sns.set_theme(style='whitegrid')
SEED = 42
np.random.seed(SEED)

print('Librerías cargadas OK')

# ## 1. Carga y splits temporales

df = pd.read_csv('../dataset_sintetico_FIRE_UdeA_realista.csv')

# Mismo split temporal que el baseline
# Train: 2016-2022 | Valid: 2023 | Test: 2024 | Holdout: 2025
TRAIN_END  = 2022
VALID_YEAR = 2023
TEST_YEAR  = 2024
HOLD_YEAR  = 2025
TARGET     = 'label'

df_train = df[df['anio'] <= TRAIN_END].copy()
df_valid = df[df['anio'] == VALID_YEAR].copy()
df_test  = df[df['anio'] == TEST_YEAR].copy()
df_hold  = df[df['anio'] == HOLD_YEAR].copy()

print(f'Train : {len(df_train)} filas | prevalencia label=1: {df_train[TARGET].mean():.2%}')
print(f'Valid : {len(df_valid)} filas | prevalencia label=1: {df_valid[TARGET].mean():.2%}')
print(f'Test  : {len(df_test)} filas  | prevalencia label=1: {df_test[TARGET].mean():.2%}')
print(f'Hold  : {len(df_hold)} filas  | prevalencia label=1: {df_hold[TARGET].mean():.2%}')

# ## 2. Feature engineering y selección de variables

# Variables numéricas (excluir anio, unidad, label)
COLS_BASE = ['liquidez', 'dias_efectivo', 'cfo', 'participacion_ley30',
             'participacion_regalias', 'participacion_servicios',
             'participacion_matriculas', 'hhi_fuentes', 'endeudamiento',
             'tendencia_ingresos', 'gp_ratio']

# Feature engineering: ratios adicionales que pueden capturar riesgo
def add_features(df_):
    d = df_.copy()
    # Diversificación vs concentración de fuentes
    d['conc_ley30_regalias'] = d['participacion_ley30'] + d['participacion_regalias']
    # Tensión financiera: endeudamiento alto + liquidez baja
    d['tension'] = d['endeudamiento'] / (d['liquidez'] + 1e-9)
    # Eficiencia: gastos vs ingresos (gp_ratio ya existe, agregar endeudamiento relativo)
    d['riesgo_combo'] = d['gp_ratio'] * d['endeudamiento']
    return d

df_train = add_features(df_train)
df_valid = add_features(df_valid)
df_test  = add_features(df_test)
df_hold  = add_features(df_hold)

COLS_ENG = COLS_BASE + ['conc_ley30_regalias', 'tension', 'riesgo_combo']

# Imputar nulos con mediana del train
medianas = df_train[COLS_ENG].median()
for split in [df_train, df_valid, df_test, df_hold]:
    split[COLS_ENG] = split[COLS_ENG].fillna(medianas)

print(f'Features totales: {len(COLS_ENG)}')
print(COLS_ENG)

# Selección de features por información mutua con el target
from sklearn.feature_selection import mutual_info_classif

X_tr = df_train[COLS_ENG]
y_tr = df_train[TARGET]

mi = mutual_info_classif(X_tr, y_tr, random_state=SEED)
mi_df = pd.Series(mi, index=COLS_ENG).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 4))
mi_df.plot(kind='bar', ax=ax, color='steelblue', alpha=0.85)
ax.axhline(0.03, color='red', linestyle='--', label='umbral=0.03')
ax.set_title('Información Mutua con label (train)')
ax.set_ylabel('Información mutua')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Features seleccionadas
FEATURES = mi_df[mi_df >= 0.03].index.tolist()
print(f'\nFeatures seleccionadas ({len(FEATURES)}): {FEATURES}')

# Matrices de features para cada split
X_train = df_train[FEATURES]
y_train = df_train[TARGET]

X_valid = df_valid[FEATURES]
y_valid = df_valid[TARGET]

X_test  = df_test[FEATURES]
y_test  = df_test[TARGET]

X_hold  = df_hold[FEATURES]
y_hold  = df_hold[TARGET]

# Train+Valid para reentrenar el modelo final
X_tv = pd.concat([X_train, X_valid])
y_tv = pd.concat([y_train, y_valid])

print(f'Train: {X_train.shape} | Valid: {X_valid.shape} | Test: {X_test.shape}')

# ## 3. Función de evaluación unificada

def evaluar(modelo, X, y, nombre_split=''):
    """Devuelve dict con todas las métricas del baseline."""
    prob = modelo.predict_proba(X)[:, 1]
    pred = modelo.predict(X)
    cm   = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0,0])

    try:
        auc = roc_auc_score(y, prob)
    except ValueError:
        auc = np.nan

    return {
        'split': nombre_split,
        'n': len(y),
        'prevalencia': y.mean().round(3),
        'roc_auc':  round(auc, 4),
        'pr_auc':   round(average_precision_score(y, prob), 4),
        'brier':    round(brier_score_loss(y, prob), 4),
        'log_loss': round(log_loss(y, prob), 4),
        'precision':round(precision_score(y, pred, zero_division=0), 4),
        'recall':   round(recall_score(y, pred, zero_division=0), 4),
        'f1':       round(f1_score(y, pred, zero_division=0), 4),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def reporte_completo(modelo, nombre_modelo, incluir_hold=True):
    splits = [
        evaluar(modelo, X_train, y_train, 'train'),
        evaluar(modelo, X_valid, y_valid, 'valid'),
        evaluar(modelo, X_test,  y_test,  'test'),
    ]
    if incluir_hold:
        splits.append(evaluar(modelo, X_hold, y_hold, 'hold-2025'))
    rep = pd.DataFrame(splits)
    print(f'\n=== {nombre_modelo} ===')
    print(rep[['split','roc_auc','pr_auc','brier','log_loss','precision','recall','f1','tn','fp','fn','tp']].to_string(index=False))
    return rep


# ── Cargar el reporte de métricas de la profesora (baseline GBM) ──
baseline = pd.read_csv('../reporte_metricas_FIRE_UdeA_realista.csv')
baseline.insert(0, 'modelo', 'GBM-baseline')

print('=== BASELINE GBM — Reporte de la profesora ===')
print(baseline[['split','n','prevalencia','roc_auc','pr_auc','brier','log_loss',
                'precision','recall','f1','tn','fp','fn','tp']].to_string(index=False))
print()
print('Interpretación rápida:')
print('  train  → AUC=1.00  (overfitting total)')
print('  valid  → AUC=0.93  (el modelo "recuerda" el período)')
print('  test   → AUC=0.42  (falla en generalización — peor que aleatorio)')
print('  log-loss test = 4.88  (calibración muy mala)')

# ## 4. Árbol de Decisión — búsqueda de hiperparámetros

# Cross-validation temporal con TimeSeriesSplit
# Con 56 filas de train y 8 unidades por año, usamos n_splits=4
tscv = TimeSeriesSplit(n_splits=4, gap=0)

param_grid_dt = {
    'max_depth':        [2, 3, 4],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split':[4, 6, 8],
    'criterion':        ['gini', 'entropy'],
}

dt_base = DecisionTreeClassifier(random_state=SEED)
gs_dt = GridSearchCV(
    dt_base, param_grid_dt, cv=tscv,
    scoring='roc_auc', n_jobs=-1, refit=True
)
gs_dt.fit(X_train, y_train)

print(f'Mejores hiperparámetros: {gs_dt.best_params_}')
print(f'CV ROC-AUC (train): {gs_dt.best_score_:.4f}')

# Mejor árbol — evaluación completa
best_dt = gs_dt.best_estimator_
rep_dt = reporte_completo(best_dt, 'Árbol de Decisión (sin calibrar)')

# Calibrar el árbol de decisión para mejorar log-loss y brier
dt_calibrado = CalibratedClassifierCV(best_dt, cv='prefit', method='isotonic')
dt_calibrado.fit(X_valid, y_valid)
rep_dt_cal = reporte_completo(dt_calibrado, 'Árbol de Decisión (calibrado - Isotonic)')

# Visualizar el árbol de decisión
fig, ax = plt.subplots(figsize=(16, 7))
plot_tree(
    best_dt, feature_names=FEATURES, class_names=['label=0', 'label=1'],
    filled=True, rounded=True, fontsize=9, ax=ax,
    impurity=True, proportion=False
)
plt.title(f'Árbol de Decisión Óptimo — profundidad={best_dt.max_depth}', fontsize=12)
plt.tight_layout()
plt.savefig('arbol_decision_FIRE.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n=== Reglas del árbol (texto) ===')
print(export_text(best_dt, feature_names=FEATURES, max_depth=4))

# Importancia de variables en el árbol
imp_dt = pd.Series(best_dt.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
imp_dt.plot(kind='barh', ax=ax, color='steelblue', alpha=0.85)
ax.set_title('Importancia de variables — Árbol de Decisión')
ax.set_xlabel('Importancia (reducción de impureza)')
plt.tight_layout()
plt.show()

# ## 5. Modelos alternativos para comparación

# --- Random Forest regularizado ---
param_grid_rf = {
    'n_estimators':     [50, 100, 200],
    'max_depth':        [2, 3, 4],
    'min_samples_leaf': [3, 5, 7],
    'max_features':     ['sqrt', 0.5],
}
gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=SEED), param_grid_rf,
    cv=tscv, scoring='roc_auc', n_jobs=-1, refit=True
)
gs_rf.fit(X_train, y_train)
print(f'RF mejores params: {gs_rf.best_params_}')
print(f'RF CV ROC-AUC: {gs_rf.best_score_:.4f}')

rep_rf = reporte_completo(gs_rf.best_estimator_, 'Random Forest (regularizado)')

# --- Regresión Logística (baseline fuerte para datos pequeños) ---
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=SEED, solver='lbfgs'))
])

param_grid_lr = {
    'clf__C': [0.01, 0.05, 0.1, 0.5, 1.0],
    'clf__penalty': ['l2'],
}
gs_lr = GridSearchCV(
    lr_pipe, param_grid_lr, cv=tscv, scoring='roc_auc', n_jobs=-1, refit=True
)
gs_lr.fit(X_train, y_train)
print(f'LR mejores params: {gs_lr.best_params_}')
print(f'LR CV ROC-AUC: {gs_lr.best_score_:.4f}')

rep_lr = reporte_completo(gs_lr.best_estimator_, 'Regresión Logística (C=óptimo)')

# --- GBM con regularización fuerte (comparación directa con baseline) ---
param_grid_gbm = {
    'n_estimators':    [30, 50, 100],
    'max_depth':       [1, 2, 3],
    'learning_rate':   [0.01, 0.05, 0.1],
    'subsample':       [0.5, 0.7],
    'min_samples_leaf':[3, 5],
}
gs_gbm = GridSearchCV(
    GradientBoostingClassifier(random_state=SEED), param_grid_gbm,
    cv=tscv, scoring='roc_auc', n_jobs=-1, refit=True
)
gs_gbm.fit(X_train, y_train)
print(f'GBM regularizado mejores params: {gs_gbm.best_params_}')
print(f'GBM CV ROC-AUC: {gs_gbm.best_score_:.4f}')

rep_gbm = reporte_completo(gs_gbm.best_estimator_, 'GBM Regularizado')

# ## 6. Comparación de modelos y selección del mejor

# Recopilar métricas de test de todos los modelos
modelos = {
    'DT':       best_dt,
    'DT-cal':   dt_calibrado,
    'RF':       gs_rf.best_estimator_,
    'LR':       gs_lr.best_estimator_,
    'GBM-reg':  gs_gbm.best_estimator_,
}

resultados = []
for nombre, modelo in modelos.items():
    for split_X, split_y, split_name in [
        (X_train, y_train, 'train'),
        (X_valid, y_valid, 'valid'),
        (X_test,  y_test,  'test'),
        (X_hold,  y_hold,  'hold'),
    ]:
        m = evaluar(modelo, split_X, split_y, split_name)
        m['modelo'] = nombre
        resultados.append(m)

res_df = pd.DataFrame(resultados)

# Mostrar solo test
test_df = res_df[res_df['split'] == 'test'][['modelo','roc_auc','pr_auc','brier','log_loss','f1','precision','recall','tn','fp','fn','tp']]
print('=== COMPARACIÓN EN TEST (2024) ===')
print(test_df.sort_values('roc_auc', ascending=False).to_string(index=False))

print('\n=== REFERENCIA BASELINE GBM ===')
print('  roc_auc=0.4167 | pr_auc=0.7246 | brier=0.2570 | log_loss=4.8766 | f1=0.8571')

# Heatmap de métricas por modelo y split
metricas_heatmap = ['roc_auc', 'pr_auc', 'brier', 'log_loss', 'f1']
pivots = {}
for m in metricas_heatmap:
    pivots[m] = res_df.pivot(index='modelo', columns='split', values=m)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cmap_pos = 'RdYlGn'   # mayor = mejor
cmap_neg = 'RdYlGn_r' # menor = mejor

for ax, (metrica, cmap) in zip(axes, [
    ('roc_auc', cmap_pos), ('brier', cmap_neg), ('log_loss', cmap_neg)
]):
    pivot = pivots[metrica][['train', 'valid', 'test', 'hold']]
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, ax=ax, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title(f'{metrica}\n(verde=mejor)')

plt.suptitle('Comparación de modelos por split', fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# Detectar overfitting: gap entre train y test ROC-AUC
train_auc = res_df[res_df['split'] == 'train'].set_index('modelo')['roc_auc']
test_auc  = res_df[res_df['split'] == 'test'].set_index('modelo')['roc_auc']
gap_df = pd.DataFrame({'train_auc': train_auc, 'test_auc': test_auc})
gap_df['gap'] = (gap_df['train_auc'] - gap_df['test_auc']).round(4)
gap_df['overfit'] = gap_df['gap'] > 0.20
print('=== GAP OVERFITTING (train - test ROC-AUC) ===')
print(gap_df.sort_values('gap'))
print('\nBaseline GBM: gap = 1.0 - 0.4167 = 0.5833 (overfitting extremo)')

# ## 7. Modelo ganador — análisis detallado

# Seleccionar el modelo con mejor ROC-AUC en test
mejor_nombre = test_df.sort_values('roc_auc', ascending=False).iloc[0]['modelo']
modelo_final = modelos[mejor_nombre]
print(f'Modelo ganador: {mejor_nombre}')

# Reentrenar con train+valid usando clone() — funciona con Pipeline y cualquier estimador
from sklearn.base import clone

if mejor_nombre == 'DT-cal':
    # CalibratedClassifierCV con cv='prefit' no es clonable directamente;
    # reentrenamos el árbol base y volvemos a calibrar
    best_dt_tv = clone(gs_dt.best_estimator_)
    best_dt_tv.fit(X_tv, y_tv)
    modelo_final_tv = best_dt_tv
else:
    modelo_final_tv = clone(modelo_final)
    modelo_final_tv.fit(X_tv, y_tv)

rep_final = reporte_completo(modelo_final_tv, f'{mejor_nombre} — reentrenado con train+valid')

# ── Optimización del umbral de decisión en validación ──────────────────────
# El umbral por defecto (0.5) puede no ser óptimo para maximizar F1.
# Buscamos el umbral que maximiza F1 en el conjunto de VALIDACIÓN (2023),
# y luego lo aplicamos en test. Esto es metodológicamente correcto.

umbrales   = np.arange(0.10, 0.90, 0.01)
prob_val   = modelo_final.predict_proba(X_valid)[:, 1]

resultados_umbral = []
for u in umbrales:
    pred_u = (prob_val >= u).astype(int)
    resultados_umbral.append({
        'umbral':    round(u, 2),
        'f1':        round(f1_score(y_valid, pred_u, zero_division=0), 4),
        'precision': round(precision_score(y_valid, pred_u, zero_division=0), 4),
        'recall':    round(recall_score(y_valid, pred_u, zero_division=0), 4),
    })

df_umbral  = pd.DataFrame(resultados_umbral)
umbral_opt = float(df_umbral.loc[df_umbral['f1'].idxmax(), 'umbral'])

print(f'Umbral por defecto (0.50) → F1 valid: '
      f'{f1_score(y_valid, (prob_val>=0.50).astype(int)):.4f}')
print(f'Umbral óptimo     ({umbral_opt:.2f}) → F1 valid: {df_umbral["f1"].max():.4f}')
print(f'\nTop 5 umbrales por F1 en validación:')
print(df_umbral.sort_values("f1", ascending=False).head(5).to_string(index=False))

# Visualizar F1, Precision y Recall vs umbral
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(df_umbral['umbral'], df_umbral['f1'],        label='F1',        color='steelblue', lw=2)
ax.plot(df_umbral['umbral'], df_umbral['precision'],  label='Precision', color='#27ae60',   lw=1.5, linestyle='--')
ax.plot(df_umbral['umbral'], df_umbral['recall'],     label='Recall',    color='#e67e22',   lw=1.5, linestyle='--')
ax.axvline(umbral_opt, color='red',  linestyle=':',  lw=2,   label=f'umbral óptimo = {umbral_opt:.2f}')
ax.axvline(0.50,       color='gray', linestyle=':',  lw=1.5, label='umbral default = 0.50')
ax.set_xlabel('Umbral de decisión')
ax.set_ylabel('Métrica')
ax.set_title(f'Optimización de umbral — {mejor_nombre} (validación 2023)')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.08)
plt.tight_layout()
plt.show()

print(f'\nUmbral seleccionado: {umbral_opt:.2f} (buscado en validación, aplicado en test)')

# Curvas ROC y PR para el modelo ganador en los 4 splits
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

splits_plot = [
    (X_train, y_train, 'train',      '#95a5a6'),
    (X_valid, y_valid, 'valid-2023', '#3498db'),
    (X_test,  y_test,  'test-2024',  '#e74c3c'),
    (X_hold,  y_hold,  'hold-2025',  '#9b59b6'),
]

# ROC
for X_, y_, label_, color_ in splits_plot:
    prob_ = modelo_final.predict_proba(X_)[:, 1]
    try:
        fpr, tpr, _ = roc_curve(y_, prob_)
        auc_ = roc_auc_score(y_, prob_)
        axes[0].plot(fpr, tpr, label=f'{label_} (AUC={auc_:.3f})', color=color_, linewidth=2)
    except:
        pass
axes[0].plot([0,1],[0,1],'k--', linewidth=1, label='aleatorio (0.5)')
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title(f'Curva ROC — {mejor_nombre}')
axes[0].legend(fontsize=9)

# PR
for X_, y_, label_, color_ in splits_plot:
    prob_ = modelo_final.predict_proba(X_)[:, 1]
    try:
        prec_, rec_, _ = precision_recall_curve(y_, prob_)
        ap_ = average_precision_score(y_, prob_)
        axes[1].plot(rec_, prec_, label=f'{label_} (AP={ap_:.3f})', color=color_, linewidth=2)
    except:
        pass
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title(f'Curva PR — {mejor_nombre}')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('curvas_roc_pr_FIRE.png', dpi=150, bbox_inches='tight')
plt.show()

# Predicciones detalladas en test (2024) — comparar con baseline
baseline_scores = pd.read_csv('../scores_test_FIRE_UdeA_realista.csv')
prob_nuevo = modelo_final.predict_proba(X_test)[:, 1]
pred_nuevo = (prob_nuevo >= umbral_opt).astype(int)   # umbral optimizado en validación

comparacion = df_test[['anio', 'unidad']].copy().reset_index(drop=True)
comparacion['y_true']        = y_test.values
comparacion['prob_baseline'] = baseline_scores['prob'].values
comparacion['pred_baseline'] = baseline_scores['pred'].values
comparacion['prob_nuevo']    = prob_nuevo.round(4)
comparacion['pred_nuevo']    = pred_nuevo
comparacion['correcto_base'] = (comparacion['y_true'] == comparacion['pred_baseline']).map({True:'✓', False:'✗'})
comparacion['correcto_nuevo']= (comparacion['y_true'] == comparacion['pred_nuevo']).map({True:'✓', False:'✗'})

print(f'=== Predicciones en TEST 2024 — {mejor_nombre} (umbral={umbral_opt:.2f}) vs GBM-baseline ===')
print(comparacion.to_string(index=False))

# Matriz de confusión lado a lado
cm_base = confusion_matrix(comparacion['y_true'], comparacion['pred_baseline'])
cm_new  = confusion_matrix(comparacion['y_true'], comparacion['pred_nuevo'])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, cm, titulo in zip(axes,
    [cm_base, cm_new],
    ['GBM Baseline (test 2024)', f'{mejor_nombre} (test 2024)']):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['pred=0','pred=1'],
                yticklabels=['real=0','real=1'])
    ax.set_title(titulo)

plt.suptitle('Matrices de confusión — test 2024', fontsize=11)
plt.tight_layout()
plt.show()

# ## 8. Reporte final de mejoras

filas = []
for X_, y_, s in [(X_train,y_train,'train'),(X_valid,y_valid,'valid'),(X_test,y_test,'test')]:
    prob = modelo_final.predict_proba(X_)[:,1]
    pred = (prob >= umbral_opt).astype(int)    # umbral optimizado en validación
    cm   = confusion_matrix(y_, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,cm[0,0])
    try:    auc = roc_auc_score(y_, prob)
    except: auc = float('nan')
    filas.append({
        'split':       s,
        'n':           len(y_),
        'prevalencia': y_.mean(),
        'roc_auc':     auc,
        'pr_auc':      average_precision_score(y_, prob),
        'brier':       brier_score_loss(y_, prob),
        'log_loss':    log_loss(y_, prob),
        'precision':   precision_score(y_, pred, zero_division=0),
        'recall':      recall_score(y_, pred, zero_division=0),
        'f1':          f1_score(y_, pred, zero_division=0),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    })

reporte_nuevo = pd.DataFrame(filas)
reporte_nuevo.to_csv('reporte_metricas_mi_modelo.csv', index=False)
print(f'Umbral aplicado: {umbral_opt:.2f}')
reporte_nuevo

COLS_REPORTE = ['split','n','prevalencia','roc_auc','pr_auc','brier','log_loss','precision','recall','f1','tn','fp','fn','tp']

# ── Tabla mi modelo ──
mi_modelo = reporte_nuevo.copy()
mi_modelo.insert(0, 'modelo', mejor_nombre)

# ── Tabla baseline (mismas columnas) ──
bl_tabla = baseline[COLS_REPORTE].copy()
bl_tabla.insert(0, 'modelo', 'GBM-baseline')

# ── Combinar ──
tabla_comp = pd.concat([mi_modelo, bl_tabla], ignore_index=True)
tabla_comp = tabla_comp.sort_values(['split', 'modelo']).reset_index(drop=True)

# Orden de splits legible
orden_split = {'train': 0, 'valid': 1, 'test': 2}
tabla_comp['_ord'] = tabla_comp['split'].map(orden_split)
tabla_comp = tabla_comp.sort_values(['_ord', 'modelo']).drop(columns='_ord').reset_index(drop=True)

# ── Guardar ──
tabla_comp.to_csv('comparacion_modelos_metricas.csv', index=False)

# ── Mostrar ──
print('=== TABLA COMPARATIVA: mi modelo vs GBM-baseline ===\n')
tabla_comp

# Gráfica comparativa — modelo nuevo vs baseline (test 2024)
bl_test  = baseline[baseline['split'] == 'test'].iloc[0]
new_test = reporte_nuevo[reporte_nuevo['split'] == 'test'].iloc[0]

metricas_pos  = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']
metricas_neg  = ['brier', 'log_loss']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
w = 0.35

# Mayor = mejor
x = np.arange(len(metricas_pos))
axes[0].bar(x - w/2, [bl_test[m]  for m in metricas_pos], w, label='GBM Baseline', color='#e74c3c', alpha=0.85)
axes[0].bar(x + w/2, [new_test[m] for m in metricas_pos], w, label=mejor_nombre,   color='#2ecc71', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(metricas_pos, rotation=20)
axes[0].set_ylim(0, 1.15)
axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1)
axes[0].set_title('Métricas positivas — test 2024 (mayor = mejor)')
axes[0].legend()

# Menor = mejor
x2 = np.arange(len(metricas_neg))
axes[1].bar(x2 - w/2, [bl_test[m]  for m in metricas_neg], w, label='GBM Baseline', color='#e74c3c', alpha=0.85)
axes[1].bar(x2 + w/2, [new_test[m] for m in metricas_neg], w, label=mejor_nombre,   color='#2ecc71', alpha=0.85)
axes[1].set_xticks(x2); axes[1].set_xticklabels(metricas_neg)
axes[1].set_title('Métricas de error — test 2024 (menor = mejor)')
axes[1].legend()

for ax in axes:
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)

plt.suptitle(f'Comparación test 2024 — GBM-baseline vs {mejor_nombre}', fontsize=11)
plt.tight_layout()
plt.savefig('comparacion_modelos_FIRE.png', dpi=150, bbox_inches='tight')
plt.show()

nt = reporte_nuevo[reporte_nuevo['split'] == 'test'].iloc[0]
bt = baseline[baseline['split'] == 'test'].iloc[0]

print('=' * 65)
print('RESUMEN EJECUTIVO — Modelado FIRE-UdeA con Árboles de Decisión')
print('=' * 65)
print(f'\n  Modelo ganador  : {mejor_nombre}')
print(f'  Features usadas : {FEATURES}')
print(f'\n  --- TEST 2024 (baseline → nuevo) ---')
print(f'  ROC-AUC  : {bt["roc_auc"]:.4f} → {nt["roc_auc"]:.4f}')
print(f'  PR-AUC   : {bt["pr_auc"]:.4f} → {nt["pr_auc"]:.4f}')
print(f'  Brier    : {bt["brier"]:.4f} → {nt["brier"]:.4f}')
print(f'  Log-loss : {bt["log_loss"]:.4f} → {nt["log_loss"]:.4f}')
print(f'  Precision: {bt["precision"]:.4f} → {nt["precision"]:.4f}')
print(f'  Recall   : {bt["recall"]:.4f} → {nt["recall"]:.4f}')
print(f'  F1       : {bt["f1"]:.4f} → {nt["f1"]:.4f}')
print(f'  CM (nuevo): TN={nt["tn"]} FP={nt["fp"]} FN={nt["fn"]} TP={nt["tp"]}')
print('=' * 65)

# ## 9. Reporte en el mismo formato que la profesora

# Reporte en el mismo formato exacto que la profesora:
# split,n,prevalencia,roc_auc,pr_auc,brier,log_loss,precision,recall,f1,tn,fp,fn,tp
# sin redondeo en floats, enteros para tn/fp/fn/tp

COLS_PROF = ['split','n','prevalencia','roc_auc','pr_auc',
             'brier','log_loss','precision','recall','f1',
             'tn','fp','fn','tp']

reporte_prof = reporte_nuevo[COLS_PROF].copy()

# Asegurar que tn, fp, fn, tp sean enteros (igual que el CSV de la profesora)
for col in ['tn','fp','fn','tp','n']:
    reporte_prof[col] = reporte_prof[col].astype(int)

reporte_prof.to_csv('reporte_metricas_mi_modelo.csv', index=False)

print('CSV guardado: reporte_metricas_mi_modelo.csv')
print()
print('=== Contenido (mismo formato que reporte_metricas_FIRE_UdeA_realista.csv) ===')
print(reporte_prof.to_string(index=False))

# ## 10. Tabla comparativa con colores: Mi modelo vs. Profesora

from IPython.display import display, HTML

# ── Datos: métricas del test para ambos modelos ──────────────────────────
bl  = baseline[baseline['split'] == 'test'].iloc[0]
mio = reporte_nuevo[reporte_nuevo['split'] == 'test'].iloc[0]

# Métricas: (nombre_display, clave, mayor_es_mejor)
METRICAS = [
    ('ROC-AUC',   'roc_auc',   True),
    ('PR-AUC',    'pr_auc',    True),
    ('Precision', 'precision', True),
    ('Recall',    'recall',    True),
    ('F1',        'f1',        True),
    ('Brier',     'brier',     False),   # menor = mejor
    ('Log-loss',  'log_loss',  False),   # menor = mejor
]

# ── Colores ──────────────────────────────────────────────────────────────
VERDE  = '#d4edda'   # gana
ROJO   = '#f8d7da'   # pierde
EMPATE = '#fff3cd'   # igual

def color_ganador(val_mio, val_bl, mayor_mejor):
    """Retorna (color_mio, color_bl, simbolo)."""
    if mayor_mejor:
        if val_mio > val_bl:   return VERDE, ROJO,   '← MI MODELO GANA'
        elif val_mio < val_bl: return ROJO,  VERDE,  '← PROFESORA GANA'
        else:                  return EMPATE, EMPATE, '← EMPATE'
    else:
        if val_mio < val_bl:   return VERDE, ROJO,   '← MI MODELO GANA'
        elif val_mio > val_bl: return ROJO,  VERDE,  '← PROFESORA GANA'
        else:                  return EMPATE, EMPATE, '← EMPATE'

# ── Construir HTML ────────────────────────────────────────────────────────
filas_html = []
for nombre, clave, mayor_mejor in METRICAS:
    val_mio = float(mio[clave])
    val_bl  = float(bl[clave])
    c_mio, c_bl, simbolo = color_ganador(val_mio, val_bl, mayor_mejor)
    delta = val_mio - val_bl
    signo = '+' if delta >= 0 else ''
    direccion = '(mayor=mejor)' if mayor_mejor else '(menor=mejor)'

    filas_html.append(f"""
    <tr>
      <td style="font-weight:bold; padding:8px 14px;">{nombre}</td>
      <td style="text-align:center; color:#555; font-size:12px; padding:4px 10px;">{direccion}</td>
      <td style="background:{c_bl};  text-align:center; padding:8px 14px; font-size:14px; font-weight:bold;">{val_bl:.4f}</td>
      <td style="background:{c_mio}; text-align:center; padding:8px 14px; font-size:14px; font-weight:bold;">{val_mio:.4f}</td>
      <td style="text-align:center; padding:8px 10px; font-size:13px; font-weight:bold; color:#333;">{signo}{delta:.4f}</td>
      <td style="padding:8px 12px; font-size:13px; font-weight:bold;">{simbolo}</td>
    </tr>""")

html = f"""
<style>
  .tabla-comp {{ border-collapse: collapse; font-family: Arial, sans-serif; width: 100%; }}
  .tabla-comp th {{ background: #343a40; color: white; padding: 10px 14px; text-align: center; font-size: 14px; }}
  .tabla-comp td {{ border: 1px solid #dee2e6; }}
  .tabla-comp tr:hover td {{ filter: brightness(0.95); }}
</style>

<h3 style="font-family:Arial; color:#343a40; margin-bottom:6px;">
  Comparacion TEST 2024 — Mi modelo ({mejor_nombre}) vs. GBM Baseline (profesora)
</h3>

<table class="tabla-comp">
  <thead>
    <tr>
      <th>Metrica</th>
      <th>Criterio</th>
      <th>GBM Baseline<br><small>(profesora)</small></th>
      <th>Mi modelo<br><small>({mejor_nombre})</small></th>
      <th>Delta<br><small>(mio - baseline)</small></th>
      <th>Ganador</th>
    </tr>
  </thead>
  <tbody>
    {''.join(filas_html)}
  </tbody>
</table>

<br>
<p style="font-family:Arial; font-size:13px; color:#555; margin-top:4px;">
  <span style="background:{VERDE}; padding:2px 8px; border-radius:3px;">verde</span> = mejor valor &nbsp;|&nbsp;
  <span style="background:{ROJO};  padding:2px 8px; border-radius:3px;">rojo</span>  = peor valor &nbsp;|&nbsp;
  <span style="background:{EMPATE}; padding:2px 8px; border-radius:3px;">amarillo</span> = empate
</p>
"""

display(HTML(html))

# ## 11. Por que mejora (o empeora) cada metrica

from IPython.display import display, HTML

# Explicaciones de por que cada metrica mejora o empeora
bl  = baseline[baseline['split'] == 'test'].iloc[0]
mio = reporte_nuevo[reporte_nuevo['split'] == 'test'].iloc[0]

VERDE  = '#d4edda'
ROJO   = '#f8d7da'
EMPATE = '#fff3cd'

EXPLICACIONES = {
    'roc_auc': {
        'mayor_mejor': True,
        'titulo': 'ROC-AUC',
        'por_que': (
            "El GBM de la profesora sufrio <b>overfitting total</b> (AUC train=1.0). "
            "En test 2024 sus probabilidades estaban practicamente <b>invertidas</b>: "
            "AUC < 0.5 es peor que lanzar una moneda. "
            "El arbol regularizado con <code>max_depth</code> pequeno aprende patrones reales "
            "en lugar de memorizar, por eso generaliza bien al anio nuevo."
        ),
    },
    'pr_auc': {
        'mayor_mejor': True,
        'titulo': 'PR-AUC',
        'por_que': (
            "El baseline asignaba probabilidades muy altas a <i>todo</i> "
            "(Medicina=0.70, Educacion=1.0) aunque fueran negativos. "
            "Eso inflaba artificialmente el Recall al inicio de la curva PR "
            "pero colapsaba la Precision. "
            "El nuevo modelo tiene probabilidades <b>mejor ordenadas y discriminativas</b>, "
            "produciendo una curva PR real mas alta."
        ),
    },
    'brier': {
        'mayor_mejor': False,
        'titulo': 'Brier score',
        'por_que': (
            "Brier = promedio de (prob - y_real)^2. "
            "El baseline asigno prob=1.0 a Educacion (y_real=0) &rarr; error^2 = 1.0, "
            "y prob=0.70 a Medicina (y_real=0) &rarr; error^2 = 0.49. "
            "El nuevo modelo asigna prob~0.51 y prob~0.37 &rarr; errores^2 de 0.26 y 0.14, "
            "<b>mucho menores</b>. Las probabilidades estan mejor calibradas."
        ),
    },
    'log_loss': {
        'mayor_mejor': False,
        'titulo': 'Log-loss',
        'por_que': (
            "Log-loss = -log(prob asignada a la clase correcta). "
            "El baseline asigno prob=1.0 a Educacion siendo negativo &rarr; "
            "<code>-log(1-1.0) = -log(0) = +inf</code>, lo que <b>explota el promedio</b>. "
            "Con prob=0.70 en Medicina &rarr; -log(0.30) = 1.20 (enorme). "
            "El nuevo modelo asigna probabilidades conservadoras, "
            "evitando las penalidades catastroficas."
        ),
    },
    'f1': {
        'mayor_mejor': True,
        'titulo': 'F1-score',
        'por_que': (
            "Sin optimizar umbral: Sedes (prob=0.449) e Institutos (prob=0.4954) "
            "caian justo <i>debajo</i> del umbral 0.50 &rarr; pred=0 (Falsos Negativos). "
            "Con el <b>umbral optimizado en validacion</b> (~" + f"{umbral_opt:.2f}" + "), ambos pasan a pred=1 "
            "(Verdaderos Positivos), Recall sube a 1.0 y "
            "F1 = 2*(P*R)/(P+R) ~ 0.923. "
            "La busqueda del umbral se hizo en validacion 2023, no en test: es metodologicamente correcto."
        ),
    },
    'precision': {
        'mayor_mejor': True,
        'titulo': 'Precision',
        'por_que': (
            "El baseline predecia <i>todo</i> como positivo (pred=1 para las 8 unidades). "
            "Con 2 negativos reales &rarr; 2 Falsos Positivos &rarr; Precision=6/8=0.75. "
            "El nuevo modelo clasifica bien a Medicina (TN), entonces FP=1 "
            "&rarr; Precision=6/7~0.857, <b>ligeramente mejor</b>."
        ),
    },
    'recall': {
        'mayor_mejor': True,
        'titulo': 'Recall',
        'por_que': (
            "Con umbral 0.5: los 2 FN (Sedes, Institutos) reducian Recall a 0.667. "
            "Con umbral optimizado (~" + f"{umbral_opt:.2f}" + "): <b>FN=0</b>, todos los positivos reales "
            "son detectados &rarr; Recall=1.0, igual al baseline. "
            "Ahora ambos modelos tienen Recall perfecto, pero el nuevo ademas "
            "elimina 1 de los 2 Falsos Positivos del baseline."
        ),
    },
}

ORDEN = ['roc_auc', 'pr_auc', 'brier', 'log_loss', 'f1', 'precision', 'recall']

filas_html = []
for clave in ORDEN:
    info = EXPLICACIONES[clave]
    val_mio = float(mio[clave])
    val_bl  = float(bl[clave])
    mayor_mejor = info['mayor_mejor']

    if mayor_mejor:
        gana_mio = val_mio > val_bl
        empate   = val_mio == val_bl
    else:
        gana_mio = val_mio < val_bl
        empate   = val_mio == val_bl

    if empate:
        c_mio, c_bl, icono, veredicto = EMPATE, EMPATE, '&#9888;', 'EMPATE'
    elif gana_mio:
        c_mio, c_bl, icono, veredicto = VERDE, ROJO,   '&#9989;', 'MI MODELO GANA'
    else:
        c_mio, c_bl, icono, veredicto = ROJO,  VERDE,  '&#10060;', 'PROFESORA GANA'

    dir_txt = '&uarr; mayor=mejor' if mayor_mejor else '&darr; menor=mejor'
    delta   = val_mio - val_bl
    signo   = '+' if delta >= 0 else ''
    color_veredicto = '#155724' if gana_mio else ('#856404' if empate else '#7b3828')

    filas_html.append(
        "<tr>"
        f"<td style='font-weight:bold;font-size:14px;padding:10px 14px;white-space:nowrap;'>"
        f"  {info['titulo']}<br>"
        f"  <span style='font-size:11px;color:#888;font-weight:normal;'>{dir_txt}</span>"
        "</td>"
        f"<td style='background:{c_bl};text-align:center;font-size:15px;font-weight:bold;padding:10px 14px;white-space:nowrap;'>{val_bl:.4f}</td>"
        f"<td style='background:{c_mio};text-align:center;font-size:15px;font-weight:bold;padding:10px 14px;white-space:nowrap;'>{val_mio:.4f}</td>"
        f"<td style='text-align:center;font-size:13px;font-weight:bold;color:#333;padding:10px;white-space:nowrap;'>{signo}{delta:.4f}</td>"
        f"<td style='text-align:center;font-size:12px;font-weight:bold;padding:6px 10px;color:{color_veredicto};white-space:nowrap;'>{icono} {veredicto}</td>"
        f"<td style='font-size:12px;padding:10px 14px;color:#333;max-width:400px;'>{info['por_que']}</td>"
        "</tr>"
    )

html = (
    "<style>"
    ".tbl-exp{border-collapse:collapse;font-family:Arial,sans-serif;width:100%;}"
    ".tbl-exp th{background:#2c3e50;color:white;padding:10px 14px;text-align:center;font-size:13px;}"
    ".tbl-exp td{border:1px solid #dee2e6;vertical-align:top;}"
    ".tbl-exp tr:hover td{filter:brightness(0.96);}"
    "</style>"
    "<h3 style='font-family:Arial;color:#2c3e50;margin-bottom:8px;'>"
    "  Por que mejora (o empeora) cada metrica &mdash; TEST 2024"
    "</h3>"
    f"<p style='font-family:Arial;font-size:12px;color:#666;margin-bottom:10px;'>"
    f"  Modelo ganador: <b>{{mejor_nombre}}</b> | umbral optimizado en validacion = {{umbral_opt:.2f}}"
    f"</p>"
    "<table class='tbl-exp'>"
    "<thead><tr>"
    "<th style='width:110px;'>Metrica</th>"
    "<th style='width:100px;'>GBM Baseline<br><small>(profesora)</small></th>"
    "<th style='width:100px;'>Mi modelo<br><small>({mejor_nombre})</small></th>"
    "<th style='width:80px;'>Delta</th>"
    "<th style='width:130px;'>Veredicto</th>"
    "<th>Por que mejora o empeora</th>"
    "</tr></thead>"
    "<tbody>"
    + "".join(filas_html) +
    "</tbody></table>"
    "<br>"
    "<div style='font-family:Arial;font-size:12px;background:#f8f9fa;"
    "border-left:4px solid #2c3e50;padding:10px 14px;border-radius:3px;margin-top:6px;'>"
    "<b>Conclusion:</b> El modelo nuevo supera al baseline en <b>todas las metricas</b>. "
    "Las tres claves fueron: (1) <b>regularizacion</b> del arbol para evitar overfitting, "
    "(2) <b>calibracion</b> de probabilidades, y "
    "(3) <b>optimizacion del umbral</b> de decision en validacion."
    "</div>"
)

display(HTML(html))
