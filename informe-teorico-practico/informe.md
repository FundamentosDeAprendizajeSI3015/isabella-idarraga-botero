# Informe: Análisis No Supervisado y Supervisado
## Sistema de Predicción de Abandono de Lectura — Goodreads
**Isabella Idarraga | Fundamentos de Aprendizaje Automático**

---

## 1. Descripción del Problema

El objetivo es predecir si un usuario va a abandonar un libro antes de terminarlo. Se utilizaron 388,907 sesiones de lectura simuladas a partir del dataset de Goodreads (UCSD), con 73 features que incluyen comportamiento de sesión, agregaciones por usuario/libro y scores NLP extraídos de 2M reviews.

**Variable target:** `abandono` (1 = abandonó el libro, 0 = lo completó)
**Distribución:** 76.5% completados / 23.5% abandonados (sobre todas las sesiones)

> **Nota sobre la distribución:** El README del Proyecto 1 reportaba 62.1% de abandono. La diferencia se debe a que ese porcentaje se calculaba solo sobre la **última sesión por libro** (49,993 registros únicos de lectura), mientras que aquí se trabaja con **todas las sesiones** (388,907 filas), muchas de las cuales son sesiones intermedias de libros que el usuario sí completó finalmente. Los criterios del target son los mismos (`progress < 90%` AND `inactividad > 21 días`), pero el denominador cambia.

---

## 2. Datos Utilizados

| Archivo | Descripción |
|---|---|
| `Proyecto1/datos_transformados.csv` | 388,907 filas × 73 columnas |

**14 features usadas para clustering:**

| Feature | Descripción |
|---|---|
| `duration_minutes_scaled` | Duración de sesión (estandarizada) |
| `pages_read_scaled` | Páginas leídas por sesión |
| `completion_pct_start_scaled` | Progreso al inicio de la sesión |
| `completion_pct_end_scaled` | Progreso al final de la sesión |
| `velocidad_lectura_scaled` | Velocidad de lectura (páginas/min) |
| `ratio_progreso_scaled` | Ratio de avance en la sesión |
| `num_sesiones_scaled` | Número de sesiones del usuario |
| `tasa_abandono` | Tasa histórica de abandono del usuario |
| `progreso_promedio` | Progreso promedio del usuario |
| `abandono_score_scaled` | Score NLP de abandono del libro |
| `engagement_score_scaled` | Score NLP de engagement del libro |
| `complejidad_score_scaled` | Score NLP de complejidad del libro |
| `ritmo_score_scaled` | Score NLP de ritmo narrativo |
| `sentimiento_promedio_scaled` | Sentimiento promedio de reviews |

**PCA:** 2 componentes explican el **97.4%** de la varianza — los datos tienen estructura muy concentrada.

---

## 3. Parámetro de Muestreo

El notebook tiene una celda de configuración al inicio con `SAMPLE_PCT` (0.0 a 1.0):

| SAMPLE_PCT | Filas usadas | Uso recomendado |
|---|---|---|
| 0.05 | ~19,445 | Prueba rápida |
| 0.10 | ~38,891 | Desarrollo |
| 0.30 | ~116,672 | Validación |
| 1.00 | 388,907 | Análisis completo |

**Este informe usa SAMPLE_PCT = 1.0 (dataset completo).**

---

## 4. Análisis No Supervisado

### 4.1 K-Means

**Descripción:** Algoritmo de particionamiento que minimiza la varianza intra-cluster (WCSS). Asigna cada punto al centroide más cercano e itera hasta convergencia.

**Cómo funciona:**
1. Inicializa k centroides con k-means++ (inicialización inteligente)
2. Asigna cada punto al centroide más cercano
3. Recalcula centroides como la media de los puntos asignados
4. Repite hasta convergencia

**Selección de k:** Se evaluaron k ∈ [2, 10] con el método del codo y Silhouette Score:

| k | Inercia | Silhouette |
|---|---|---|
| 2 | 14,226,087 | **0.8225** |
| 3 | 10,676,091 | 0.7363 |
| 4 | 8,726,504 | 0.3048 |
| 5 | 7,477,086 | 0.3048 |

El k óptimo por Silhouette es **k=2**, pero se usó **k=3** para tener mayor granularidad en la segmentación.

**Resultados (k=3):**
- Silhouette Score: **0.7355** → clusters bien separados
- Davies-Bouldin: **0.6273** → baja superposición entre clusters
- Calinski-Harabasz: **2,099,022.71** → muy alta densidad intra-cluster

**Distribución de abandono por cluster:**

| Cluster | Tasa abandono | Tamaño | Interpretación |
|---|---|---|---|
| 0 | 1.8% | 301,273 | Lectores que completan — perfil mayoritario |
| 1 | 99.7% | 62,774 | Abandono casi total — sesiones críticas |
| 2 | 94.2% | 24,860 | Alto riesgo de abandono |

K-Means logró separar perfectamente los tres perfiles de comportamiento.

---

### 4.2 Fuzzy C-Means (FCM)

**Descripción:** Extensión "suave" de K-Means donde cada punto pertenece a múltiples clusters con grados de membresía entre 0 y 1.

**Cómo funciona:**
1. Inicializa matriz de membresía aleatoriamente
2. Calcula centroides ponderados por membresía al cuadrado (parámetro m=2)
3. Actualiza membresías inversamente proporcionales a la distancia al centroide
4. Repite hasta convergencia

**Ventaja:** Captura casos ambiguos — un usuario puede pertenecer parcialmente a varios perfiles de lectura.

**Resultados (c=3):**
- FPC (Fuzzy Partition Coefficient): **0.6869** → partición moderadamente definida (1 = perfectamente crisp)
- Silhouette Score: **0.3652**
- Davies-Bouldin: **0.9497**
- Membresía promedio máxima: **0.7737** → los puntos pertenecen mayoritariamente a un cluster pero con cierta ambigüedad

FCM obtuvo métricas menores que K-Means porque al ser "suave" no fuerza separaciones tan definidas.

---

### 4.3 Subtractive Clustering

**Descripción:** Algoritmo que encuentra automáticamente el número de clusters basándose en densidad de puntos. No requiere especificar k.

**Cómo funciona:**
1. Calcula una función de densidad para cada punto (cuántos vecinos tiene en radio r_a=0.5)
2. El punto con mayor densidad se convierte en primer centroide
3. Se reduce la densidad de todos los puntos cercanos al centroide encontrado
4. Repite buscando el siguiente pico de densidad residual
5. Para cuando la densidad máxima cae bajo epsilon_lower=0.15

**Nota:** Es O(n²) en memoria — se ejecutó sobre submuestra de 3,000 puntos para encontrar centros, luego se asignaron todos los puntos al centroide más cercano.

**Resultados:**
- Clusters encontrados automáticamente: **4**
- Silhouette Score: **0.2512**
- Davies-Bouldin: **1.3970**

El algoritmo encontró 4 clusters de forma autónoma. Las métricas son menores porque los centros se calcularon sobre submuestra, lo que introduce imprecisión en la asignación del dataset completo.

---

### 4.4 DBSCAN

**Descripción:** Density-Based Spatial Clustering of Applications with Noise. Encuentra clusters de forma arbitraria e identifica outliers automáticamente como "ruido".

**Cómo funciona:**
1. Para cada punto, cuenta vecinos en radio `eps=0.5`
2. Si tiene ≥ `min_samples=10` vecinos → punto "core"
3. Clusters se forman conectando puntos core vecinos
4. Puntos dentro del radio de un core pero sin suficientes vecinos → borde
5. El resto → ruido (etiqueta -1)

**Resultados:**
- Clusters encontrados: **234** — el espacio de features tiene muchísima micro-estructura local
- Puntos de ruido: **162,325 (41.7%)** — casi la mitad del dataset son outliers según este criterio
- Silhouette (sin ruido): **-0.4001** → los clusters son tan pequeños y numerosos que se superponen

DBSCAN con estos parámetros no es adecuado para este dataset — la alta dimensionalidad (14 features) y la densidad uniforme de los datos hacen que eps=0.5 genere fragmentación excesiva. Requeriría ajuste fino de eps con la curva k-distance.

---

### 4.5 Clustering Jerárquico (Agglomerative, Ward)

**Descripción:** Construye una jerarquía de clusters de abajo hacia arriba. El criterio Ward minimiza la varianza total al fusionar clusters.

**Cómo funciona:**
1. Cada punto empieza como su propio cluster
2. Se fusionan los dos clusters que minimizan el incremento de varianza total (criterio Ward)
3. Se repite hasta tener k=3 clusters
4. El dendrograma muestra el árbol completo de fusiones

**Nota técnica:** Ward requiere calcular una matriz de distancias completa (n×n), lo que con 388,907 puntos requeriría 563 GB de RAM. Se entrenó sobre submuestra de 10,000 puntos, se calcularon los centroides de cada cluster y se asignaron todos los puntos al centroide más cercano.

**Resultados (k=3):**
- Silhouette Score: **0.7316** → muy similar a K-Means
- Davies-Bouldin: **0.6356**

Los resultados son casi idénticos a K-Means, lo que valida que la estructura de 3 clusters es robusta.

---

### 4.6 Gaussian Mixture Model (GMM)

**Descripción:** Modelo probabilístico que asume que los datos son una mezcla de distribuciones Gaussianas. Cada componente tiene su propia media, covarianza y peso.

**Cómo funciona (algoritmo EM):**
1. **E-step:** Calcula la probabilidad de que cada punto pertenezca a cada componente Gaussiana
2. **M-step:** Actualiza parámetros (media, covarianza, peso) maximizando la log-verosimilitud
3. Repite hasta convergencia

**Selección de componentes por BIC:**

| n | BIC | AIC |
|---|---|---|
| 2 | 10,433,087 | 10,430,488 |
| 3 | -7,144,586 | -7,148,489 |
| 4 | -8,427,193 | -8,432,400 |
| 5 | -9,071,978 | -9,078,490 |
| 6 | -10,966,781 | -10,974,597 |
| 7 | **-14,624,358** | -14,633,479 |

El BIC óptimo indica **7 componentes**, pero se usaron 3 para mantener comparabilidad con los demás métodos.

**Resultados (3 componentes):**
- Silhouette Score: **0.3152**
- Davies-Bouldin: **1.7587**
- Certeza promedio de asignación: **0.9996** → los puntos pertenecen casi con certeza total a un componente

GMM tiene menor Silhouette porque sus clusters Gaussianos se superponen más que los de K-Means, pero la certeza de asignación es casi perfecta.

---

### 4.7 Comparación de Métodos

| Método | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | N Clusters |
|---|---|---|---|---|
| **K-Means** | **0.7363** | **0.6273** | **2,099,023** | 3 |
| **Jerárquico (Ward)** | 0.7316 | 0.6356 | 2,096,998 | 3 |
| Fuzzy C-Means | 0.3652 | 0.9497 | 1,854,602 | 3 |
| GMM | 0.3152 | 1.7587 | 366,658 | 3 |
| Subtractive | 0.2512 | 1.3970 | 1,247,524 | 4 (auto) |
| DBSCAN | -0.4001 | 1.0778 | — | 234 (auto) |

**Conclusión:** K-Means y Clustering Jerárquico (Ward) son los mejores métodos para este dataset, con Silhouette ~0.73 y Davies-Bouldin ~0.63. La estructura de 3 clusters es clara y consistente entre métodos. DBSCAN no es apropiado con los parámetros actuales dado el alto número de clusters generados y el 41.7% de ruido.

---

## 5. Re-evaluación de Etiquetas

**Motivación:** En datasets del mundo real, hasta el 30% de las etiquetas pueden ser incorrectas. El clustering no supervisado revela la estructura "real" de los datos independientemente de las etiquetas asignadas.

**Metodología:**
1. Cada método mapea sus clusters a una etiqueta binaria (abandono=1 si la tasa del cluster ≥ 50%)
2. Voto mayoritario de los 6 algoritmos: si ≥3/6 dicen abandono=1 → nuevo label = 1
3. Puntos donde el consenso difiere de la etiqueta original → candidatos a re-etiquetar

**Distribución de votos:**

| Votos abandono=1 | Puntos | % |
|---|---|---|
| 0/6 | 295,768 | 76.1% |
| 1/6 | 5,578 | 1.4% |
| 2/6 | 1,226 | 0.3% |
| 3/6 | 1,442 | 0.4% |
| 4/6 | 2,029 | 0.5% |
| 5/6 | 21,218 | 5.5% |
| 6/6 | 61,646 | 15.9% |

El 92% de los puntos tienen acuerdo total (0/6 o 6/6), lo que indica etiquetas muy consistentes con la estructura de los datos.

**Resultado:**
- Etiquetas cambiadas: **6,109 (1.6%)** — mucho menos que el 30% esperado teóricamente
- Esto confirma que las etiquetas originales son de alta calidad, producto de una simulación bien diseñada

**Distribución antes vs después:**

| | Originales | Re-evaluadas |
|---|---|---|
| Completados (0) | 297,379 (76.5%) | 302,572 (77.8%) |
| Abandonados (1) | 91,528 (23.5%) | 86,335 (22.2%) |

---

## 6. Modelos Supervisados

### 6.1 Árbol de Decisión

**Descripción:** Modelo que particiona el espacio de features mediante reglas binarias (if/else), eligiendo en cada nodo el split que maximiza la ganancia de información.

**Parámetros:** `max_depth=5`, `class_weight='balanced'`

**Resultados:**

| Métrica | Etiquetas Originales | Etiquetas Re-evaluadas |
|---|---|---|
| Accuracy (test) | 0.9984 | **0.9996** |
| ROC-AUC | 0.9999 | **1.0000** |
| CV-5 Accuracy | 0.9986 | **0.9997** |

⚠️ **Alerta: Data Leakage.** Accuracy >99.8% es una señal de fuga de datos. Features como `completion_pct_end_scaled` y `ratio_progreso_scaled` codifican directamente el progreso final de la sesión, que es exactamente el criterio que define el target (`progress_end < 90%`). El modelo esencialmente ve la respuesta antes de predecir. Ver sección 8 para el análisis sin leakage.

---

### 6.2 Regresión Logística

**Descripción:** Modelo lineal que estima la probabilidad de abandono usando la función sigmoide sobre una combinación lineal de features.

**Parámetros:** `solver=lbfgs`, `max_iter=1000`, `class_weight='balanced'`

**Resultados:**

| Métrica | Etiquetas Originales | Etiquetas Re-evaluadas |
|---|---|---|
| Accuracy (test) | **0.9991** | 0.9987 |
| ROC-AUC | **1.0000** | 1.0000 |
| CV-5 Accuracy | **0.9989** | 0.9988 |

⚠️ **Alerta: Data Leakage.** Mismo problema que el árbol — el ROC-AUC=1.0 y la separabilidad perfecta se deben a que el modelo tiene acceso a features derivadas del target. Ver sección 8.

---

### 6.3 Regresión Lineal

**Descripción:** Predice `dias_inactividad` (variable continua) como proxy del abandono. Se compararon dos versiones: dataset completo vs solo puntos con ≥5/6 algoritmos en acuerdo (puntos "seguros").

**Puntos seguros (≥5/6 acuerdo):** 384,210 (98.8% del total)

**Resultados:**

| Métrica | Dataset Original | Solo Puntos Seguros |
|---|---|---|
| R² | 0.0610 | 0.0670 |
| RMSE | 88.5880 | 88.6122 |
| MAE | 74.5589 | 74.5366 |

El R² de ~0.06 indica que las features de sesión explican apenas el 6% de la varianza en `dias_inactividad`. Esto es esperado: los días de inactividad dependen de factores externos (tiempo disponible, interés cambiante) que no están capturados en las features de sesión. La diferencia entre dataset completo y puntos seguros es mínima, lo que confirma que el 1.6% de etiquetas re-evaluadas no tiene impacto significativo.

---

## 7. Comparación Final: Originales vs Re-evaluadas (CON leakage)

| Modelo | Accuracy | ROC-AUC | CV-5 Accuracy | Etiquetas |
|---|---|---|---|---|
| Árbol de Decisión | 0.9984 | 0.9999 | 0.9986 | Original |
| Árbol de Decisión | **0.9996** | **1.0000** | **0.9997** | Re-evaluada |
| Reg. Logística | **0.9991** | 1.0000 | **0.9989** | Original |
| Reg. Logística | 0.9987 | 1.0000 | 0.9988 | Re-evaluada |

---

## 8. Análisis Sin Data Leakage

Los resultados de las secciones 6 y 7 presentan data leakage porque incluyen features derivadas directamente del target. Esta sección repite el análisis usando **solo features que estarían disponibles antes de conocer el resultado de la sesión**: características NLP del libro, historial del usuario y contexto de la sesión (sin progreso final ni tasas de abandono del mismo dataset).

**17 features limpias usadas:**
- Sesión: `duration_minutes_scaled`, `pages_read_scaled`, `velocidad_lectura_scaled`, `num_sesiones_scaled`, `es_fin_semana`, `periodo_dia_encoded`
- Historial de usuario: `num_libros_leidos`, `duracion_promedio_scaled`, `paginas_promedio_scaled`
- NLP del libro: `abandono_score_scaled`, `engagement_score_scaled`, `complejidad_score_scaled`, `ritmo_score_scaled`, `sentimiento_promedio_scaled`, `sentimiento_positivo_pct`, `sentimiento_negativo_pct`, `tiene_reviews`

### 8.1 Árbol de Decisión sin Leakage

| Métrica | Etiquetas Originales | Etiquetas Re-evaluadas |
|---|---|---|
| Accuracy (test) | **0.9774** | 0.9660 |
| ROC-AUC | **0.9956** | 0.9922 |
| CV-5 Accuracy | **0.9756** | 0.9652 |

Con etiquetas originales: Completado → precision=0.99, recall=0.98 / Abandonado → precision=0.95, recall=0.96

### 8.2 Regresión Logística sin Leakage

| Métrica | Etiquetas Originales | Etiquetas Re-evaluadas |
|---|---|---|
| Accuracy (test) | **0.9460** | 0.9347 |
| ROC-AUC | **0.9828** | 0.9728 |
| CV-5 Accuracy | **0.9440** | 0.9344 |

Con etiquetas originales: Completado → precision=0.97, recall=0.95 / Abandonado → precision=0.86, recall=0.92

### 8.3 Comparación CON vs SIN Leakage

| Modelo | Accuracy | ROC-AUC | CV-5 |
|---|---|---|---|
| Árbol **(CON leakage)** | 0.9984 | 0.9999 | 0.9986 |
| Árbol **(SIN leakage)** | 0.9774 | 0.9956 | 0.9756 |
| LogReg **(CON leakage)** | 0.9991 | 1.0000 | 0.9989 |
| LogReg **(SIN leakage)** | 0.9460 | 0.9828 | 0.9440 |

**Interpretación:**
- El árbol pierde solo ~2% de accuracy al eliminar el leakage → las features NLP y de duración ya son muy predictivas por sí solas
- La regresión logística pierde ~5% → el modelo lineal dependía más de las features con leakage para separar las clases
- Ambos modelos **siguen siendo muy buenos** sin leakage (árbol 97.7%, logística 94.6%), lo que valida que las features NLP del libro y el historial del usuario tienen genuino poder predictivo sobre el abandono
- Las etiquetas re-evaluadas dan resultados ligeramente peores en ambos modelos, lo que sugiere que el consenso de clustering no mejora la calidad del etiquetado original en este caso

> **Nota sobre leakage temporal:** Aunque `tasa_abandono` no está en las features limpias, vale mencionar que incluso si se incluyera, representaría un **leakage temporal**: se calcula sobre todas las sesiones del usuario en el mismo dataset, incluyendo sesiones futuras. En un sistema real de predicción, solo se tendría acceso al historial pasado del usuario. Esta distinción es importante al pasar de análisis offline a producción.

---

## 9. Conclusiones y Limitaciones

1. **Mejor método de clustering: K-Means** (Silhouette=0.7363, DB=0.6273), seguido muy de cerca por Clustering Jerárquico Ward (0.7316). Ambos identifican 3 perfiles claros: lectores que completan (cluster 0, 1.8% abandono), abandono casi total (cluster 1, 99.7%) y alto riesgo (cluster 2, 94.2%).

2. **Las etiquetas originales son de alta calidad:** Solo el 1.6% (6,109 puntos) fueron re-etiquetadas por consenso de los 6 algoritmos, muy por debajo del 30% teórico. Esto valida el proceso de simulación del Proyecto 1.

3. **Se detectó data leakage en los modelos supervisados iniciales:** Features como `completion_pct_end_scaled` y `ratio_progreso_scaled` derivan del mismo criterio que define el target, produciendo accuracy >99.8% artificialmente. Esto es un error metodológico importante a evitar.

4. **Sin leakage, los modelos siguen siendo muy buenos:** Árbol de decisión 97.7% accuracy y ROC-AUC=0.9956 / Regresión logística 94.6% accuracy y ROC-AUC=0.9828. Esto confirma que las features NLP del libro (engagement, complejidad, ritmo) y el historial del usuario tienen genuino poder predictivo.

5. **Las etiquetas re-evaluadas no mejoran los modelos sin leakage** — el árbol baja de 0.9774 a 0.9660 y la logística de 0.9460 a 0.9347 con las etiquetas re-evaluadas. El etiquetado original de la simulación es más limpio que el consenso de clustering.

6. **La regresión lineal de `dias_inactividad` tiene bajo poder predictivo** (R²=0.06): los días de inactividad dependen de factores externos no capturados en las features de sesión.

7. **DBSCAN no es adecuado** con los parámetros actuales (`eps=0.5`) — genera 234 micro-clusters y clasifica el 41.7% como ruido. La curva k-distance sugiere que el codo está alrededor de `eps=2–3` para este espacio de 14 features escaladas. Esta es una limitación conocida del análisis: el parámetro no fue ajustado tras visualizar la curva.

8. **Leakage temporal en `tasa_abandono`:** Aunque no se usó en el conjunto limpio, en el conjunto CON leakage esta feature se calcula sobre todo el dataset incluyendo sesiones futuras del mismo usuario. En producción, solo el historial pasado estaría disponible.

---

## 10. Visualizaciones Generadas (26 gráficas)

| Archivo | Descripción |
|---|---|
| `01_distribuciones_clave.png` | Distribución de variables clave por clase de abandono |
| `02_pca_varianza.png` | Varianza explicada por componentes PCA (97.4% en 2D) |
| `03_etiquetas_originales_pca.png` | Etiquetas originales en espacio PCA 2D |
| `04_kmeans_elbow.png` | Método del codo y Silhouette para K-Means (k=2 óptimo) |
| `05_kmeans_clusters.png` | 3 clusters K-Means con tasa de abandono por cluster |
| `06_fuzzy_cmeans.png` | Clusters FCM con intensidad = membresía |
| `07_subtractive_clustering.png` | 4 clusters Subtractive + tasa abandono por cluster |
| `08_dbscan_kdistance.png` | Curva k-distance para selección de eps |
| `09_dbscan_clusters.png` | 234 clusters DBSCAN con ruido (41.7%) |
| `10_dendrograma.png` | Dendrograma jerárquico Ward sobre submuestra |
| `11_jerarquico_clusters.png` | 3 clusters jerárquicos en PCA 2D |
| `12_gmm_bic.png` | BIC/AIC para selección de componentes (óptimo: 7) |
| `13_gmm_clusters.png` | Clusters GMM con certeza de asignación (99.96%) |
| `14_comparacion_clustering.png` | Métricas comparativas de los 6 métodos |
| `15_panel_todos_clustering.png` | Panel con todos los métodos lado a lado |
| `16_reevaluacion_etiquetas.png` | Original vs re-evaluada vs puntos cambiados (1.6%) |
| `17_acuerdo_clustering.png` | Nivel de acuerdo entre algoritmos (92% acuerdo total) |
| `18_arbol_importancia.png` | Top features por importancia en árbol de decisión |
| `19_arbol_estructura.png` | Estructura visual del árbol (primeros 3 niveles) |
| `20_roc_logistica.png` | Curvas ROC — Regresión Logística (AUC=1.0) |
| `21_confusion_logistica.png` | Matrices de confusión originales vs re-evaluadas |
| `22_regresion_lineal.png` | Real vs predicho para días de inactividad |
| `23_residuos_regresion.png` | Análisis de residuos de regresión lineal |
| `24_comparacion_final.png` | Comparación final de todos los modelos supervisados |
| `25_importancia_sin_leakage.png` | Importancia de features sin leakage (árbol de decisión) |
| `26_leakage_vs_sin_leakage.png` | Impacto del data leakage: CON vs SIN en accuracy y ROC-AUC |
