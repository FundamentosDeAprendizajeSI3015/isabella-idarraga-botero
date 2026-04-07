# Fundamentos de Aprendizaje Automatico

**Autora:** Isabella Idarraga Botero  
**Curso:** Fundamentos de Aprendizaje Automatico  
**Periodo:** 2026-1

---

## Descripcion General

Repositorio de trabajo para el curso de Fundamentos de Aprendizaje Automatico. Contiene laboratorios, ejercicios practicos, un informe teorico-practico y un proyecto integrador que cubren el ciclo de vida completo de un proyecto de Machine Learning: desde la exploracion y limpieza de datos hasta clustering, arboles de decision y preparacion de datasets para modelado.

---

## Contenido

| Carpeta | Tema | Descripcion |
|---------|------|-------------|
| `lecture2/` | Clasificacion con Penguins | Ciclo de vida completo de ML: EDA, PCA, Regresion Logistica y evaluacion sobre el dataset Palmer Penguins. |
| `lecture3/` | Preprocesamiento Fintech | Limpieza, feature engineering financiero y division temporal train/test sobre un dataset sintetico de metricas fintech. |
| `lecture4/` | EDA Peliculas | Analisis Exploratorio de Datos completo sobre un dataset de peliculas: limpieza, medidas de tendencia central y dispersion, deteccion/eliminacion de outliers, histogramas, graficos de dispersion y transformaciones de columnas. |
| `lecture5/` | Regresion Peliculas | Regresion Lineal (Ridge y Lasso con RandomizedSearchCV) para predecir RATING y Regresion Logistica para clasificar peliculas exitosas, con validacion cruzada y metricas de evaluacion. |
| `lecture6/` | Arboles de Decision — Abandono | Random Forest y Gradient Boosting para predecir si un usuario abandonara un libro, usando el dataset del Proyecto 1. |
| `lecture8/` | Pipeline FIRE UdeA | Pipeline de analisis exploratorio y modelado con arboles de decision sobre el dataset financiero FIRE UdeA (real y sintetico). |
| `lecture9/` | Agrupamiento (Clustering) | K-Means y DBSCAN aplicados al dataset FIRE UdeA. Incluye metodo del codo, analisis de clusters y versiones sobre datos sinteticos y reales. |
| `lecture10/` | Auditoria Financiera 3D | Analisis avanzado con K-Means y UMAP 3D sobre el dataset FIRE UdeA realista, segmentado por unidad financiera (Nivel Central, Educacion, Facultades, etc.) para auditar la calidad del etiquetado de riesgo. |
| `informe-teorico-practico/` | Informe Clustering + Supervisado | Analisis no supervisado completo (K-Means, Fuzzy C-Means, Subtractive, DBSCAN, Jerarquico, GMM) sobre el sistema de prediccion de abandono de lectura, seguido de re-evaluacion de etiquetas por consenso de clusters y comparacion de modelos supervisados (Arbol de Decision, Regresion Logistica, Regresion Lineal) con etiquetas originales vs re-evaluadas. |
| `Proyecto1/` | Prediccion de Abandono de Lectura | Pipeline completo: simulacion de sesiones de lectura, limpieza, EDA, transformaciones, NLP sobre reviews de Goodreads y merge de features. Ver [README del proyecto](Proyecto1/README.md). |

---

## Requisitos

- Python 3.9 o superior
- Dependencias principales:

```
pandas >= 1.5.0
numpy >= 1.23.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
scikit-learn >= 1.2.0
```

Instalacion rapida (desde la carpeta Proyecto1):

```bash
pip install -r requirements.txt
```

---

## Como Ejecutar

Cada modulo es independiente. Para ejecutar cualquier script:

```bash
# Lecture 2
cd lecture2
python idarraga_isabella_penguins_analysis.py

# Lecture 3
cd lecture3
python lab_fintech_sintetico_2025.py

# Lecture 4
cd lecture4
python eda_peliculas.py

# Lecture 5
cd lecture5
python regresion_peliculas.py

# Lecture 6 (requiere haber ejecutado Proyecto 1 completo antes)
cd lecture6
python arboles_abandono.py

# Lecture 8
cd lecture8
python 01_pipeline_analisis.py
python 03_modelado_arbol_decision.py

# Lecture 9
cd lecture9
python agrupamiento_FIRE_UdeA.py
# o version sintetica:
python agrupamiento_sintetico_FIRE_UdeA.py

# Lecture 10
cd lecture10
python analisis_por_dependencia_realista.py

# Informe teorico-practico (requiere haber ejecutado Proyecto 1 completo antes)
cd informe-teorico-practico
python analisis_clustering_supervisado.py

# Proyecto 1 (ejecutar en orden)
cd Proyecto1
python 01_simular_datos_lectura.py
python 01b_analizar_reviews.py
python 02_limpieza_datos.py
python 03_eda_analisis.py
python 04_transformaciones.py
python 05_merge_nlp_features.py
python 06_visualizar_reviews.py
```

> **Nota:** El Proyecto 1 requiere los archivos JSON y CSV de Goodreads en la carpeta `datos_goodreads/`. Consultar el README del proyecto para instrucciones de descarga.

> **Dependencia de datos:** `lecture6/` e `informe-teorico-practico/` consumen el archivo `Proyecto1/datos_transformados.csv`. Es necesario ejecutar el pipeline completo del Proyecto 1 (pasos 01 al 05) antes de correr esos modulos.

---

