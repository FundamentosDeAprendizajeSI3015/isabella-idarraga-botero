# Fundamentos de Aprendizaje Automatico

**Autora:** Isabella Idarraga Botero  
**curso:** Fundamentos de aprendizaje automatico  
**Periodo:** 2026-1 

---

## Descripcion General

Repositorio de trabajo para el curso de Fundamentos de Aprendizaje Automatico. Contiene laboratorios, ejercicios practicos y un proyecto integrador que cubren el ciclo de vida completo de un proyecto de Machine Learning: desde la exploracion y limpieza de datos hasta la ingenieria de features y la preparacion de datasets para modelado.

---

## Contenido

| Carpeta | Tema | Descripcion |
|---------|------|-------------|
| `lecture2/` | Clasificacion con Penguins | Ciclo de vida completo de ML: EDA, PCA, Regresion Logistica y evaluacion sobre el dataset Palmer Penguins. |
| `lecture3/` | Preprocesamiento Fintech | Limpieza, feature engineering financiero y division temporal train/test sobre un dataset sintetico de metricas fintech. |
| `Proyecto1/` | Recomendacion de Libros | Pipeline completo para un sistema de recomendacion con prediccion de abandono, integrando analisis NLP de reviews de Goodreads. Ver [README del proyecto](Proyecto1/README.md). |

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

# Proyecto 1 (ejecutar en orden)
cd Proyecto1
python 01_simular_datos_lectura.py
python 01b_analizar_reviews.py
python 02_limpieza_datos.py
python 03_eda_analisis.py
python 04_transformaciones.py
python 06_visualizar_reviews.py
```

> **Nota:** El Proyecto 1 requiere los archivos JSON y CSV de Goodreads en la carpeta `datos_goodreads/`. Consultar el README del proyecto para instrucciones de descarga.

---

