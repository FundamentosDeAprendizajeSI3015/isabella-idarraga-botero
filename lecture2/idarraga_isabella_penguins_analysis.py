# -*- coding: utf-8 -*-
"""
Ciclo de vida completo de ML usando el dataset Penguins.
Incluye: EDA, preparación, modelado y evaluación.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


#Carga de datos
penguins = sns.load_dataset("penguins")
print(penguins.head())


#Limpieza y preparación
penguins = penguins.dropna()

X = penguins.drop(columns=["species"])
y = penguins["species"]

# One-hot encoding para variables categóricas
X = pd.get_dummies(X, drop_first=True)


#Análisis exploratorio (EDA)
print("\nDistribución de clases:")
print(y.value_counts())

sns.pairplot(penguins, hue="species")
plt.suptitle("Pairplot Penguins", y=1.02)
plt.show()

# Reducción de dimensionalidad (PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=y,
    palette="Set2"
)
plt.title("PCA 2D - Penguins")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# Partición de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# Entrenamiento del modelo

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)


#Evaluación
y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=y.unique(),
            yticklabels=y.unique())
plt.title("Matriz de confusión - Penguins")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()
