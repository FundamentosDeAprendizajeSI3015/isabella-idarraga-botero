"""
VISUALIZACIÓN DE FEATURES DE REVIEWS
=====================================
Genera gráficos para analizar las características extraídas de reviews

Autor: Isabella Idarraga
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualizar_features_reviews(features_file='features_reviews.csv',
                                 output_dir='graficos_eda'):
    """
    Crea visualizaciones de las features extraídas de reviews

    Parameters:
    -----------
    features_file : str
        Ruta al archivo CSV con features de reviews (output de 01b_analizar_reviews.py)
    output_dir : str
        Directorio donde guardar los gráficos generados
    """

    print("="*70)
    print("VISUALIZACIÓN DE FEATURES DE REVIEWS")
    print("="*70)

    # Cargar los datos con features previamente calculadas
    print("\nCargando features...")
    df = pd.read_csv(features_file)
    print(f"✓ Cargadas features de {len(df):,} libros")

    # Crear directorio de salida para almacenar gráficos
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Configuración de visualización
    # Usar estilo oscuro con grilla para mejor legibilidad
    plt.style.use('seaborn-v0_8-darkgrid')
    # Usar paleta de colores husl (perceptualmente uniforme)
    sns.set_palette("husl")
    
    # ========================================
    # GRÁFICO 1: Distribución de Scores
    # ========================================
    print("\n[1/5] Generando distribuciones de scores...")

    # Crear figura con 2 filas × 3 columnas para los 6 scores principales
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Lista de scores principales extraídos de reviews (varían en rango típicamente -1 a +1)
    scores = ['abandono_score', 'engagement_score', 'complejidad_score',
              'ritmo_score', 'emocional_score', 'sentimiento_promedio']

    # Iterar sobre cada score y crear histograma
    for idx, score in enumerate(scores):
        # Verificar que la columna existe en el dataframe
        if score not in df.columns:
            continue

        # Posicionar en la cuadrícula (fila, columna)
        ax = axes[idx // 3, idx % 3]

        # Crear histograma para ver la distribución
        df[score].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7, color='steelblue')

        # Calcular estadísticas de centralidad
        media = df[score].mean()  # Promedio aritmético
        mediana = df[score].median()  # Valor central (resistente a outliers)

        # Agregar líneas verticales para media y mediana (referencias visuales)
        ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.3f}')
        ax.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.3f}')

        # Configurar etiquetas y título
        ax.set_title(f'Distribución: {score}', fontsize=12, fontweight='bold')
        ax.set_xlabel(score)
        ax.set_ylabel('Frecuencia')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Guardar figura y cerrar
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_reviews_distribuciones.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_dir}/07_reviews_distribuciones.png")
    plt.close()
    
    # ========================================
    # GRÁFICO 2: Correlación entre Features de Reviews
    # ========================================
    print("\n[2/5] Generando matriz de correlación...")

    # Seleccionar solo columnas numéricas relevantes que contienen scores o porcentajes
    # Filtrar por nombres que contengan 'score', 'promedio', o 'pct' (percentage)
    scores_cols = [col for col in df.columns if 'score' in col or 'promedio' in col or 'pct' in col]
    # Asegurar que solo incluimos columnas numéricas (float64, int64)
    scores_cols = [col for col in scores_cols if df[col].dtype in ['float64', 'int64']]

    # Si hay al menos 2 variables, calcular y visualizar matriz de correlación
    if len(scores_cols) > 0:
        # Calcular matriz de correlación (coeficiente de Pearson)
        # Mide la relación lineal entre pares de variables (-1 a +1)
        corr_matrix = df[scores_cols].corr()

        # Crear figura grande para el heatmap
        plt.figure(figsize=(14, 12))
        # Visualizar matriz con colores (rojo=correlación negativa, azul=positiva)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={'label': 'Correlación'})
        plt.title('Correlación entre Features de Reviews', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/08_reviews_correlaciones.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Guardado: {output_dir}/08_reviews_correlaciones.png")
        plt.close()
    
    # ========================================
    # GRÁFICO 3: Relación Abandono vs Otras Features
    # ========================================
    print("\n[3/5] Generando scatter plots...")

    # Crear figura con 2 filas × 2 columnas para analizar relaciones clave
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # === 1. Abandono vs Engagement ===
    # Pregunta: ¿Los libros más engaging tienen menos menciones de abandono?
    ax1 = axes[0, 0]
    # Scatter plot con color basado en complejidad (tercera variable)
    scatter1 = ax1.scatter(df['engagement_score'], df['abandono_score'],
                          alpha=0.5, s=30, c=df['complejidad_score'], cmap='viridis')
    ax1.set_xlabel('Engagement Score')
    ax1.set_ylabel('Abandono Score')
    ax1.set_title('Abandono vs Engagement (color = complejidad)')
    # Agregar barra de color para mostrar escala de complejidad
    plt.colorbar(scatter1, ax=ax1, label='Complejidad Score')
    ax1.grid(True, alpha=0.3)

    # === 2. Abandono vs Complejidad ===
    # Pregunta: ¿Los libros más complejos tienen más menciones de abandono?
    ax2 = axes[0, 1]
    # Scatter plot con color basado en engagement (tercera variable)
    scatter2 = ax2.scatter(df['complejidad_score'], df['abandono_score'],
                          alpha=0.5, s=30, c=df['engagement_score'], cmap='plasma')
    ax2.set_xlabel('Complejidad Score')
    ax2.set_ylabel('Abandono Score')
    ax2.set_title('Abandono vs Complejidad (color = engagement)')
    plt.colorbar(scatter2, ax=ax2, label='Engagement Score')
    ax2.grid(True, alpha=0.3)

    # === 3. Engagement vs Ritmo ===
    # Pregunta: ¿El ritmo narrativo afecta el engagement?
    ax3 = axes[1, 0]
    ax3.scatter(df['ritmo_score'], df['engagement_score'], alpha=0.5, s=30, color='coral')
    ax3.set_xlabel('Ritmo Score (negativo=lento, positivo=rápido)')
    ax3.set_ylabel('Engagement Score')
    ax3.set_title('Engagement vs Ritmo')
    # Agregar líneas de referencia en (0,0) para interpretación de cuadrantes
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)

    # === 4. Complejidad vs Sentimiento ===
    # Pregunta: ¿Los libros complejos generan reacciones sentimentales diferentes?
    ax4 = axes[1, 1]
    if 'sentimiento_promedio' in df.columns:
        ax4.scatter(df['complejidad_score'], df['sentimiento_promedio'], alpha=0.5, s=30, color='teal')
        ax4.set_xlabel('Complejidad Score')
        ax4.set_ylabel('Sentimiento Promedio')
        ax4.set_title('Complejidad vs Sentimiento')
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_reviews_scatter_plots.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_dir}/09_reviews_scatter_plots.png")
    plt.close()
    
    # ========================================
    # GRÁFICO 4: Categorización de Libros
    # ========================================
    print("\n[4/5] Generando categorización de libros...")

    # Crear categorías de abandono basadas en cortes de valores (binning)
    # Bajo: 0-5%, Medio: 5-15%, Alto: >15% de menciones de abandono
    df['categoria_abandono'] = pd.cut(df['abandono_score'],
                                      bins=[0, 0.05, 0.15, 1.0],
                                      labels=['Bajo', 'Medio', 'Alto'])

    # Crear categorías de engagement: distribuidas alrededor de cero
    # Bajo: <0, Medio: 0-0.5, Alto: >0.5
    df['categoria_engagement'] = pd.cut(df['engagement_score'],
                                        bins=[-1.0, 0, 0.5, 10.0],
                                        labels=['Bajo', 'Medio', 'Alto'])

    # Crear categorías de complejidad: distribuidas alrededor de cero
    # Simple: <-0.1, Medio: -0.1 a 0.1, Complejo: >0.1
    df['categoria_complejidad'] = pd.cut(df['complejidad_score'],
                                         bins=[-1.0, -0.1, 0.1, 10.0],
                                         labels=['Simple', 'Medio', 'Complejo'])

    # Crear figura con 1 fila × 3 columnas para mostrar distribuciones categóricas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # === Gráfico 1: Distribución por Categoría de Abandono ===
    ax1 = axes[0]
    # Contar libros en cada categoría y visualizar como barras
    df['categoria_abandono'].value_counts().plot(kind='bar', ax=ax1, color=['green', 'orange', 'red'])
    ax1.set_title('Libros por Nivel de Abandono Mencionado', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Categoría')
    ax1.set_ylabel('Número de Libros')
    ax1.tick_params(axis='x', rotation=0)

    # === Gráfico 2: Distribución por Engagement ===
    ax2 = axes[1]
    # Rojo=bajo engagement, naranja=medio, verde=alto engagement
    df['categoria_engagement'].value_counts().plot(kind='bar', ax=ax2, color=['red', 'orange', 'green'])
    ax2.set_title('Libros por Nivel de Engagement', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Categoría')
    ax2.set_ylabel('Número de Libros')
    ax2.tick_params(axis='x', rotation=0)

    # === Gráfico 3: Distribución por Complejidad ===
    ax3 = axes[2]
    # Verde=simple, naranja=medio, rojo=complejo
    df['categoria_complejidad'].value_counts().plot(kind='bar', ax=ax3, color=['green', 'orange', 'red'])
    ax3.set_title('Libros por Nivel de Complejidad', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Categoría')
    ax3.set_ylabel('Número de Libros')
    ax3.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_reviews_categorizacion.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_dir}/10_reviews_categorizacion.png")
    plt.close()
    
    # ========================================
    # GRÁFICO 5: Top Libros por Features
    # ========================================
    print("\n[5/5] Generando top libros...")

    # Crear figura con 2 filas × 2 columnas para mostrar 4 rankings diferentes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # === 1. Top 20 libros con más menciones de abandono ===
    # Estos son los libros que más frecuentemente se mencionan como abandonados
    ax1 = axes[0, 0]
    top_abandono = df.nlargest(20, 'menciones_abandono')[['book_id', 'menciones_abandono']]
    top_abandono.plot(x='book_id', y='menciones_abandono', kind='barh', ax=ax1, color='red', legend=False)
    ax1.set_title('Top 20 Libros con Más Menciones de Abandono', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Menciones de Abandono')
    ax1.set_ylabel('Book ID')

    # === 2. Top 20 libros más engaging ===
    # Estos son los libros que más frecuentemente se describedn como adictivos, atrapantes
    ax2 = axes[0, 1]
    top_engaging = df.nlargest(20, 'menciones_engagement_positivo')[['book_id', 'menciones_engagement_positivo']]
    top_engaging.plot(x='book_id', y='menciones_engagement_positivo', kind='barh', ax=ax2, color='green', legend=False)
    ax2.set_title('Top 20 Libros Más Engaging', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Menciones de Engagement')
    ax2.set_ylabel('Book ID')

    # === 3. Top 20 libros más complejos ===
    # Estos son los libros que más frecuentemente se describen como densos, complicados
    ax3 = axes[1, 0]
    top_complejo = df.nlargest(20, 'menciones_complejidad')[['book_id', 'menciones_complejidad']]
    top_complejo.plot(x='book_id', y='menciones_complejidad', kind='barh', ax=ax3, color='orange', legend=False)
    ax3.set_title('Top 20 Libros Más Complejos', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Menciones de Complejidad')
    ax3.set_ylabel('Book ID')

    # === 4. Top 20 con ritmo más lento ===
    # Estos son los libros que más frecuentemente se describen como lentos en su ritmo narrativo
    ax4 = axes[1, 1]
    top_lento = df.nlargest(20, 'menciones_ritmo_lento')[['book_id', 'menciones_ritmo_lento']]
    top_lento.plot(x='book_id', y='menciones_ritmo_lento', kind='barh', ax=ax4, color='purple', legend=False)
    ax4.set_title('Top 20 Libros con Ritmo Más Lento', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Menciones de Ritmo Lento')
    ax4.set_ylabel('Book ID')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/11_reviews_top_libros.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_dir}/11_reviews_top_libros.png")
    plt.close()
    
    # ========================================
    # REPORTE ESTADÍSTICO
    # ========================================
    print("\n" + "="*70)
    print("ESTADÍSTICAS DE FEATURES DE REVIEWS")
    print("="*70)

    # === ABANDONO SCORE ===
    print("\n📊 Scores Principales:")
    print(f"\nAbandono Score (frecuencia de menciones de abandono en reviews):")
    print(f"  Media: {df['abandono_score'].mean():.4f}")
    print(f"  Std: {df['abandono_score'].std():.4f}")
    print(f"  Min: {df['abandono_score'].min():.4f}")
    print(f"  Max: {df['abandono_score'].max():.4f}")
    # Contar y porcentaje de libros con alto abandono
    libros_alto_abandono = (df['abandono_score'] > 0.10).sum()
    pct_alto_abandono = libros_alto_abandono / len(df) * 100
    print(f"  Libros con alta mención (>0.10): {libros_alto_abandono} ({pct_alto_abandono:.1f}%)")

    # === ENGAGEMENT SCORE ===
    print(f"\nEngagement Score (frecuencia de menciones positivas de engagement):")
    print(f"  Media: {df['engagement_score'].mean():.4f}")
    print(f"  Std: {df['engagement_score'].std():.4f}")
    # Contar y porcentaje de libros muy engaging
    libros_muy_engaging = (df['engagement_score'] > 0.50).sum()
    pct_muy_engaging = libros_muy_engaging / len(df) * 100
    print(f"  Libros muy engaging (>0.50): {libros_muy_engaging} ({pct_muy_engaging:.1f}%)")

    # === COMPLEJIDAD SCORE ===
    print(f"\nComplejidad Score (balance complejidad alta vs lectura fácil):")
    print(f"  Media: {df['complejidad_score'].mean():.4f}")
    print(f"  Std: {df['complejidad_score'].std():.4f}")
    # Contar libros complejos y simples
    libros_complejos = (df['complejidad_score'] > 0.30).sum()
    pct_complejos = libros_complejos / len(df) * 100
    libros_simples = (df['complejidad_score'] < -0.30).sum()
    pct_simples = libros_simples / len(df) * 100
    print(f"  Libros complejos (>0.30): {libros_complejos} ({pct_complejos:.1f}%)")
    print(f"  Libros simples (<-0.30): {libros_simples} ({pct_simples:.1f}%)")

    print(f"\n🎯 Insights Clave:")

    # === Correlación Abandono-Engagement ===
    # Calcular correlación de Pearson entre abandono y engagement
    corr_aband_eng = df[['abandono_score', 'engagement_score']].corr().iloc[0, 1]
    print(f"  • Correlación Abandono-Engagement: {corr_aband_eng:.3f}")
    # Interpretar correlación
    if corr_aband_eng < -0.3:
        print("    → Libros engaging tienen menos menciones de abandono ✓")

    # === Correlación Abandono-Complejidad ===
    # Calcular correlación de Pearson entre abandono y complejidad
    corr_aband_comp = df[['abandono_score', 'complejidad_score']].corr().iloc[0, 1]
    print(f"  • Correlación Abandono-Complejidad: {corr_aband_comp:.3f}")
    # Interpretar correlación
    if corr_aband_comp > 0.2:
        print("    → Libros complejos tienen más menciones de abandono ✓")

    print("\n" + "="*70)
    print("✅ VISUALIZACIONES COMPLETADAS")
    print("="*70)
    print(f"\nGráficos generados en: {output_dir}/")
    print("  • 07_reviews_distribuciones.png")
    print("  • 08_reviews_correlaciones.png")
    print("  • 09_reviews_scatter_plots.png")
    print("  • 10_reviews_categorizacion.png")
    print("  • 11_reviews_top_libros.png")


if __name__ == "__main__":
    # Permite ejecutar desde línea de comandos con parámetros opcionales
    import sys

    # Si se proporciona un argumento, usarlo como nombre del archivo de features
    # Ejemplo: python 06_visualizar_reviews.py features_reviews.csv
    if len(sys.argv) > 1:
        # El primer argumento (índice 1) es el archivo de features
        features_file = sys.argv[1]
    else:
        # Por defecto, usar features_reviews.csv
        features_file = 'features_reviews.csv'

    try:
        # Llamar función de visualización con el archivo especificado
        visualizar_features_reviews(features_file)
    except FileNotFoundError:
        # Manejar error si el archivo no existe
        print(f"\n❌ Error: No se encuentra {features_file}")
        print("\nPrimero ejecuta: python 01b_analizar_reviews.py")