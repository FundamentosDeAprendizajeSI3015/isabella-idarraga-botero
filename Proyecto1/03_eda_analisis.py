"""
PIPELINE DE PREPROCESAMIENTO - PARTE 2: EDA (ANÁLISIS EXPLORATORIO)
====================================================================
Implementa técnicas de Exploratory Data Analysis vistas en clase

Incluye:
- Análisis univariado (distribuciones, estadísticas descriptivas)
- Análisis bivariado (correlaciones, relaciones)
- Visualizaciones clave
- Detección de patrones y anomalías

Autor: Isabella Idarraga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
# Usar estilo oscuro con grilla para mejor legibilidad
plt.style.use('seaborn-v0_8-darkgrid')
# Usar paleta de colores husl (perceptualmente uniforme)
sns.set_palette("husl")

class AnalizadorExploratorio:
    """
    Clase para realizar Análisis Exploratorio de Datos (EDA) completo
    """

    def __init__(self, df, output_dir='graficos_eda'):
        # DataFrame con los datos a analizar
        self.df = df
        # Directorio donde guardar los gráficos generados
        self.output_dir = output_dir

        # Crear directorio para gráficos si no existe
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Diccionario para almacenar estadísticas calculadas
        self.estadisticas = {}
    
    # ==========================================
    # 1. ANÁLISIS UNIVARIADO
    # ==========================================
    
    def analisis_descriptivo(self):
        """
        Estadísticas descriptivas de todas las variables numéricas
        """
        print("\n" + "="*70)
        print("ANÁLISIS DESCRIPTIVO")
        print("="*70)

        # Seleccionar solo columnas numéricas
        numericas = self.df.select_dtypes(include=[np.number]).columns

        print("\nEstadísticas descriptivas:")
        # Mostrar count, mean, std, min, 25%, 50%, 75%, max
        print(self.df[numericas].describe())

        # Guardar estadísticas para uso posterior
        self.estadisticas['descriptivas'] = self.df[numericas].describe()

        # Análisis de asimetría y curtosis
        print("\n" + "-"*70)
        print("ASIMETRÍA Y CURTOSIS")
        print("-"*70)

        # Calcular e interpretar asimetría y curtosis para cada variable
        for col in numericas:
            # Calcular asimetría (skweness): mide la simetría de la distribución
            skew = stats.skew(self.df[col].dropna())
            # Calcular curtosis: mide qué tan puntiaguda es la distribución
            kurt = stats.kurtosis(self.df[col].dropna())
            print(f"\n{col}:")
            print(f"  Asimetría: {skew:.3f}", end="")
            # Interpretar asimetría
            if abs(skew) < 0.5:
                print(" (distribución simétrica)")
            elif skew > 0:
                print(" (sesgo positivo - cola derecha más larga)")
            else:
                print(" (sesgo negativo - cola izquierda más larga)")

            print(f"  Curtosis: {kurt:.3f}", end="")
            # Interpretar curtosis
            if abs(kurt) < 0.5:
                print(" (mesocúrtica - similar a normal)")
            elif kurt > 0:
                print(" (leptocúrtica - más puntiaguda que normal)")
            else:
                print(" (platicúrtica - más plana que normal)")
    
    def visualizar_distribuciones(self):
        """
        Visualiza distribuciones de variables clave con histogramas y boxplots
        """
        print("\n" + "="*70)
        print("VISUALIZACIÓN DE DISTRIBUCIONES")
        print("="*70)

        # Variables clave a analizar
        variables = ['duration_minutes', 'pages_read', 'completion_pct_end']

        # Crear figura con 3 filas × 2 columnas (histograma y boxplot para cada variable)
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        for idx, var in enumerate(variables):
            if var not in self.df.columns:
                continue

            # === HISTOGRAMA (columna 0) ===
            ax1 = axes[idx, 0]
            self.df[var].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7)
            ax1.set_title(f'Histograma: {var}')
            ax1.set_xlabel(var)
            ax1.set_ylabel('Frecuencia')
            # Agregar línea vertical para media
            ax1.axvline(self.df[var].mean(), color='red', linestyle='--',
                       label=f'Media: {self.df[var].mean():.2f}')
            # Agregar línea vertical para mediana
            ax1.axvline(self.df[var].median(), color='green', linestyle='--',
                       label=f'Mediana: {self.df[var].median():.2f}')
            ax1.legend()

            # ===BOXPLOT (columna 1) ===
            ax2 = axes[idx, 1]
            self.df.boxplot(column=var, ax=ax2)
            ax2.set_title(f'Boxplot: {var}')
            ax2.set_ylabel(var)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_distribuciones.png', dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {self.output_dir}/01_distribuciones.png")
        plt.close()
    
    def analisis_categoricas(self):
        """
        Análisis de variables categóricas si existen
        """
        # Seleccionar columnas de tipo objeto (categóricas)
        categoricas = self.df.select_dtypes(include=['object']).columns

        if len(categoricas) > 0:
            print("\n" + "="*70)
            print("ANÁLISIS DE VARIABLES CATEGÓRICAS")
            print("="*70)

            # Mostrar conteo de valores únicos para cada variable categórica
            for col in categoricas:
                print(f"\n{col}:")
                print(self.df[col].value_counts())
    
    # ==========================================
    # 2. ANÁLISIS BIVARIADO
    # ==========================================
    
    def analisis_correlaciones(self):
        """
        Análisis de correlaciones entre variables numéricas
        """
        print("\n" + "="*70)
        print("ANÁLISIS DE CORRELACIONES")
        print("="*70)

        # Seleccionar solo columnas numéricas
        numericas = self.df.select_dtypes(include=[np.number]).columns

        # Calcular matriz de correlación (coeficiente de Pearson)
        corr_matrix = self.df[numericas].corr()

        print("\nMatriz de correlación:")
        print(corr_matrix)

        # Encontrar correlaciones significativas
        print("\n" + "-"*70)
        print("CORRELACIONES SIGNIFICATIVAS (|r| > 0.5)")
        print("-"*70)

        # Iterar sobre todas las combinaciones de variables
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                # Mostrar si correlación es fuerte (> 0.5 en valor absoluto)
                if abs(corr_val) > 0.5:
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    print(f"{var1} <-> {var2}: {corr_val:.3f}")

        # Crear heatmap de correlaciones
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1)
        plt.title('Matriz de Correlación', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_correlaciones.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Heatmap guardado: {self.output_dir}/02_correlaciones.png")
        plt.close()

        # Guardar matriz de correlaciones para uso posterior
        self.estadisticas['correlaciones'] = corr_matrix
    
    def scatter_plots_clave(self):
        """
        Scatter plots de relaciones clave para entender comportamientos
        """
        print("\n" + "="*70)
        print("SCATTER PLOTS DE RELACIONES CLAVE")
        print("="*70)

        # Crear figura con 2 filas × 2 columnas para diferentes scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # === 1. Duración vs Páginas leídas ===
        ax1 = axes[0, 0]
        # Scatter plot: cada punto es una sesión
        ax1.scatter(self.df['pages_read'], self.df['duration_minutes'], alpha=0.5)
        ax1.set_xlabel('Páginas leídas')
        ax1.set_ylabel('Duración (minutos)')
        ax1.set_title('Duración vs Páginas Leídas')

        # Agregar línea de tendencia (regresión lineal)
        z = np.polyfit(self.df['pages_read'].dropna(),
                       self.df['duration_minutes'].dropna(), 1)
        p = np.poly1d(z)
        ax1.plot(self.df['pages_read'], p(self.df['pages_read']),
                "r--", alpha=0.8, label=f'Tendencia: y={z[0]:.2f}x+{z[1]:.2f}')
        ax1.legend()

        # === 2. Progreso inicial vs Final ===
        ax2 = axes[0, 1]
        ax2.scatter(self.df['completion_pct_start'], self.df['completion_pct_end'], alpha=0.5)
        ax2.set_xlabel('Progreso inicial (%)')
        ax2.set_ylabel('Progreso final (%)')
        ax2.set_title('Progreso Inicial vs Final')
        # Línea de identidad (donde progress_start == progress_end)
        ax2.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='y=x')
        ax2.legend()

        # === 3. Duración por rangos de progreso ===
        ax3 = axes[1, 0]
        # Crear categor progreso en rangos: 0-25%, 25-50%, 50-75%, 75-100%
        self.df['rango_progreso'] = pd.cut(self.df['completion_pct_end'],
                                            bins=[0, 25, 50, 75, 100],
                                            labels=['0-25%', '25-50%', '50-75%', '75-100%'])
        # Boxplot: comparar duración en cada rango de progreso
        self.df.boxplot(column='duration_minutes', by='rango_progreso', ax=ax3)
        ax3.set_xlabel('Rango de progreso')
        ax3.set_ylabel('Duración (minutos)')
        ax3.set_title('Duración por Rango de Progreso')
        plt.sca(ax3)
        plt.xticks(rotation=45)

        # === 4. Páginas leídas por rangos de duración ===
        ax4 = axes[1, 1]
        # Crear categorías de duración: <30min, 30-60min, 60-120min, >120min
        self.df['rango_duracion'] = pd.cut(self.df['duration_minutes'],
                                            bins=[0, 30, 60, 120, 500],
                                            labels=['<30min', '30-60min', '60-120min', '>120min'])
        # Boxplot: comparar páginas leídas en cada rango de duración
        self.df.boxplot(column='pages_read', by='rango_duracion', ax=ax4)
        ax4.set_xlabel('Rango de duración')
        ax4.set_ylabel('Páginas leídas')
        ax4.set_title('Páginas Leídas por Rango de Duración')
        plt.sca(ax4)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_scatter_plots.png', dpi=300, bbox_inches='tight')
        print(f"✓ Scatter plots guardados: {self.output_dir}/03_scatter_plots.png")
        plt.close()
    
    # ==========================================
    # 3. ANÁLISIS TEMPORAL
    # ==========================================
    
    def analisis_temporal(self):
        """
        Análisis de patrones temporales en las sesiones de lectura
        """
        print("\n" + "="*70)
        print("ANÁLISIS TEMPORAL")
        print("="*70)

        # Convertir columnas de timestamp a datetime
        self.df['session_start'] = pd.to_datetime(self.df['session_start'])
        self.df['session_end'] = pd.to_datetime(self.df['session_end'])

        # Extraer componentes temporales para análisis
        self.df['hora'] = self.df['session_start'].dt.hour  # Hora: 0-23
        self.df['dia_semana'] = self.df['session_start'].dt.dayofweek  # Día: 0=Lun, 6=Dom
        self.df['mes'] = self.df['session_start'].dt.month  # Mes: 1-12

        # Crear figura con 2 filas × 2 columnas
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # === 1. Sesiones por hora del día ===
        ax1 = axes[0, 0]
        # Contar sesiones por cada hora (0-23)
        sesiones_por_hora = self.df['hora'].value_counts().sort_index()
        sesiones_por_hora.plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('Distribución de Sesiones por Hora del Día')
        ax1.set_xlabel('Hora')
        ax1.set_ylabel('Número de Sesiones')

        # === 2. Sesiones por día de la semana ===
        ax2 = axes[0, 1]
        # Mapear números de día a nombres
        dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        sesiones_por_dia = self.df['dia_semana'].value_counts().sort_index()
        # Reemplazar números por nombres de días
        sesiones_por_dia.index = [dias[i] for i in sesiones_por_dia.index]
        sesiones_por_dia.plot(kind='bar', ax=ax2, color='coral')
        ax2.set_title('Distribución de Sesiones por Día de la Semana')
        ax2.set_xlabel('Día')
        ax2.set_ylabel('Número de Sesiones')
        ax2.tick_params(axis='x', rotation=45)

        # === 3. Duración promedio por hora ===
        ax3 = axes[1, 0]
        # Calcular duración promedio para cada hora del día
        duracion_por_hora = self.df.groupby('hora')['duration_minutes'].mean()
        duracion_por_hora.plot(kind='line', ax=ax3, marker='o', color='green')
        ax3.set_title('Duración Promedio por Hora del Día')
        ax3.set_xlabel('Hora')
        ax3.set_ylabel('Duración Promedio (minutos)')
        ax3.grid(True, alpha=0.3)

        # === 4. Heatmap: Hora vs Día de semana ===
        ax4 = axes[1, 1]
        # Crear tabla de frecuencias: día × hora
        heatmap_data = self.df.groupby(['dia_semana', 'hora']).size().unstack(fill_value=0)
        # Reemplazar números de días por nombres
        heatmap_data.index = [dias[i] for i in heatmap_data.index]
        # Visualizar como mapa de calor
        sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Número de Sesiones'})
        ax4.set_title('Mapa de Calor: Día vs Hora')
        ax4.set_xlabel('Hora')
        ax4.set_ylabel('Día de la Semana')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_analisis_temporal.png', dpi=300, bbox_inches='tight')
        print(f"✓ Análisis temporal guardado: {self.output_dir}/04_analisis_temporal.png")
        plt.close()
    
    # ==========================================
    # 4. ANÁLISIS DE ABANDONO
    # ==========================================
    
    def definir_abandono(self):
        """
        Define la variable target de abandono basándose en criterios operativos

        Abandono = 1 si:
        - Progreso < 90% (no completó el libro)
        - Inactividad > 21 días desde última sesión
        """
        print("\n" + "="*70)
        print("DEFINICIÓN DE ABANDONO (VARIABLE TARGET)")
        print("="*70)

        # Calcular la última sesión para cada usuario-libro
        ultimas_sesiones = self.df.groupby(['user_id', 'book_id'])['session_end'].max().reset_index()
        ultimas_sesiones.columns = ['user_id', 'book_id', 'ultima_sesion']

        # Calcular progreso máximo alcanzado para cada usuario-libro
        progreso_max = self.df.groupby(['user_id', 'book_id'])['completion_pct_end'].max().reset_index()
        progreso_max.columns = ['user_id', 'book_id', 'progreso_maximo']

        # Unir información de última sesión y progreso máximo al dataframe principal
        self.df = self.df.merge(ultimas_sesiones, on=['user_id', 'book_id'], how='left')
        self.df = self.df.merge(progreso_max, on=['user_id', 'book_id'], how='left')

        # Calcular días de inactividad desde la última sesión
        now = pd.Timestamp.now()
        self.df['dias_inactividad'] = (now - pd.to_datetime(self.df['ultima_sesion'])).dt.days

        # Definir variable target: abandono = 1 si cumple ambas condiciones
        self.df['abandono'] = (
            (self.df['progreso_maximo'] < 90) &  # No completó el libro
            (self.df['dias_inactividad'] > 21)   # Más de 21 días sin actividad
        ).astype(int)

        # Marcar solo la última sesión de cada usuario-libro para evitar duplicados
        self.df['es_ultima_sesion'] = (self.df['session_end'] == self.df['ultima_sesion']).astype(int)

        print(f"\nCriterios de abandono:")
        print("  • Progreso < 90% (no completó)")
        print("  • Inactividad > 21 días")

        # Calcular estadísticas de abandono
        abandonados = self.df[self.df['es_ultima_sesion'] == 1]['abandono'].sum()
        completados = (self.df[self.df['es_ultima_sesion'] == 1]['abandono'] == 0).sum()
        total = abandonados + completados

        print(f"\nEstadísticas de la variable target:")
        print(f"  • Libros abandonados: {abandonados} ({(abandonados/total)*100:.1f}%)")
        print(f"  • Libros completados: {completados} ({(completados/total)*100:.1f}%)")
        print(f"  • Total: {total}")

        # Visualizar distribución de abandono
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # === Gráfico 1: Distribución de abandono ===
        ax1 = axes[0]
        abandono_counts = self.df[self.df['es_ultima_sesion'] == 1]['abandono'].value_counts()
        # Verde para completados (0), rojo para abandonados (1)
        abandono_counts.plot(kind='bar', ax=ax1, color=['green', 'red'])
        ax1.set_title('Distribución de Abandono')
        ax1.set_xlabel('Abandono (0=No, 1=Sí)')
        ax1.set_ylabel('Frecuencia')
        ax1.set_xticklabels(['Completado', 'Abandonado'], rotation=0)

        # === Gráfico 2: Progreso máximo por categoría ===
        ax2 = axes[1]
        # Comparar progreso alcanzado en completados vs abandonados
        self.df[self.df['es_ultima_sesion'] == 1].boxplot(column='progreso_maximo',
                                                            by='abandono', ax=ax2)
        ax2.set_title('Progreso Máximo por Categoría de Abandono')
        ax2.set_xlabel('Abandono')
        ax2.set_ylabel('Progreso Máximo (%)')
        ax2.set_xticklabels(['Completado', 'Abandonado'])

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_analisis_abandono.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Análisis de abandono guardado: {self.output_dir}/05_analisis_abandono.png")
        plt.close()
    
    # ==========================================
    # PIPELINE COMPLETO
    # ==========================================
    
    def ejecutar_eda_completo(self):
        """
        Ejecuta el pipeline completo de EDA en el orden correcto
        """
        print("="*70)
        print("EJECUTANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("="*70)

        # Paso 1: Análisis univariado
        self.analisis_descriptivo()
        self.visualizar_distribuciones()
        self.analisis_categoricas()

        # Paso 2: Análisis bivariado
        self.analisis_correlaciones()
        self.scatter_plots_clave()

        # Paso 3: Análisis temporal
        self.analisis_temporal()

        # Paso 4: Definir variable target (abandono)
        self.definir_abandono()

        # Mostrar resumen final
        print("\n" + "="*70)
        print("EDA COMPLETADO")
        print("="*70)
        print(f"\nTodos los gráficos guardados en: {self.output_dir}/")

        return self.df


def ejecutar_eda(input_file, output_file='datos_con_target.csv', output_dir='graficos_eda'):
    """
    Función principal para ejecutar el pipeline EDA
    """
    # Cargar dataset limpio
    print("\nCargando datos limpios...")
    df = pd.read_csv(input_file)
    print(f"✓ Cargados {len(df)} registros")

    # Crear instancia del analizador exploratorio
    analizador = AnalizadorExploratorio(df, output_dir=output_dir)
    # Ejecutar el EDA completo
    df_con_target = analizador.ejecutar_eda_completo()

    # Guardar dataset con la nueva variable target de abandono
    df_con_target.to_csv(output_file, index=False)
    print(f"\n✓ Datos con variable target guardados en: {output_file}")

    return df_con_target


if __name__ == "__main__":
    # ========================================
    # Ejecutar EDA
    # ========================================
    # Archivo con datos limpios
    df_con_target = ejecutar_eda(
        input_file='datos_sesiones_limpios.csv',     # Datos de entrada (limpios)
        output_file='datos_con_target.csv',          # Datos de salida (con variable target)
        output_dir='graficos_eda'                    # Directorio para gráficos
    )