"""
PIPELINE DE PREPROCESAMIENTO - PARTE 3: TRANSFORMACIONES
=========================================================
Implementa técnicas de transformación vistas en Lecture04

Incluye:
- Normalización y estandarización
- Transformaciones logarítmicas y power transform
- Encoding de variables categóricas
- Feature Engineering (creación de nuevas variables)
- Binning y discretización

Autor: Isabella Idarraga
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   PowerTransformer, QuantileTransformer,
                                   LabelEncoder, OneHotEncoder)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class TransformadorDatos:
    """
    Clase para aplicar transformaciones y crear features
    """

    def __init__(self, verbose=True):
        # Flag para imprimir mensajes de progreso durante transformaciones
        self.verbose = verbose
        # Diccionario para almacenar objetos scalers (normalizadores)
        # Útil para aplicar las mismas transformaciones en datos nuevos
        self.scalers = {}
        # Diccionario para almacenar objetos encoders (codificadores categóricos)
        # Permite replicar el encoding en nuevos datos
        self.encoders = {}
        # Lista para rastrear todas las features creadas durante el feature engineering
        self.features_creadas = []
    
    def log(self, mensaje):
        # Método auxiliar para imprimir mensajes solo si verbose=True
        if self.verbose:
            print(mensaje)
    
    # ==========================================
    # 1. NORMALIZACIÓN Y ESTANDARIZACIÓN
    # ==========================================
    
    def estandarizar_variables(self, df, columnas=None, metodo='standard'):
        """
        Aplica estandarización a variables numéricas
        
        Métodos:
        - 'standard': Z-score standardization (media=0, std=1)
        - 'minmax': Min-Max scaling (rango [0,1])
        - 'robust': Robust scaling (usa mediana y IQR, resistente a outliers)
        """
        self.log(f"\n[1.1] Aplicando estandarización (método: {metodo})...")
        
        if columnas is None:
            columnas = df.select_dtypes(include=[np.number]).columns.tolist()
            # Excluir IDs (columnas con 'id' en el nombre) y variables binarias (abandono, es_ultima)
            columnas = [c for c in columnas if not any(x in c.lower() for x in ['id', 'abandono', 'es_ultima'])]

        # Crear copia del dataframe para no modificar el original
        df_transformed = df.copy()

        # Aplicar transformación a cada columna
        for col in columnas:
            # Validar que la columna existe
            if col not in df.columns:
                continue

            # Seleccionar el escalador según el método especificado
            # StandardScaler: normaliza con media=0 y desviación estándar=1, sensible a outliers
            if metodo == 'standard':
                scaler = StandardScaler()
            # MinMaxScaler: escala al rango [0,1], preserva forma de distribución
            elif metodo == 'minmax':
                scaler = MinMaxScaler()
            # RobustScaler: usa mediana y rango intercuartílico, robusto a outliers
            elif metodo == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Método desconocido: {metodo}")

            # Extraer valores y aplicar transformación
            values = df[[col]].values
            transformed = scaler.fit_transform(values)
            # Crear nueva columna con valores estandarizados
            df_transformed[f'{col}_scaled'] = transformed

            # Guardar scaler para aplicarlo a datos nuevos en el futuro (producción)
            self.scalers[col] = scaler
        
        self.log(f"   ✓ Estandarizadas {len(columnas)} variables")
        
        return df_transformed
    
    # ==========================================
    # 2. TRANSFORMACIONES PARA NORMALIDAD
    # ==========================================
    
    def aplicar_transformacion_log(self, df, columnas):
        """
        Aplica transformación logarítmica para reducir asimetría
        Útil para variables con distribución muy sesgada
        """
        self.log("\n[2.1] Aplicando transformación logarítmica...")

        # Crear copia del dataframe
        df_transformed = df.copy()

        # Aplicar transformación log a cada columna
        for col in columnas:
            # Validar que la columna existe
            if col not in df.columns:
                continue

            # Evitar log(0) sumando una constante pequeña si hay valores <= 0
            min_val = df[col].min()
            if min_val <= 0:
                # Si hay valores negativos o cero, desplazar el mínimo a valores positivos
                shift = abs(min_val) + 1
            else:
                # Si todos son positivos, no es necesario desplazar
                shift = 0

            # np.log1p = log(1 + x), evita problemas con log(0)
            # Aplicar a la columna desplazada si es necesario
            df_transformed[f'{col}_log'] = np.log1p(df[col] + shift)
            self.log(f"   ✓ {col} → {col}_log")

        return df_transformed
    
    def aplicar_power_transform(self, df, columnas, metodo='yeo-johnson'):
        """
        Aplica transformaciones de potencia para hacer datos más gaussianos
        
        Métodos:
        - 'yeo-johnson': Permite valores negativos
        - 'box-cox': Solo valores positivos
        """
        self.log(f"\n[2.2] Aplicando Power Transform ({metodo})...")

        # Crear copia del dataframe
        df_transformed = df.copy()
        # Instancia del transformador de potencia
        pt = PowerTransformer(method=metodo, standardize=False)

        # Aplicar transformación Power a cada columna
        for col in columnas:
            # Validar que la columna existe
            if col not in df.columns:
                continue

            # Extraer valores de la columna
            values = df[[col]].values

            # Box-Cox requiere que todos los valores sean estrictamente positivos (> 0)
            # Si hay valores <= 0, usar Yeo-Johnson en su lugar (más flexible)
            if metodo == 'box-cox' and (values <= 0).any():
                self.log(f"   ⚠️  {col} tiene valores ≤0, usando Yeo-Johnson")
                pt_temp = PowerTransformer(method='yeo-johnson', standardize=False)
                transformed = pt_temp.fit_transform(values)
            else:
                # Aplicar el transformador especificado
                transformed = pt.fit_transform(values)

            # Crear nueva columna con valores transformados por potencia
            df_transformed[f'{col}_power'] = transformed
            self.log(f"   ✓ {col} → {col}_power")

        return df_transformed
    
    # ==========================================
    # 3. ENCODING DE VARIABLES CATEGÓRICAS
    # ==========================================
    
    def label_encoding(self, df, columnas):
        """
        Label Encoding para variables categóricas ordinales
        """
        self.log("\n[3.1] Aplicando Label Encoding...")

        # Crear copia del dataframe
        df_encoded = df.copy()

        # Aplicar Label Encoding a cada columna categórica
        for col in columnas:
            # Validar que la columna existe
            if col not in df.columns:
                continue

            # Crear instancia del codificador de etiquetas
            # Label Encoding asigna enteros 0, 1, 2... a cada categoría única
            le = LabelEncoder()
            # Convertir a string para manejar valores faltantes o tipos mixtos
            # fit_transform: ajusta el encoder y aplica la transformación en un paso
            df_encoded[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            # Guardar el encoder para usarlo en datos nuevos (predicción)
            self.encoders[col] = le

            # Reportar el número de categorías encontradas
            self.log(f"   ✓ {col} → {col}_encoded ({len(le.classes_)} categorías)")

        return df_encoded
    
    def one_hot_encoding(self, df, columnas, max_categories=10):
        """
        One-Hot Encoding para variables categóricas nominales
        """
        self.log("\n[3.2] Aplicando One-Hot Encoding...")

        # Crear copia del dataframe
        df_encoded = df.copy()

        # Aplicar One-Hot Encoding a cada columna categórica
        for col in columnas:
            # Validar que la columna existe
            if col not in df.columns:
                continue

            # Contar categorías únicas
            value_counts = df[col].value_counts()
            # Si hay muchas categorías, limitar a las más frecuentes para evitar dimensionalidad muy alta
            if len(value_counts) > max_categories:
                # Seleccionar las top N categorías más frecuentes
                top_categories = value_counts.head(max_categories).index
                # Crear copia para modificar
                df_temp = df[col].copy()
                # Agrupar categorías menos frecuentes como 'otros'
                df_temp[~df_temp.isin(top_categories)] = 'otros'
            else:
                # Si no hay muchas categorías, usar todas
                df_temp = df[col]

            # pd.get_dummies: crear columnas binarias (0/1) para cada categoría
            # prefix: añadir nombre de la columna original como prefijo a los nombres nuevos
            # drop_first: eliminar la primera categoría para evitar multicolinealidad perfecta
            dummies = pd.get_dummies(df_temp, prefix=col, drop_first=True)
            # Concatenar las nuevas columnas dummy al dataframe
            df_encoded = pd.concat([df_encoded, dummies], axis=1)

            # Reportar cuántas columnas dummy se crearon
            self.log(f"   ✓ {col} → {len(dummies.columns)} columnas dummy")

        return df_encoded
    
    # ==========================================
    # 4. FEATURE ENGINEERING
    # ==========================================
    
    def crear_features_usuario(self, df):
        """
        Crea features agregadas a nivel de usuario
        """
        self.log("\n[4.1] Creando features de usuario...")

        # Agrupar datos por usuario y calcular estadísticas agregadas
        # Estas features capturan el comportamiento y patrones de lectura de cada usuario
        usuario_features = df.groupby('user_id').agg({
            'book_id': 'nunique',  # Cantidad de libros únicos leídos por usuario
            'duration_minutes': ['mean', 'median', 'std'],  # Duración: promedio, mediana, desviación
            'pages_read': ['mean', 'sum'],  # Páginas: promedio por sesión y total
            'completion_pct_end': 'mean',  # Progreso promedio en todos sus libros
            'abandono': 'mean'  # Tasa de abandono del usuario (proporción de libros abandonados)
        }).reset_index()

        # Aplanar los nombres de columnas (eliminar estructuras anidadas creadas por groupby)
        # Esto convierte nombres como ('duration_minutes', 'mean') a 'duracion_promedio'
        usuario_features.columns = ['user_id',
                                    'num_libros_leidos',
                                    'duracion_promedio',
                                    'duracion_mediana',
                                    'duracion_std',
                                    'paginas_promedio',
                                    'paginas_totales',
                                    'progreso_promedio',
                                    'tasa_abandono']

        # Unir las features del usuario al dataframe original
        # Ahora cada fila tendrá información agregada de su usuario
        df_features = df.merge(usuario_features, on='user_id', how='left', suffixes=('', '_usuario'))

        # Añadir los nombres de features creadas a la lista de seguimiento
        self.features_creadas.extend([
            'num_libros_leidos', 'duracion_promedio', 'duracion_mediana',
            'duracion_std', 'paginas_promedio', 'paginas_totales',
            'progreso_promedio', 'tasa_abandono'
        ])

        self.log(f"   ✓ Creadas {len(self.features_creadas)} features de usuario")

        return df_features
    
    def crear_features_libro(self, df):
        """
        Crea features agregadas a nivel de libro
        """
        self.log("\n[4.2] Creando features de libro...")

        # Agrupar datos por libro y calcular estadísticas agregadas
        # Estas features capturan las características de cada libro basadas en cómo los leen los usuarios
        libro_features = df.groupby('book_id').agg({
            'user_id': 'nunique',  # Cantidad de lectores únicos del libro
            'duration_minutes': 'mean',  # Duración promedio de lectura del libro
            'pages_read': 'mean',  # Páginas leídas promedio por sesión
            'completion_pct_end': 'mean',  # Progreso promedio alcanzado en el libro
            'abandono': 'mean'  # Tasa de abandono del libro (proporción de usuarios que abandonan)
        }).reset_index()

        # Renombrar columnas untuk mejor legibilidad y distinguir de features de usuario
        libro_features.columns = ['book_id',
                                  'num_lectores',
                                  'duracion_promedio_libro',
                                  'paginas_promedio_libro',
                                  'progreso_promedio_libro',
                                  'tasa_abandono_libro']

        # Unir las features del libro al dataframe original
        # Ahora cada fila tendrá información agregada del libro que están leyendo
        df_features = df.merge(libro_features, on='book_id', how='left', suffixes=('', '_libro'))

        # Features creadas a nivel de libro
        features_libro = ['num_lectores', 'duracion_promedio_libro',
                         'paginas_promedio_libro', 'progreso_promedio_libro',
                         'tasa_abandono_libro']
        self.features_creadas.extend(features_libro)

        self.log(f"   ✓ Creadas {len(features_libro)} features de libro")

        return df_features
    
    def crear_features_temporales(self, df):
        """
        Crea features basadas en patrones temporales
        """
        self.log("\n[4.3] Creando features temporales...")

        # Crear copia del dataframe
        df_features = df.copy()

        # Asegurar que las columnas de timestamp están en formato datetime
        # Esto permite extraer componentes temporales (hora, día, mes, etc)
        df_features['session_start'] = pd.to_datetime(df_features['session_start'])
        df_features['session_end'] = pd.to_datetime(df_features['session_end'])

        # Extraer componentes temporales de session_start
        # Hora: 0-23 (a qué hora del día ocurrió la sesión)
        df_features['hora'] = df_features['session_start'].dt.hour
        # Día de semana: 0=Lunes, 6=Domingo
        df_features['dia_semana'] = df_features['session_start'].dt.dayofweek
        # Variable binaria: 1 si es fin de semana (viernes-domingo), 0 si es entre semana
        df_features['es_fin_semana'] = (df_features['dia_semana'] >= 5).astype(int)
        # Mes: 1-12
        df_features['mes'] = df_features['session_start'].dt.month

        # Categorizar horas en periodos del día (útil para análisis de patrones)
        # madrugada (0-6), mañana (6-12), tarde (12-18), noche (18-24)
        df_features['periodo_dia'] = pd.cut(df_features['hora'],
                                             bins=[0, 6, 12, 18, 24],
                                             labels=['madrugada', 'mañana', 'tarde', 'noche'],
                                             include_lowest=True)

        # Features temporales creadas
        features_temp = ['hora', 'dia_semana', 'es_fin_semana', 'mes', 'periodo_dia']
        self.features_creadas.extend(features_temp)

        self.log(f"   ✓ Creadas {len(features_temp)} features temporales")

        return df_features
    
    def crear_features_interaccion(self, df):
        """
        Crea features de interacción entre variables
        """
        self.log("\n[4.4] Creando features de interacción...")

        # Crear copia del dataframe
        df_features = df.copy()

        # Velocidad de lectura: páginas leídas por minuto
        # Indica cuán rápido lee el usuario en esta sesión
        # +1 en denominador para evitar división por cero
        df_features['velocidad_lectura'] = df_features['pages_read'] / (df_features['duration_minutes'] + 1)

        # Ratio de progreso: progreso relativo en esta sesión
        # Mide cuánto avanzó en esta sesión comparado con dónde empezó
        # +1 en denominador para evitar división por cero cuando empieza en 0
        df_features['ratio_progreso'] = (
            df_features['completion_pct_end'] - df_features['completion_pct_start']
        ) / (df_features['completion_pct_start'] + 1)

        # Calcular número de sesiones por usuario-libro
        # Agrupa todas las sesiones del mismo usuario con el mismo libro
        sesiones_por_libro = df.groupby(['user_id', 'book_id']).size().reset_index(name='num_sesiones')
        # Unir al dataframe principal para tener el conteo de sesiones en cada fila
        df_features = df_features.merge(sesiones_por_libro, on=['user_id', 'book_id'], how='left')

        # Densidad de lectura: páginas leídas por sesión promedio
        # Indica la intensidad de lectura (páginas por sesión)
        df_features['densidad_lectura'] = df_features['pages_read'] / (df_features['num_sesiones'] + 1)

        # Features de interacción creadas
        features_inter = ['velocidad_lectura', 'ratio_progreso', 'num_sesiones', 'densidad_lectura']
        self.features_creadas.extend(features_inter)

        self.log(f"   ✓ Creadas {len(features_inter)} features de interacción")

        return df_features
    
    # ==========================================
    # 5. BINNING Y DISCRETIZACIÓN
    # ==========================================
    
    def crear_bins(self, df, columna, n_bins=5, estrategia='quantile'):
        """
        Discretiza variables continuas en bins

        Estrategias:
        - 'uniform': bins de igual ancho
        - 'quantile': bins con igual número de observaciones
        - 'kmeans': bins usando K-means clustering
        """
        self.log(f"\n[5.1] Creando bins para {columna} (estrategia: {estrategia})...")

        # Crear copia del dataframe
        df_binned = df.copy()

        # Aplicar estrategia de binning según la especificada
        if estrategia == 'uniform':
            # pd.cut: divide el rango en n_bins intervalos de igual ancho
            # Ejemplo: si rango es 0-100 y n_bins=5, intervalos de 20
            df_binned[f'{columna}_binned'] = pd.cut(df[columna], bins=n_bins)
        elif estrategia == 'quantile':
            # pd.qcut: divide en n_bins intervalos con igual número de observaciones
            # Ejemplo: si hay 1000 valores y n_bins=5, cada bin tiene ~200 valores
            # duplicates='drop': elimina bins duplicados si hay valores iguales
            df_binned[f'{columna}_binned'] = pd.qcut(df[columna], q=n_bins, duplicates='drop')
        else:
            # K-means binning: agrupa los valores usando K-means clustering
            from sklearn.preprocessing import KBinsDiscretizer
            # encode='ordinal': retorna enteros 0, 1, 2... para cada bin
            # strategy='kmeans': usa K-means clustering para determinar los límites
            kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
            df_binned[f'{columna}_binned'] = kbd.fit_transform(df[[columna]])

        self.log(f"   ✓ Creados {n_bins} bins para {columna}")

        return df_binned
    
    # ==========================================
    # 6. SELECCIÓN DE FEATURES
    # ==========================================
    
    def seleccionar_features_importantes(self, df, target_col='abandono', k=20):
        """
        Selecciona las K features más importantes usando mutual information
        """
        self.log(f"\n[6.1] Seleccionando top {k} features más importantes...")

        # Preparar datos para selección de features
        # Filtrar solo la última sesión de cada usuario-libro para evitar información duplicada
        df_target = df[df['es_ultima_sesion'] == 1].copy()

        # Seleccionar solo columns numéricas (los modelos requieren datos numéricos)
        numeric_cols = df_target.select_dtypes(include=[np.number]).columns.tolist()

        # Excluir columnas que no son features útiles para predicción:
        # - target_col: la variable a predecir (no es predictor)
        # - user_id, book_id: identificadores, no patrones (data leakage)
        # - es_ultima_sesion, ultima_sesion, progreso_maximo, dias_inactividad: variables de control/target engineering
        exclude = [target_col, 'user_id', 'book_id', 'es_ultima_sesion',
                  'ultima_sesion', 'progreso_maximo', 'dias_inactividad']
        feature_cols = [c for c in numeric_cols if c not in exclude]

        # Eliminar filas con valores faltantes (NaN)
        # SelectKBest no puede manejar valores faltantes
        df_clean = df_target[feature_cols + [target_col]].dropna()

        # Preparar X (features) e y (target)
        X = df_clean[feature_cols]
        y = df_clean[target_col]

        # Mutual Information para clasificación
        # Mide la dependencia entre cada feature y el target
        # Mayor score = feature más informativa para predecir el target
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(feature_cols)))
        # Ajustar el selector a los datos
        selector.fit(X, y)

        # Obtener scores y crear dataframe con resultados
        scores = pd.DataFrame({
            'feature': feature_cols,
            'score': selector.scores_  # Score de mutual information para cada feature
        }).sort_values('score', ascending=False)

        # Mostrar top K features
        self.log("\n   Top features por importancia:")
        for idx, row in scores.head(k).iterrows():
            # idx+1 para que la numeración comience en 1
            self.log(f"   {idx+1}. {row['feature']}: {row['score']:.4f}")

        # Visualizar top features como gráfico de barras
        plt.figure(figsize=(12, 8))
        top_features = scores.head(k)
        # Gráfico horizontal (barh) para mejor legibilidad de nombres largos
        plt.barh(range(len(top_features)), top_features['score'])
        # Usar nombres de features como etiquetas del eje Y
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mutual Information Score')
        plt.title(f'Top {k} Features Más Importantes para Predicción de Abandono')
        plt.tight_layout()
        # Guardar figura
        plt.savefig('graficos_eda/06_feature_importance.png', dpi=300, bbox_inches='tight')
        self.log("\n   ✓ Gráfico guardado: graficos_eda/06_feature_importance.png")
        plt.close()

        return scores
    
    # ==========================================
    # PIPELINE COMPLETO
    # ==========================================
    
    def transformar_completo(self, df):
        """
        Ejecuta el pipeline completo de transformaciones
        """
        self.log("="*70)
        self.log("INICIANDO TRANSFORMACIONES Y FEATURE ENGINEERING")
        self.log("="*70)

        # PASO 1: Feature Engineering primero (crear nuevas variables)
        # Las features se crean antes de las transformaciones para que se apliquen a variables nuevas
        df = self.crear_features_usuario(df)  # Características agregadas a nivel usuario
        df = self.crear_features_libro(df)  # Características agregadas a nivel libro
        df = self.crear_features_temporales(df)  # Componentes temporales (hora, día, etc)
        df = self.crear_features_interaccion(df)  # Interacciones entre variables

        # PASO 2: Transformaciones de normalidad
        # Aplicar transformaciones logarítmicas y de potencia para reducir asimetría
        vars_asimetricas = ['duration_minutes', 'pages_read']
        df = self.aplicar_transformacion_log(df, vars_asimetricas)  # Log transform
        df = self.aplicar_power_transform(df, vars_asimetricas)  # Power transform (Yeo-Johnson)

        # PASO 3: Encoding de variables categóricas
        # Convertir variables categóricas a numéricas usando Label Encoding
        if 'periodo_dia' in df.columns:
            df = self.label_encoding(df, ['periodo_dia'])

        # PASO 4: Estandarización (normalización)
        # Escalamiento de variables numéricas a media=0, std=1
        # Seleccionar variables numéricas originales y features creadas que necesitan escalamiento
        vars_estandarizar = [
            'duration_minutes', 'pages_read', 'completion_pct_start',
            'completion_pct_end', 'velocidad_lectura', 'ratio_progreso',
            'duracion_promedio', 'paginas_promedio', 'num_sesiones'
        ]
        # Filtrar solo las que existen en el dataframe
        vars_estandarizar = [v for v in vars_estandarizar if v in df.columns]
        df = self.estandarizar_variables(df, vars_estandarizar, metodo='standard')

        # PASO 5: Binning/Discretización de variables clave
        # Convertir variables continuas en categorías discretas
        if 'duration_minutes' in df.columns:
            df = self.crear_bins(df, 'duration_minutes', n_bins=5, estrategia='quantile')

        # PASO 6: Selección de features
        # Identificar las features más importantes para el modelo
        self.seleccionar_features_importantes(df, target_col='abandono', k=20)

        # Mostrar resumen final de transformaciones
        self.log("\n" + "="*70)
        self.log("TRANSFORMACIONES COMPLETADAS")
        self.log("="*70)
        self.log(f"\nFeatures creadas: {len(self.features_creadas)}")
        self.log(f"Dimensiones finales: {df.shape}")

        return df


def ejecutar_transformaciones(input_file, output_file='datos_transformados.csv'):
    """
    Función principal para ejecutar transformaciones
    """
    # Cargar datos con variable target (output del módulo EDA)
    print("\nCargando datos con target...")
    df = pd.read_csv(input_file)
    print(f"✓ Cargados {len(df)} registros")

    # Crear instancia del transformador de datos
    transformador = TransformadorDatos(verbose=True)
    # Ejecutar el pipeline completo de transformaciones
    df_transformado = transformador.transformar_completo(df)

    # Guardar los datos transformados en un archivo CSV
    df_transformado.to_csv(output_file, index=False)
    print(f"\n✓ Datos transformados guardados en: {output_file}")

    # Guardar lista de features creadas en un archivo de texto
    # Útil para documentación y referencia posterior
    with open('features_creadas.txt', 'w') as f:
        f.write("FEATURES CREADAS\\n")
        f.write("="*70 + "\\n\\n")
        # Escribir cada feature en una línea numerada
        for i, feat in enumerate(transformador.features_creadas, 1):
            f.write(f"{i}. {feat}\\n")

    print("✓ Lista de features guardada en: features_creadas.txt")

    return df_transformado


if __name__ == "__main__":
    # Ejecutar transformaciones con archivos de entrada/salida específicos
    df_transformado = ejecutar_transformaciones(
        # Archivo CSV con datos originales + variable target (output de EDA)
        input_file='datos_con_target.csv',
        # Archivo CSV de salida con datos transformados y features creadas
        output_file='datos_transformados.csv'
    )