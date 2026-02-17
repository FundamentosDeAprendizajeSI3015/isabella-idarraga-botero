"""
PIPELINE DE PREPROCESAMIENTO - PARTE 1: LIMPIEZA DE DATOS
==========================================================
Implementa las técnicas vistas en Lecture03 - LimpiezaDatos

Incluye:
- Detección y eliminación de duplicados
- Manejo de valores faltantes
- Detección y tratamiento de outliers
- Validación de consistencia temporal
- Detección de datos anómalos

Autor: Isabella Idarraga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LimpiadorDatos:
    """
    Clase para realizar limpieza exhaustiva de datos de sesiones de lectura
    """

    def __init__(self, verbose=True):
        # Flag para controlar si mostrar mensajes de progreso
        self.verbose = verbose
        # Diccionario para registrar las acciones realizadas durante la limpieza
        self.reporte_limpieza = {
            'duplicados_eliminados': 0,      # Número de filas duplicadas eliminadas
            'outliers_detectados': 0,        # Número de outliers detectados y tratados
            'valores_imputados': 0,          # Número de valores faltantes rellenados
            'inconsistencias_corregidas': 0  # Número de inconsistencias corregidas
        }

    def log(self, mensaje):
        """Helper para imprimir mensajes si verbose=True"""
        # Solo imprime si verbose está activado
        if self.verbose:
            print(mensaje)
    
    # ==========================================
    # 1. DETECCIÓN Y ELIMINACIÓN DE DUPLICADOS
    # ==========================================
    
    def detectar_duplicados_exactos(self, df):
        """
        Detecta y elimina filas completamente duplicadas
        """
        self.log("\n[1.1] Detectando duplicados exactos...")

        # Contar filas antes de eliminar
        duplicados_antes = len(df)
        # Eliminar filas que sean idénticas en todas las columnas
        df_limpio = df.drop_duplicates()
        # Calcular cuántas fueron eliminadas
        duplicados_eliminados = duplicados_antes - len(df_limpio)

        # Registrar en el reporte
        self.reporte_limpieza['duplicados_eliminados'] += duplicados_eliminados

        if duplicados_eliminados > 0:
            self.log(f"   ⚠️  Eliminados {duplicados_eliminados} duplicados exactos")
        else:
            self.log("   ✓ No se encontraron duplicados exactos")

        return df_limpio

    def detectar_duplicados_sesiones(self, df):
        """
        Detecta sesiones sospechosamente similares (mismo usuario, libro, timestamps muy cercanos)
        Esto puede indicar errores en la generación de datos o doble registro
        """
        self.log("\n[1.2] Detectando sesiones duplicadas (mismo usuario+libro+tiempo)...")

        # Ordenar por usuario, libro y timestamp
        df = df.sort_values(['user_id', 'book_id', 'session_start'])

        # Lista para almacenar índices de sesiones duplicadas
        # Marcar como duplicado si:
        # - Mismo user_id y book_id
        # - session_start dentro de 5 minutos
        duplicados_sesion = []

        # Agrupar por usuario y libro
        for (user_id, book_id), group in df.groupby(['user_id', 'book_id']):
            # Si hay solo 1 o menos sesiones para este usuario+libro, no hay duplicados
            if len(group) <= 1:
                continue

            # Convertir timestamps a datetime para calcular diferencias
            timestamps = pd.to_datetime(group['session_start'])
            # Comparar cada sesión con la siguiente
            for i in range(len(timestamps) - 1):
                # Calcular diferencia en minutos
                diff = (timestamps.iloc[i+1] - timestamps.iloc[i]).total_seconds() / 60
                # Si diferencia < 5 minutos es probablemente un duplicado
                if diff < 5:
                    duplicados_sesion.append(group.index[i+1])

        if len(duplicados_sesion) > 0:
            self.log(f"   ⚠️  Detectadas {len(duplicados_sesion)} sesiones duplicadas sospechosas")
            # Eliminar las sesiones duplicadas
            df_limpio = df.drop(duplicados_sesion)
            self.reporte_limpieza['duplicados_eliminados'] += len(duplicados_sesion)
        else:
            self.log("   ✓ No se encontraron sesiones duplicadas")
            df_limpio = df

        return df_limpio
    
    # ==========================================
    # 2. DETECCIÓN Y MANEJO DE VALORES FALTANTES
    # ==========================================
    
    def analizar_valores_faltantes(self, df):
        """
        Analiza y reporta valores faltantes en cada columna
        """
        self.log("\n[2.1] Analizando valores faltantes...")

        # Contar valores nulos en cada columna
        missing = df.isnull().sum()
        # Calcular porcentaje de valores faltantes
        missing_pct = (missing / len(df)) * 100

        # Crear DataFrame con resumen de valores faltantes
        missing_df = pd.DataFrame({
            'Columna': missing.index,
            'Faltantes': missing.values,
            'Porcentaje': missing_pct.values
        })
        # Filtrar solo columnas que tienen valores faltantes y ordenar descendente
        missing_df = missing_df[missing_df['Faltantes'] > 0].sort_values('Faltantes', ascending=False)

        if len(missing_df) > 0:
            self.log("\n   Valores faltantes encontrados:")
            for _, row in missing_df.iterrows():
                self.log(f"   • {row['Columna']}: {row['Faltantes']} ({row['Porcentaje']:.2f}%)")
        else:
            self.log("   ✓ No hay valores faltantes")

        return missing_df

    def imputar_valores_faltantes(self, df):
        """
        Estrategia de imputación inteligente según el tipo de variable
        """
        self.log("\n[2.2] Imputando valores faltantes...")

        df_clean = df.copy()
        valores_imputados = 0

        # duration_minutes: calcular desde timestamps si está faltante
        if df_clean['duration_minutes'].isnull().any():
            # Máscara para filas donde duration_minutes es nulo
            mask = df_clean['duration_minutes'].isnull()
            # Calcular duración desde la diferencia de timestamps
            df_clean.loc[mask, 'duration_minutes'] = (
                pd.to_datetime(df_clean.loc[mask, 'session_end']) -
                pd.to_datetime(df_clean.loc[mask, 'session_start'])
            ).dt.total_seconds() / 60
            valores_imputados += mask.sum()
            self.log(f"   ✓ Imputados {mask.sum()} valores de duration_minutes desde timestamps")

        # pages_read: calcular desde progreso si está faltante
        if df_clean['pages_read'].isnull().any():
            # Máscara para filas donde pages_read es nulo
            mask = df_clean['pages_read'].isnull()
            # Calcular páginas leídas como diferencia de progreso
            df_clean.loc[mask, 'pages_read'] = (
                df_clean.loc[mask, 'progress_end'] - df_clean.loc[mask, 'progress_start']
            )
            valores_imputados += mask.sum()
            self.log(f"   ✓ Imputados {mask.sum()} valores de pages_read desde progreso")

        # completion_pct: calcular si está faltante (requeriría num_pages del libro)
        # Por ahora dejamos como está

        self.reporte_limpieza['valores_imputados'] = valores_imputados

        return df_clean
    
    # ==========================================
    # 3. DETECCIÓN Y TRATAMIENTO DE OUTLIERS
    # ==========================================
    
    def detectar_outliers_iqr(self, df, columna, factor=1.5):
        """
        Detecta outliers usando el método IQR (Rango Intercuartílico)
        IQR = Q3 - Q1. Los outliers están fuera de [Q1 - factor*IQR, Q3 + factor*IQR]
        """
        # Calcular primer cuartil (percentil 25)
        Q1 = df[columna].quantile(0.25)
        # Calcular tercer cuartil (percentil 75)
        Q3 = df[columna].quantile(0.75)
        # Calcular rango intercuartílico
        IQR = Q3 - Q1

        # Calcular límites para detectar outliers
        lower_bound = Q1 - factor * IQR  # Límite inferior
        upper_bound = Q3 + factor * IQR  # Límite superior

        # Identificar valores que están fuera de los límites
        outliers = (df[columna] < lower_bound) | (df[columna] > upper_bound)

        return outliers, lower_bound, upper_bound

    def detectar_outliers_zscore(self, df, columna, threshold=3):
        """
        Detecta outliers usando Z-score (desviación estándar)
        Z-score = (valor - media) / std. Outlier si |z| > threshold
        """
        # Calcular Z-scores absolutos para cada valor
        z_scores = np.abs((df[columna] - df[columna].mean()) / df[columna].std())
        # Marcar como outlier si Z-score supera el threshold
        outliers = z_scores > threshold

        return outliers

    def analizar_outliers(self, df):
        """
        Analiza y reporta outliers en variables numéricas clave
        """
        self.log("\n[3.1] Analizando outliers...")

        # Variables numéricas a analizar para detectar outliers
        variables_numericas = ['duration_minutes', 'pages_read', 'completion_pct_end']

        outliers_report = {}

        # Analizar outliers para cada variable
        for var in variables_numericas:
            if var not in df.columns:
                continue

            # Detectar outliers usando IQR
            outliers_iqr, lower, upper = self.detectar_outliers_iqr(df, var)
            # Contar número de outliers
            n_outliers = outliers_iqr.sum()

            # Guardar información del análisis
            outliers_report[var] = {
                'n_outliers': n_outliers,                      # Cantidad de outliers
                'pct_outliers': (n_outliers / len(df)) * 100,  # Porcentaje
                'lower_bound': lower,                          # Límite inferior válido
                'upper_bound': upper                           # Límite superior válido
            }

            # Mostrar reporte
            self.log(f"\n   {var}:")
            self.log(f"   • Outliers detectados (IQR): {n_outliers} ({(n_outliers/len(df))*100:.2f}%)")
            self.log(f"   • Rango válido: [{lower:.2f}, {upper:.2f}]")

        return outliers_report
    
    def tratar_outliers(self, df, metodo='clip'):
        """
        Trata outliers según el método especificado

        Métodos:
        - 'clip': Recortar a los límites del IQR
        - 'remove': Eliminar filas con outliers
        - 'keep': Mantener pero marcar
        """
        self.log(f"\n[3.2] Tratando outliers (método: {metodo})...")

        df_clean = df.copy()
        outliers_tratados = 0

        # Tratar duration_minutes: duración imposible
        # Mín: 1 minuto, Máx: usar IQR (máximo 8 horas)
        outliers_dur, lower_dur, upper_dur = self.detectar_outliers_iqr(df, 'duration_minutes', factor=2.0)

        # Casos claramente imposibles (valores físicamente imposibles)
        # Sesiones de menos de 1 minuto o más de 8 horas
        imposibles_dur = (df_clean['duration_minutes'] < 1) | (df_clean['duration_minutes'] > 480)

        if metodo == 'clip':
            # Recortar outliers a los límites válidos
            df_clean.loc[outliers_dur, 'duration_minutes'] = df_clean.loc[outliers_dur, 'duration_minutes'].clip(
                lower=max(1, lower_dur), upper=min(480, upper_dur)
            )
            outliers_tratados += outliers_dur.sum()
        elif metodo == 'remove':
            # Eliminar filas con outliers
            df_clean = df_clean[~outliers_dur]
            outliers_tratados += outliers_dur.sum()

        # Eliminar siempre los valores imposibles
        n_imposibles = imposibles_dur.sum()
        if n_imposibles > 0:
            df_clean = df_clean[~imposibles_dur]
            self.log(f"   ✓ Eliminadas {n_imposibles} sesiones con duración imposible")
            outliers_tratados += n_imposibles

        # Tratar pages_read: páginas imposibles
        # Máximo 500 páginas por sesión (valor realista)
        imposibles_pag = (df_clean['pages_read'] < 0) | (df_clean['pages_read'] > 500)
        n_imposibles_pag = imposibles_pag.sum()
        if n_imposibles_pag > 0:
            df_clean = df_clean[~imposibles_pag]
            self.log(f"   ✓ Eliminadas {n_imposibles_pag} sesiones con páginas imposibles")
            outliers_tratados += n_imposibles_pag

        # Registrar total de outliers tratados
        self.reporte_limpieza['outliers_detectados'] = outliers_tratados
        self.log(f"   ✓ Total outliers tratados: {outliers_tratados}")

        return df_clean
    
    # ==========================================
    # 4. VALIDACIÓN DE CONSISTENCIA
    # ==========================================
    
    def validar_consistencia_temporal(self, df):
        """
        Valida que los timestamps sean consistentes
        - session_end > session_start
        - duration_minutes coherente con diferencia de timestamps
        """
        self.log("\n[4.1] Validando consistencia temporal...")

        df_clean = df.copy()
        # Convertir strings de timestamp a datetime
        df_clean['session_start'] = pd.to_datetime(df_clean['session_start'])
        df_clean['session_end'] = pd.to_datetime(df_clean['session_end'])

        # Calcular duración real desde timestamps (en minutos)
        duracion_real = (df_clean['session_end'] - df_clean['session_start']).dt.total_seconds() / 60

        # Detectar inconsistencias:
        # - session_end <= session_start (fin antes del inicio)
        # - diferencia > 5 minutos entre duración registrada y calculada
        inconsistencias = (
            (df_clean['session_end'] <= df_clean['session_start']) |  # end antes que start
            (abs(duracion_real - df_clean['duration_minutes']) > 5)  # diferencia >5min
        )

        n_inconsistencias = inconsistencias.sum()

        if n_inconsistencias > 0:
            self.log(f"   ⚠️  Detectadas {n_inconsistencias} inconsistencias temporales")

            # Corregir: usar timestamps como fuente de verdad
            df_clean.loc[inconsistencias, 'duration_minutes'] = duracion_real[inconsistencias]

            # Si end <= start, eliminar (son inválidos)
            invalidos = df_clean['session_end'] <= df_clean['session_start']
            n_invalidos = invalidos.sum()
            if n_invalidos > 0:
                df_clean = df_clean[~invalidos]
                self.log(f"   ✓ Eliminadas {n_invalidos} sesiones con timestamps inválidos")

            self.reporte_limpieza['inconsistencias_corregidas'] = n_inconsistencias
        else:
            self.log("   ✓ No se encontraron inconsistencias temporales")

        return df_clean

    def validar_consistencia_progreso(self, df):
        """
        Valida que el progreso sea lógico
        - progress_end > progress_start
        - pages_read = progress_end - progress_start
        - completion_pct coherente
        """
        self.log("\n[4.2] Validando consistencia de progreso...")

        df_clean = df.copy()
        inconsistencias = 0

        # Validar que progress_end >= progress_start
        progreso_invalido = df_clean['progress_end'] < df_clean['progress_start']
        if progreso_invalido.any():
            n_inv = progreso_invalido.sum()
            self.log(f"   ⚠️  {n_inv} sesiones con progreso negativo")
            # Eliminar sesiones con progreso inválido
            df_clean = df_clean[~progreso_invalido]
            inconsistencias += n_inv

        # Validar que pages_read coincida con la diferencia de progreso
        # Calcular páginas según progreso
        pages_calculadas = df_clean['progress_end'] - df_clean['progress_start']
        # Calcular diferencia con pages_read registrado
        diferencia = abs(df_clean['pages_read'] - pages_calculadas)

        # Si diferencia > 2 páginas, hay inconsistencia (tolerancia de 2 páginas)
        pages_inconsistentes = diferencia > 2
        if pages_inconsistentes.any():
            n_incons = pages_inconsistentes.sum()
            self.log(f"   ⚠️  {n_incons} sesiones con pages_read inconsistente")
            # Corregir usando el progreso como fuente de verdad
            df_clean.loc[pages_inconsistentes, 'pages_read'] = pages_calculadas[pages_inconsistentes]
            inconsistencias += n_incons

        if inconsistencias == 0:
            self.log("   ✓ No se encontraron inconsistencias de progreso")

        # Actualizar reporte
        self.reporte_limpieza['inconsistencias_corregidas'] += inconsistencias

        return df_clean

    def validar_rangos(self, df):
        """
        Valida que los valores estén en rangos esperados
        """
        self.log("\n[4.3] Validando rangos de valores...")

        df_clean = df.copy()
        fuera_rango = 0

        # Validar que completion_pct esté entre 0 y 100
        # Controlar tanto el inicio como el final de la sesión
        pct_invalido = (df_clean['completion_pct_start'] < 0) | (df_clean['completion_pct_start'] > 100) | \
                       (df_clean['completion_pct_end'] < 0) | (df_clean['completion_pct_end'] > 100)

        if pct_invalido.any():
            n_inv = pct_invalido.sum()
            self.log(f"   ⚠️  {n_inv} sesiones con porcentajes fuera de rango [0,100]")
            # Recortar porcentajes a rango válido [0, 100]
            df_clean.loc[pct_invalido, 'completion_pct_start'] = df_clean.loc[pct_invalido, 'completion_pct_start'].clip(0, 100)
            df_clean.loc[pct_invalido, 'completion_pct_end'] = df_clean.loc[pct_invalido, 'completion_pct_end'].clip(0, 100)
            fuera_rango += n_inv
        else:
            self.log("   ✓ Todos los porcentajes en rango válido")

        return df_clean
    
    # ==========================================
    # PIPELINE COMPLETO
    # ==========================================
    
    def limpiar(self, df):
        """
        Ejecuta el pipeline completo de limpieza
        """
        self.log("="*70)
        self.log("INICIANDO LIMPIEZA DE DATOS")
        self.log("="*70)
        self.log(f"\nDataset original: {len(df)} filas, {len(df.columns)} columnas")

        # Paso 1: Detectar y eliminar duplicados
        df = self.detectar_duplicados_exactos(df)
        df = self.detectar_duplicados_sesiones(df)

        # Paso 2: Analizar e imputar valores faltantes
        self.analizar_valores_faltantes(df)
        df = self.imputar_valores_faltantes(df)

        # Paso 3: Detectar y tratar outliers
        self.analizar_outliers(df)
        df = self.tratar_outliers(df, metodo='clip')

        # Paso 4: Validar consistencia de los datos
        df = self.validar_consistencia_temporal(df)
        df = self.validar_consistencia_progreso(df)
        df = self.validar_rangos(df)

        # Mostrar resumen de la limpieza
        self.log("\n" + "="*70)
        self.log("LIMPIEZA COMPLETADA")
        self.log("="*70)
        self.log(f"\nDataset limpio: {len(df)} filas")
        # Calcular filas eliminadas
        self.log(f"Filas eliminadas: {self.reporte_limpieza['duplicados_eliminados'] + self.reporte_limpieza['outliers_detectados']}")
        self.log("\nResumen de acciones:")
        for key, value in self.reporte_limpieza.items():
            self.log(f"  • {key}: {value}")

        return df

    def generar_reporte_limpieza(self, df_original, df_limpio, output_file='reporte_limpieza.txt'):
        """
        Genera un reporte detallado de la limpieza en un archivo de texto
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Encabezado del reporte
            f.write("="*70 + "\n")
            f.write("REPORTE DE LIMPIEZA DE DATOS\n")
            f.write("="*70 + "\n\n")

            # Fecha y hora del reporte
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Resumen general de la limpieza
            f.write("RESUMEN GENERAL\n")
            f.write("-" * 70 + "\n")
            f.write(f"Filas originales: {len(df_original)}\n")
            f.write(f"Filas finales: {len(df_limpio)}\n")
            f.write(f"Filas eliminadas: {len(df_original) - len(df_limpio)}\n")
            # Calcular porcentaje de datos retenidos
            f.write(f"Porcentaje retenido: {(len(df_limpio)/len(df_original))*100:.2f}%\n\n")

            # Detalle de acciones realizadas
            f.write("ACCIONES REALIZADAS\n")
            f.write("-" * 70 + "\n")
            for key, value in self.reporte_limpieza.items():
                f.write(f"{key}: {value}\n")

        self.log(f"\n✓ Reporte guardado en: {output_file}")


# Función principal para ejecutar la limpieza
def ejecutar_limpieza(input_file, output_file='datos_sesiones_limpios.csv'):
    """
    Función principal para ejecutar el pipeline de limpieza
    """
    # Cargar el archivo CSV con los datos originales
    print("\nCargando datos...")
    df = pd.read_csv(input_file)
    print(f"✓ Cargados {len(df)} registros")

    # Crear instancia del limpiador de datos
    limpiador = LimpiadorDatos(verbose=True)
    # Ejecutar la limpieza completa
    df_limpio = limpiador.limpiar(df)

    # Guardar los datos limpios en un nuevo archivo CSV
    df_limpio.to_csv(output_file, index=False)
    print(f"\n✓ Datos limpios guardados en: {output_file}")

    # Generar reporte detallado de la limpieza
    limpiador.generar_reporte_limpieza(df, df_limpio)

    return df_limpio


if __name__ == "__main__":
    # ========================================
    # Ejecutar limpieza de datos
    # ========================================
    # Archivo CSV con datos originales
    df_limpio = ejecutar_limpieza(
        input_file='datos_sesiones_lectura.csv',        # Datos originales a limpiar
        output_file='datos_sesiones_limpios.csv'        # Archivo de salida con datos limpios
    )