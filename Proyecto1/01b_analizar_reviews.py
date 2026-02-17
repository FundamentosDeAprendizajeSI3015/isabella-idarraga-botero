"""
ANÁLISIS DE REVIEWS - EXTRACCIÓN DE CARACTERÍSTICAS DE ESTILO
==============================================================
Procesa reviews de Goodreads para extraer features cualitativas que
ayuden a predecir el abandono de lectura.

Características extraídas:
- Complejidad del estilo de escritura
- Ritmo narrativo (lento vs rápido)
- Menciones explícitas de abandono
- Nivel de engagement (adictivo, page-turner)
- Sentimiento general
- Emocionalidad

Autor: Isabella Idarraga
"""

import pandas as pd
import numpy as np
import json
import gzip
from collections import Counter
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AnalizadorReviews:
    """
    Analiza reviews de libros para extraer características de estilo
    """

    def __init__(self, verbose=True):
        # verbose: flag para imprimir mensajes de progreso
        self.verbose = verbose
        # estadisticas: diccionario para almacenar estadísticas del análisis (no se usa actualmente)
        self.estadisticas = {}

        # Palabras clave que indican que el lector abandonó el libro
        self.keywords_abandono = [
            'abandon', 'dnf', 'did not finish', 'could not finish',
            'gave up', 'stopped reading', 'quit', 'dropped',
            'couldn\'t finish', 'never finished'
        ]

        # Palabras clave que indicar alto engagement (el libro es adictivo, atrapa al lector)
        self.keywords_engagement_alto = [
            'addictive', 'page turner', 'page-turner', 'couldn\'t put down',
            'could not put down', 'gripping', 'compelling', 'captivating',
            'engrossing', 'unputdownable', 'hooked', 'riveting',
            'fast paced', 'fast-paced', 'kept me reading'
        ]

        # Palabras clave que indican bajo engagement (el libro es aburrido, tedioso)
        self.keywords_engagement_bajo = [
            'boring', 'dull', 'tedious', 'dragged', 'slow',
            'uninteresting', 'monotonous', 'struggled to read'
        ]

        # Palabras clave que indican complejidad alta en el estilo de escritura
        self.keywords_complejo = [
            'complex', 'complicated', 'dense', 'difficult',
            'challenging', 'hard to follow', 'confusing',
            'intricate', 'convoluted', 'hard to understand',
            'requires concentration', 'demanding'
        ]

        # Palabras clave que indican lectura fácil, simple, accesible
        self.keywords_simple = [
            'easy read', 'easy to read', 'simple', 'straightforward',
            'accessible', 'light', 'quick read', 'breeze',
            'simple prose', 'easy to follow', 'effortless'
        ]

        # Palabras clave que sugieren ritmo narrativo lento
        self.keywords_ritmo_lento = [
            'slow', 'slow paced', 'slow-paced', 'dragged',
            'takes time', 'slow start', 'slow beginning',
            'plodding', 'meandering', 'sluggish'
        ]

        # Palabras clave que sugieren ritmo narrativo rápido
        self.keywords_ritmo_rapido = [
            'fast', 'fast paced', 'fast-paced', 'quick',
            'action packed', 'action-packed', 'thrilling',
            'moves quickly', 'rapid', 'brisk pace'
        ]

        # Palabras clave que indican alta carga emocional en el libro
        self.keywords_emocional_alto = [
            'emotional', 'moving', 'touching', 'cried', 'tears',
            'heartbreaking', 'powerful', 'deep', 'profound',
            'made me feel', 'emotional rollercoaster', 'impactful'
        ]

        # Palabras clave que indican baja carga emocional
        self.keywords_emocional_bajo = [
            'emotionless', 'flat', 'cold', 'detached',
            'didn\'t feel', 'no emotion', 'sterile', 'dry'
        ]

    def log(self, mensaje):
        # Imprime mensajes solo si verbose=True
        if self.verbose:
            print(mensaje)

    def limpiar_texto(self, texto):
        """Limpia y normaliza el texto de una review"""
        # Validar que el texto no sea nulo o vacío
        if pd.isna(texto) or texto == '':
            return ''

        # Convertir a minúsculas para búsqueda case-insensitive
        texto = str(texto).lower()

        # Eliminar HTML tags si los hay (eg: <br>, <p>, etc)
        texto = re.sub(r'<[^>]+>', '', texto)

        # Normalizar espacios en blanco (múltiples espacios -> un espacio)
        texto = ' '.join(texto.split())

        return texto

    def contar_keywords(self, texto, keywords):
        """Cuenta cuántas keywords aparecen en el texto"""
        # Primero limpiar el texto
        texto_limpio = self.limpiar_texto(texto)

        # Inicializar contador en cero
        contador = 0
        # Iterar sobre cada keyword y contar ocurrencias
        for keyword in keywords:
            # Contar menciones de cada keyword (case-insensitive)
            contador += texto_limpio.count(keyword)

        return contador
    
    def analizar_review_individual(self, review_texto):
        """
        Analiza una review individual y retorna features
        """
        # Si la review es nula o vacía, retornar None
        if pd.isna(review_texto) or review_texto == '':
            return None

        # Limpiar el texto antes de procesarlo
        texto = self.limpiar_texto(review_texto)

        # Extraer todas las features contando keywords para cada categoría
        features = {
            # Cuántas veces se mencionó abandono o no terminar el libro
            'abandono_mencionado': self.contar_keywords(texto, self.keywords_abandono),
            # Cuántas veces se menciona que fue adictivo/engaging
            'engagement_alto': self.contar_keywords(texto, self.keywords_engagement_alto),
            # Cuántas veces se menciona aburrimiento/bajo engagement
            'engagement_bajo': self.contar_keywords(texto, self.keywords_engagement_bajo),
            # Cuántas veces se menciona complejidad alta
            'complejidad_alta': self.contar_keywords(texto, self.keywords_complejo),
            # Cuántas veces se menciona que fue fácil de leer
            'lectura_facil': self.contar_keywords(texto, self.keywords_simple),
            # Cuántas veces se menciona ritmo lento
            'ritmo_lento': self.contar_keywords(texto, self.keywords_ritmo_lento),
            # Cuántas veces se menciona ritmo rápido
            'ritmo_rapido': self.contar_keywords(texto, self.keywords_ritmo_rapido),
            # Cuántas veces se menciona alta emocionalidad
            'emocional_alto': self.contar_keywords(texto, self.keywords_emocional_alto),
            # Cuántas veces se menciona baja emocionalidad
            'emocional_bajo': self.contar_keywords(texto, self.keywords_emocional_bajo),
        }

        return features
    
    def analizar_reviews_libro(self, reviews_texto_lista):
        """
        Analiza todas las reviews de un libro y retorna features agregadas
        """
        # Si no hay reviews, retornar None
        if len(reviews_texto_lista) == 0:
            return None

        # Analizar cada review individualmente
        features_individuales = []
        for review in reviews_texto_lista:
            feat = self.analizar_review_individual(review)
            if feat is not None:
                features_individuales.append(feat)

        # Si no se extrajeron features válidas, retornar None
        if len(features_individuales) == 0:
            return None

        # Convertir la lista de features individuales a un DataFrame para cálculos más fáciles
        df_features = pd.DataFrame(features_individuales)

        # Total de reviews analizadas
        total_reviews = len(features_individuales)

        # Calcular scores agregados para el libro
        # Los scores representan la proporción de menciones relativas al total de reviews
        features_libro = {
            # Número de reviews que se analizaron
            'num_reviews_analizadas': total_reviews,

            # Score de abandono: % de reviews que mencionan abandono
            'abandono_score': df_features['abandono_mencionado'].sum() / total_reviews,

            # Score de engagement: (engagement positivo - engagement negativo) / total
            # Rango: -1 (muy bajo engagement) a +1 (muy alto engagement)
            'engagement_score': (
                (df_features['engagement_alto'].sum() - df_features['engagement_bajo'].sum())
                / total_reviews
            ),

            # Score de complejidad: (complejidad alta - lectura fácil) / total
            # Rango: -1 (muy simple) a +1 (muy complejo)
            'complejidad_score': (
                (df_features['complejidad_alta'].sum() - df_features['lectura_facil'].sum())
                / total_reviews
            ),

            # Score de ritmo: (ritmo rápido - ritmo lento) / total
            # Rango: -1 (muy lento) a +1 (muy rápido)
            'ritmo_score': (
                (df_features['ritmo_rapido'].sum() - df_features['ritmo_lento'].sum())
                / total_reviews
            ),

            # Score emocional: (emocionalidad alta - emocionalidad baja) / total
            # Rango: -1 (muy poco emocional) a +1 (muy emocional)
            'emocional_score': (
                (df_features['emocional_alto'].sum() - df_features['emocional_bajo'].sum())
                / total_reviews
            ),

            # Menciones absolutas (cantidad total de veces mentionado, sin normalizar)
            # Útil para análisis adicionales
            'menciones_abandono': df_features['abandono_mencionado'].sum(),
            'menciones_engagement_positivo': df_features['engagement_alto'].sum(),
            'menciones_complejidad': df_features['complejidad_alta'].sum(),
            'menciones_ritmo_lento': df_features['ritmo_lento'].sum(),
            'menciones_emocional': df_features['emocional_alto'].sum(),
        }

        return features_libro
    
    def calcular_complejidad_vocabulario(self, reviews_texto_lista):
        """
        Calcula la complejidad del vocabulario promedio en las reviews
        Indicador indirecto de la complejidad del libro
        """
        # Si no hay reviews, retornar None
        if len(reviews_texto_lista) == 0:
            return None

        # Lista para almacenar la longitud de cada palabra
        longitudes_palabra = []

        # Procesar cada review
        for review in reviews_texto_lista:
            if pd.isna(review) or review == '':
                continue

            # Limpiar y dividir en palabras
            palabras = self.limpiar_texto(review).split()
            # Agregar la longitud de cada palabra (excluyendo palabras vacías)
            longitudes_palabra.extend([len(p) for p in palabras if len(p) > 0])

        # Si no hay palabras, retornar None
        if len(longitudes_palabra) == 0:
            return None

        # Calcular estadísticas de longitud de palabras
        return {
            # Longitud promedio de palabra (indicador de complejidad vocabulario)
            'longitud_palabra_promedio': np.mean(longitudes_palabra),
            # Mediana de longitud de palabra (valor central, robusto a outliers)
            'longitud_palabra_mediana': np.median(longitudes_palabra),
            # Desviación estándar (variabilidad en las longitudes)
            'longitud_palabra_std': np.std(longitudes_palabra)
        }

    def calcular_sentimiento_basico(self, reviews_texto_lista):
        """
        Análisis de sentimiento básico usando palabras positivas/negativas
        """
        # Palabras que indican sentimiento positivo
        palabras_positivas = [
            'love', 'loved', 'amazing', 'great', 'excellent', 'wonderful',
            'fantastic', 'brilliant', 'perfect', 'beautiful', 'favorite',
            'enjoyed', 'masterpiece', 'incredible', 'outstanding'
        ]

        # Palabras que indican sentimiento negativo
        palabras_negativas = [
            'hate', 'hated', 'terrible', 'awful', 'horrible', 'worst',
            'disappointing', 'disappointed', 'waste', 'bad', 'poor',
            'boring', 'dull', 'annoying', 'frustrating'
        ]

        # Lista para almacenar los scores de sentimiento de cada review
        sentimientos = []

        # Procesar cada review
        for review in reviews_texto_lista:
            if pd.isna(review) or review == '':
                continue

            # Limpiar el texto
            texto = self.limpiar_texto(review)

            # Contar palabras positivas y negativas
            positivas = sum(texto.count(p) for p in palabras_positivas)
            negativas = sum(texto.count(n) for n in palabras_negativas)

            # Calcular sentimiento: -1 (muy negativo) a +1 (muy positivo)
            # Normalizar por el total de menciones positivas + negativas
            if positivas + negativas > 0:
                sentimiento = (positivas - negativas) / (positivas + negativas)
            else:
                # Si no hay palabras emocionales, sentimiento neutral
                sentimiento = 0

            sentimientos.append(sentimiento)

        # Si no se calcularon sentimientos, retornar None
        if len(sentimientos) == 0:
            return None

        # Calcular estadísticas de sentimiento para todas las reviews
        return {
            # Sentimiento promedio del libro
            'sentimiento_promedio': np.mean(sentimientos),
            # Desviación estándar (variabilidad de opiniones)
            'sentimiento_std': np.std(sentimientos),
            # % de reviews con sentimiento positivo (>0.2)
            'sentimiento_positivo_pct': sum(1 for s in sentimientos if s > 0.2) / len(sentimientos),
            # % de reviews con sentimiento negativo (<-0.2)
            'sentimiento_negativo_pct': sum(1 for s in sentimientos if s < -0.2) / len(sentimientos)
        }


def procesar_reviews_goodreads(reviews_file, output_file='features_reviews.csv',
                               max_libros=None, sample_reviews_per_book=50):
    """
    Procesa el archivo de reviews de Goodreads y extrae features por libro

    Parameters:
    -----------
    reviews_file : str
        Path al archivo goodreads_reviews_dedup.json
    output_file : str
        Nombre del archivo de salida
    max_libros : int
        Número máximo de libros a procesar (None = todos)
    sample_reviews_per_book : int
        Número de reviews a analizar por libro (para no saturar)
    """

    print("="*70)
    print("ANÁLISIS DE REVIEWS DE GOODREADS")
    print("="*70)

    # Crear instancia del analizador
    analizador = AnalizadorReviews(verbose=True)

    # ========================================
    # [1/4] Cargar reviews del archivo JSON
    # ========================================
    print("\n[1/4] Cargando reviews...")
    print("   (Esto puede tomar varios minutos...)")

    # Diccionario para agrupar reviews por book_id
    reviews_por_libro = {}
    # Contador total de reviews cargadas
    total_reviews = 0

    try:
        # Leer el archivo línea por línea (formato JSONL)
        with open(reviews_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Mostrar progreso cada 100k reviews
                if i % 100000 == 0:
                    print(f"   Procesadas {i:,} reviews... ({len(reviews_por_libro):,} libros)")

                try:
                    # Parsear la línea como JSON
                    review = json.loads(line)

                    # Extraer el ID del libro y texto de la review
                    book_id = review.get('book_id')
                    review_text = review.get('review_text', '')

                    # Solo procesar si tenemos ambos datos
                    if book_id and review_text:
                        # Inicializar lista de reviews para este libro si es la primera
                        if book_id not in reviews_por_libro:
                            reviews_por_libro[book_id] = []

                        # Agregar review si no alcanzamos el límite por libro
                        if len(reviews_por_libro[book_id]) < sample_reviews_per_book:
                            reviews_por_libro[book_id].append(review_text)
                            total_reviews += 1

                    # Detener si alcanzamos el máximo de libros
                    if max_libros and len(reviews_por_libro) >= max_libros:
                        break

                except json.JSONDecodeError:
                    # Ignorar líneas con JSON inválido
                    continue
                except Exception as e:
                    # Ignorar cualquier otro error de parsing
                    continue

    except FileNotFoundError:
        print(f"\n❌ Error: No se encuentra el archivo {reviews_file}")
        print("\nDebes descargar:")
        print("  goodreads_reviews_dedup.json")
        print("  desde: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html")
        print("  Sección: 'Book Reviews'")
        return None

    print(f"\n✓ Cargadas reviews de {len(reviews_por_libro):,} libros")
    print(f"✓ Total de reviews procesadas: {total_reviews:,}")

    # ========================================
    # [2/4] Analizar reviews por libro
    # ========================================
    print("\n[2/4] Analizando reviews por libro...")

    # Lista para almacenar las features finales de cada libro
    features_todos_libros = []

    # Iterar sobre cada libro y sus reviews
    for idx, (book_id, reviews) in enumerate(reviews_por_libro.items()):
        # Mostrar progreso cada 500 libros
        if idx % 500 == 0:
            print(f"   Analizados {idx:,}/{len(reviews_por_libro):,} libros")

        # Analizar todas las reviews del libro
        features_libro = analizador.analizar_reviews_libro(reviews)

        # Si se extrajeron features, agregar el book_id y otros análisis
        if features_libro:
            # Agregar ID del libro
            features_libro['book_id'] = book_id

            # Calcular características de vocabulario (complejidad)
            vocab_features = analizador.calcular_complejidad_vocabulario(reviews)
            if vocab_features:
                features_libro.update(vocab_features)

            # Calcular análisis básico de sentimiento
            sent_features = analizador.calcular_sentimiento_basico(reviews)
            if sent_features:
                features_libro.update(sent_features)

            # Agregar los features del libro a la lista total
            features_todos_libros.append(features_libro)

    print(f"\n✓ Analizados {len(features_todos_libros):,} libros con features extraídas")

    # ========================================
    # [3/4] Crear DataFrame con los features
    # ========================================
    print("\n[3/4] Creando dataset de features...")

    # Convertir la lista de diccionarios a DataFrame de pandas
    df_features = pd.DataFrame(features_todos_libros)

    # Asegurar que book_id está en la primera columna para mejor legibilidad
    cols = ['book_id'] + [col for col in df_features.columns if col != 'book_id']
    df_features = df_features[cols]

    print(f"   ✓ Dataset: {len(df_features)} libros × {len(df_features.columns)} features")

    # ========================================
    # [4/4] Guardar el archivo CSV
    # ========================================
    print("\n[4/4] Guardando features...")

    # Guardar el DataFrame a un archivo CSV
    df_features.to_csv(output_file, index=False)

    print(f"\n✓ Features guardadas en: {output_file}")
    print(f"  Tamaño: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")

    # ========================================
    # Mostrar estadísticas finales
    # ========================================
    print("\n" + "="*70)
    print("ESTADÍSTICAS DE FEATURES EXTRAÍDAS")
    print("="*70)

    print("\nFeatures creadas:")
    for col in df_features.columns:
        if col != 'book_id':
            print(f"  • {col}")

    print("\nEstadísticas descriptivas:")
    print(df_features.describe().T[['mean', 'std', 'min', 'max']])

    print("\nDistribución de scores principales:")
    # Análisis del score de abandono
    print(f"\nAbandono Score:")
    print(f"  Media: {df_features['abandono_score'].mean():.4f}")
    print(f"  Std: {df_features['abandono_score'].std():.4f}")
    print(f"  Libros con alta mención de abandono (>0.1): {(df_features['abandono_score'] > 0.1).sum()}")

    # Análisis del score de engagement
    print(f"\nEngagement Score:")
    print(f"  Media: {df_features['engagement_score'].mean():.4f}")
    print(f"  Std: {df_features['engagement_score'].std():.4f}")
    print(f"  Libros muy engaging (>0.5): {(df_features['engagement_score'] > 0.5).sum()}")

    # Análisis del score de complejidad
    print(f"\nComplejidad Score:")
    print(f"  Media: {df_features['complejidad_score'].mean():.4f}")
    print(f"  Std: {df_features['complejidad_score'].std():.4f}")
    print(f"  Libros complejos (>0.3): {(df_features['complejidad_score'] > 0.3).sum()}")

    return df_features


if __name__ == "__main__":
    # ========================================
    # Configuración del análisis
    # ========================================
    # Ruta al archivo de reviews descargado de Goodreads
    REVIEWS_FILE = "datos_goodreads/goodreads_reviews_dedup.json"
    # Nombre del archivo CSV de salida con los features
    OUTPUT_FILE = "features_reviews.csv"

    # Parámetros de procesamiento
    # None = procesar todos los libros, o especificar un número máximo
    MAX_LIBROS = None
    # Número máximo de reviews a analizar por libro (limita procesamiento)
    REVIEWS_POR_LIBRO = 50

    # Mostrar configuración usada
    print("\n" + "="*70)
    print("CONFIGURACIÓN")
    print("="*70)
    print(f"Archivo de reviews: {REVIEWS_FILE}")
    print(f"Archivo de salida: {OUTPUT_FILE}")
    print(f"Máximo libros: {MAX_LIBROS if MAX_LIBROS else 'Todos'}")
    print(f"Reviews por libro: {REVIEWS_POR_LIBRO}")
    print("\n")

    # Ejecutar el análisis de reviews
    df_features = procesar_reviews_goodreads(
        reviews_file=REVIEWS_FILE,
        output_file=OUTPUT_FILE,
        max_libros=MAX_LIBROS,
        sample_reviews_per_book=REVIEWS_POR_LIBRO
    )

    # Si el análisis fue exitoso, mostrar mensaje final
    if df_features is not None:
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPLETADO")
        print("="*70)
        print(f"\nPróximo paso: Integrar features con la simulación")
        print(f"Las features de reviews están en: {OUTPUT_FILE}")