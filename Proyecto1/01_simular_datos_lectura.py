"""
SIMULACIÓN DE DATOS DE SESIONES DE LECTURA CON FEATURES DE REVIEWS
====================================================================
Este script genera datos sintéticos pero realistas de sesiones de lectura
basándose en el dataset de Goodreads, features de reviews, y patrones 
de comportamiento humano.

MEJORA: Ahora integra características extraídas de reviews (abandono_score,
engagement_score, complejidad_score, etc.) para hacer la simulación más
realista y ajustar las probabilidades de abandono.

Autor: Isabella Idarraga
Fecha: Febrero 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de semilla para reproducibilidad
np.random.seed(42)

class SimuladorSesionesLectura:
    """
    Simula sesiones de lectura realistas basadas en:
    - Características del libro (páginas, rating, género)
    - Comportamiento del usuario (rating dado, si terminó el libro)
    - Features de reviews (abandono, engagement, complejidad) 
    - Patrones temporales naturales de lectura
    """
    
    def __init__(self):
        # ========================================
        # PARÁMETROS DE COMPORTAMIENTO DE LECTURA
        # ========================================
        # Basados en estudios psicológicos reales sobre velocidad de lectura
        # Velocidad en palabras por minuto según perfil del lector
        # Se utiliza para calcular cuánto tiempo lleva leer un cierto número de páginas
        self.velocidad_lectura_palabras_min = {
            'rapido': (250, 350),      # Lectores rápidos: 250-350 palabras/minuto (máximo 7 min por página)
            'medio': (180, 250),       # Lectores promedio: 180-250 palabras/minuto (velocidad estándar)
            'lento': (120, 180)        # Lectores lentos/cuidadosos: 120-180 palabras/minuto (máximo 5 min por página)
        }

        # ========================================
        # CALIBRACIÓN POR GÉNERO/TIPO DE LIBRO
        # ========================================
        # Palabras promedio por página según el género/formato del libro
        # Valores más altos = texto más denso/complejo
        # Valores más bajos = más espaciado (dibujos, diálogos, poesía)
        self.palabras_por_pagina = {
            'fiction': 280,           # Novelas: ~280 palabras/página (estándar)
            'non-fiction': 320,       # No-ficción: ~320 palabras/página (más densidad de información)
            'poetry': 180,            # Poesía: ~180 palabras/página (menos denso, énfasis en forma)
            'comics': 100,            # Cómics: ~100 palabras/página (mayormente imágenes)
            'default': 260            # Por defecto: ~260 palabras/página (valor general)
        }

        # ========================================
        # PROBABILIDADES BASE DE ABANDONO
        # ========================================
        # Mapeo de rating (1-5) a probabilidad de abandono del libro
        # Basado en comportamiento real: ratings bajos = mayor abandono
        # Esta es la probabilidad BASE que será AJUSTADA con features de reviews
        self.prob_abandono = {
            1: 0.85,  # Rating 1 (muy malo): 85% de probabilidad de abandono
            2: 0.70,  # Rating 2 (malo): 70% de probabilidad de abandono
            3: 0.40,  # Rating 3 (neutral): 40% de probabilidad de abandono
            4: 0.15,  # Rating 4 (bueno): 15% de probabilidad de abandono
            5: 0.05   # Rating 5 (excelente): 5% de probabilidad de abandono
        }
        
    def determinar_perfil_lector(self, user_id):
        """
        Asigna un perfil consistente de lectura a cada usuario

        Utiliza hash del user_id para garantizar que un mismo usuario siempre
        obtenga el MISMO perfil (rápido/medio/lento) independientemente de
        cuántos libros lea. Esto genera continuidad en el comportamiento simulado.

        Parameters:
        -----------
        user_id : int
            Identificador único del usuario

        Returns:
        --------
        str : Perfil de lectura ('rapido', 'medio', 'lento')
        """
        # Convertir user_id a string y aplicar hash para obtener número determinista
        # El módulo 100 convierte el hash a rango 0-99 (distribución uniforme)
        hash_user = hash(str(user_id)) % 100

        # Distribuir usuarios entre 3 perfiles basados en el hash:
        # 0-19: 20% son lectores rápidos
        # 20-74: 55% son lectores de velocidad media (la mayoría)
        # 75-99: 25% son lectores lentos/cuidadosos
        if hash_user < 20:
            return 'rapido'      # Lectores rápidos: velocidad máxima
        elif hash_user < 75:
            return 'medio'       # Lectores medios: velocidad estándar (más comunes)
        else:
            return 'lento'       # Lectores lentos: velocidad pausada, cuidadosa
    
    def calcular_duracion_sesion(self, paginas_leidas, perfil_lector, genero='default'):
        """
        Calcula duración realista de una sesión de lectura

        Considera:
        - Número de páginas a leer
        - Velocidad del lector (rápido/medio/lento)
        - Densidad de texto según el género (fiction/non-fiction/poetry/comics)
        - Variabilidad natural (algunos dias lee más rápido, otros más lento)

        Formula:
        1. Palabras totales = páginas * palabras_por_página_según_género
        2. Duración base = palabras_totales / velocidad_lectura_del_usuario
        3. Ruido realista = distribución normal con std=10% de duracion_base
        4. Resultado final = duración_base + ruido (mínimo 5 minutos)

        Parameters:
        -----------
        paginas_leidas : int
            Número de páginas a leer en esta sesión
        perfil_lector : str
            Perfil del lector ('rapido', 'medio', 'lento')
        genero : str
            Género del libro ('fiction', 'non-fiction', 'poetry', 'comics')

        Returns:
        --------
        float : Duración estimada en minutos (mínimo 5)
        """
        # Obtener palabras promedio por página según el género del libro
        # Esto calibra el cálculo para que la duración sea realista
        palabras_pag = self.palabras_por_pagina.get(genero, self.palabras_por_pagina['default'])

        # Calcular total de palabras a leer en esta sesión
        total_palabras = paginas_leidas * palabras_pag

        # Obtener velocidad de lectura del usuario (en palabras por minuto)
        # Velocidad es un rango [min, max], se elige valores centrales aleatorios
        vel_min, vel_max = self.velocidad_lectura_palabras_min[perfil_lector]
        velocidad_promedio = np.random.uniform(vel_min, vel_max)

        # Calcular duración base: palabras / velocidad = minutos
        duracion_base = total_palabras / velocidad_promedio

        # Agregar variabilidad realista (ruido gaussiano)
        # Desviación estándar = 10% de la duración (variaciónes naturales día a día)
        ruido = np.random.normal(0, duracion_base * 0.1)

        # Resultado final: duración base + ruido, con mínimo de 5 minutos
        # (no hay sesiones de lectura menores a 5 minutos en datos reales)
        return max(5, duracion_base + ruido)
    
    def generar_patron_temporal_sesiones(self, num_sesiones, fecha_inicio):
        """
        Genera timestamps realistas de sesiones considerando:

        1. PATRÓN DE FRECUENCIA:
           - Primeras sesiones más seguidas (entusiasmo inicial)
           - Sesiones posteriores más espaciadas (abandono gradual)

        2. PATRÓN POR DÍA DE SEMANA:
           - Entre semana (Lun-Vie): picos en mañana (7-9am), mediodía (12-2pm), noche (7-11pm)
           - Fines de semana (Sáb-Dom): más distribuido, picos en mañana y tarde

        3. PATRÓN POR HORA DEL DÍA:
           - Genera horas con probabilidades realistas (personas leen más por la noche)
           - Añade minutos aleatorios (0-59)

        Parameters:
        -----------
        num_sesiones : int
            Número total de sesiones a generar para este usuario-libro
        fecha_inicio : datetime
            Fecha de inicio de la primera sesión

        Returns:
        --------
        list : Lista de datetime objects con timestamps para cada sesión
        """
        timestamps = []
        fecha_actual = fecha_inicio

        # Generar timestamp para cada sesión
        for i in range(num_sesiones):
            # PATRÓN DE FRECUENCIA: Decidir cuántos días hasta la próxima sesión
            if i < num_sesiones * 0.3:
                # PRIMERAS SESIONES (primeros 30%): más frecuentes
                # Distribuidas en [0, 1, 1, 2, 3] días con probabilidades [30%, 40%, 20%, 7%, 3%]
                # Esto simula el entusiasmo inicial (muchas sesiones rápido)
                dias_hasta_proxima = np.random.choice([0, 1, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.07, 0.03])
            else:
                # SESIONES POSTERIORES (últimos 70%): más espaciadas
                # Distribuidas en [1, 2, 3, 4, 5, 7] días con probabilidades [25%, 25%, 20%, 15%, 10%, 5%]
                # Esto simula que el lector se cansa y lee con menos frecuencia
                dias_hasta_proxima = np.random.choice([1, 2, 3, 4, 5, 7], p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05])

            # Avanzar a la fecha de la próxima sesión
            fecha_actual += timedelta(days=int(dias_hasta_proxima))

            # PATRÓN HORARIO: Decidir en qué hora del día ocurre la lectura
            dia_semana = fecha_actual.weekday()  # 0=Lunes, 6=Domingo

            if dia_semana < 5:
                # ENTRE SEMANA (Lunes a Viernes):
                # Picos realistas: 7-9am (mañana), 12-2pm (almuerzo), 7-11pm (noche)
                # La noche es el pico más grande (20% + 15% + 15% + 12% = 62%)
                hora = np.random.choice(
                    [7, 8, 9, 12, 13, 19, 20, 21, 22, 23],
                    p=[0.08, 0.08, 0.06, 0.08, 0.07, 0.15, 0.15, 0.15, 0.12, 0.06]
                )
            else:
                # FINES DE SEMANA (Sábado y Domingo):
                # Más distribuido durante el día (más tiempo libre)
                # Picos más suaves en todo el día
                hora = np.random.choice(
                    range(8, 24),
                    p=[0.05, 0.08, 0.10, 0.08, 0.06, 0.05, 0.05, 0.05, 0.08, 0.10, 0.10, 0.08, 0.06, 0.04, 0.01, 0.01]
                )

            # Añadir minuto aleatorio (0-59)
            minuto = np.random.randint(0, 60)

            # Crear timestamp completo (fecha + hora + minuto)
            timestamp = fecha_actual.replace(hour=hora, minute=minuto, second=0)
            timestamps.append(timestamp)

        return timestamps
    
    def simular_sesiones_completado(self, book_id, user_id, num_pages, genero, rating):
        """
        Simula sesiones para un libro que fue completado (is_read=1, rating alto)

        CARACTERÍSTICAS:
        - El usuario completa el 100% del libro
        - Más sesiones que abandonos (proporcionales al tamaño del libro)
        - Progreso uniforme: primeras sesiones leen más (entusiasmo),
          después progreso más regular
        - Duración total: semanas/meses (datos realistas)

        Parameters:
        -----------
        book_id : int
            Identificador del libro
        user_id : int
            Identificador del usuario
        num_pages : int
            Número total de páginas del libro
        genero : str
            Género del libro (afecta densidad de texto)
        rating : int
            Rating dado por el usuario (1-5)

        Returns:
        --------
        list : Lista de diccionarios con información de cada sesión
        """
        # Determinar el perfil de lectura del usuario (rápido/medio/lento)
        perfil = self.determinar_perfil_lector(user_id)

        # DECIDIR NÚMERO DE SESIONES según tamaño del libro
        # Libros más largos → más sesiones (patrón realista)
        if num_pages < 150:
            num_sesiones = np.random.randint(3, 8)        # Libros cortos: 3-8 sesiones
        elif num_pages < 300:
            num_sesiones = np.random.randint(6, 15)       # Libros medianos: 6-15 sesiones
        elif num_pages < 500:
            num_sesiones = np.random.randint(10, 25)      # Libros largos: 10-25 sesiones
        else:
            num_sesiones = np.random.randint(15, 40)      # Libros muy largos: 15-40 sesiones

        # Generar fecha de inicio de la lectura (últimos 12 meses)
        # Simula que el usuario comenzó a leer hace entre 1 mes y 1 año
        dias_atras = np.random.randint(30, 365)
        fecha_inicio = datetime.now() - timedelta(days=dias_atras)

        # Generar timestamps para todas las sesiones
        # Esto proporciona fechas y horas realistas
        timestamps = self.generar_patron_temporal_sesiones(num_sesiones, fecha_inicio)

        # DISTRIBUIR LAS PÁGINAS A LO LARGO DE LAS SESIONES
        paginas_totales_leer = num_pages  # 100% del libro
        progreso_acumulado = 0            # Páginas leídas hasta ahora
        sesiones = []

        for i, ts_start in enumerate(timestamps):
            # Decidir cuántas páginas leer en esta sesión
            if i == num_sesiones - 1:
                # ÚLTIMA SESIÓN: leer todas las páginas restantes
                paginas_sesion = paginas_totales_leer - progreso_acumulado
            else:
                # SESIONES INTERMEDIAS: distribuir páginas con patrón realista
                if i < num_sesiones * 0.3:
                    # PRIMERAS SESIONES (primeros 30%): factor 1.3
                    # El usuario está entusiasta, lee más páginas (30% más)
                    factor = 1.3
                else:
                    # SESIONES POSTERIORES: factor 1.0
                    # El usuario mantiene ritmo regular (sin entusiasmo extra)
                    factor = 1.0

                # Calcular páginas promedio para las sesiones restantes
                paginas_restantes = paginas_totales_leer - progreso_acumulado
                paginas_promedio_por_sesion = paginas_restantes / (num_sesiones - i)

                # Generar número de páginas con variabilidad realista
                # Usar distribución normal: media=promedio*factor, std=30% de media
                paginas_sesion = max(1, int(np.random.normal(
                    paginas_promedio_por_sesion * factor,
                    paginas_promedio_por_sesion * 0.3
                )))
                # No puede exceder las páginas restantes
                paginas_sesion = min(paginas_sesion, paginas_restantes)

            # Calcular duración de la sesión basada en páginas y velocidad del lector
            duracion_minutos = self.calcular_duracion_sesion(paginas_sesion, perfil, genero)

            # Calcular timestamp de fin de sesión
            ts_end = ts_start + timedelta(minutes=duracion_minutos)

            # Crear registro de la sesión
            sesion = {
                'user_id': user_id,
                'book_id': book_id,
                'session_start': ts_start,              # Cuándo comenzó la sesión
                'session_end': ts_end,                  # Cuándo terminó la sesión
                'duration_minutes': duracion_minutos,   # Cuánto tiempo leyó
                'progress_start': progreso_acumulado,   # Páginas leídas al inicio
                'progress_end': progreso_acumulado + paginas_sesion,  # Al final
                'pages_read': paginas_sesion,           # Páginas en esta sesión
                'completion_pct_start': (progreso_acumulado / num_pages) * 100,
                'completion_pct_end': ((progreso_acumulado + paginas_sesion) / num_pages) * 100
            }

            sesiones.append(sesion)
            progreso_acumulado += paginas_sesion

        return sesiones
    
    def simular_sesiones_abandono_temprano(self, book_id, user_id, num_pages, genero, rating):
        """
        Simula sesiones para un libro abandonado tempranamente (rating bajo)

        CARACTERÍSTICAS:
        - El usuario apenas comienza a leer y abandona rápidamente
        - Pocas sesiones (1-5): no tiene tiempo para leer mucho
        - Progreso muy bajo: 5-30% del libro (mayormente no terminó)
        - Duración: días (no semanas)
        - Patrón: entusiasmo inicial seguido de abandono inmediato

        Parameters:
        -----------
        book_id : int
            Identificador del libro
        user_id : int
            Identificador del usuario
        num_pages : int
            Número total de páginas del libro
        genero : str
            Género del libro
        rating : int
            Rating muy bajo (1-2)

        Returns:
        --------
        list : Lista de sesiones (pocas y con poco progreso)
        """
        # Determinar perfil del lector
        perfil = self.determinar_perfil_lector(user_id)

        # POCAS SESIONES: entre 1 y 5
        # (El usuario abandona muy rápidamente, sin recorrido largo)
        num_sesiones = np.random.randint(1, 5)

        # Fecha de inicio más reciente (últimos 6 meses máximo)
        # Abandonos ocurren más frecuentemente en intentos recientes
        dias_atras = np.random.randint(30, 180)
        fecha_inicio = datetime.now() - timedelta(days=dias_atras)

        # Generar timestamps (con intervalos cortos)
        timestamps = self.generar_patron_temporal_sesiones(num_sesiones, fecha_inicio)

        # PROGRESO TOTAL: solo 5-30% del libro
        # El usuario apenas empezó cuando abandonó
        progreso_total = int(num_pages * np.random.uniform(0.05, 0.30))
        progreso_acumulado = 0
        sesiones = []

        for i, ts_start in enumerate(timestamps):
            # Cuántas páginas leer en esta sesión
            paginas_restantes = progreso_total - progreso_acumulado

            if i == num_sesiones - 1:
                # ÚLTIMA SESIÓN: leer todas las páginas restantes
                paginas_sesion = paginas_restantes
            else:
                # SESIONES INTERMEDIAS: patrón de decaimiento
                # Factor de decaimiento: 1.2 - (i / num_sesiones) = baja gradualmente
                # Ejemplo: si hay 4 sesiones, factores serían [1.05, 0.75, 0.55, 0]
                factor_decaimiento = 1.2 - (i / num_sesiones)

                # Calcular páginas promedio
                paginas_promedio = paginas_restantes / (num_sesiones - i)

                # Generar páginas con variabilidad (40% para simulación más realista)
                paginas_sesion = max(1, int(np.random.normal(
                    paginas_promedio * factor_decaimiento,
                    paginas_promedio * 0.4
                )))
                paginas_sesion = min(paginas_sesion, paginas_restantes)

            # Calcular duración de la sesión
            duracion_minutos = self.calcular_duracion_sesion(paginas_sesion, perfil, genero)
            ts_end = ts_start + timedelta(minutes=duracion_minutos)

            # Crear registro de sesión
            sesion = {
                'user_id': user_id,
                'book_id': book_id,
                'session_start': ts_start,
                'session_end': ts_end,
                'duration_minutes': duracion_minutos,
                'progress_start': progreso_acumulado,
                'progress_end': progreso_acumulado + paginas_sesion,
                'pages_read': paginas_sesion,
                'completion_pct_start': (progreso_acumulado / num_pages) * 100,
                'completion_pct_end': ((progreso_acumulado + paginas_sesion) / num_pages) * 100
            }

            sesiones.append(sesion)
            progreso_acumulado += paginas_sesion

        return sesiones
    
    def simular_sesiones_abandono_medio(self, book_id, user_id, num_pages, genero, rating):
        """
        Simula sesiones para un libro abandonado a mitad (rating medio-bajo)

        CARACTERÍSTICAS:
        - El usuario comienza bien pero abandona a mitad del libro
        - Sesiones moderadas (5-15): más esfuerzo que abandono temprano
        - Progreso medio: 30-70% del libro (avanzó bastante pero no terminó)
        - Duración: semanas (considerable inversión de tiempo)
        - Patrón: lectura irregular, a veces para, a veces retoma
        - Factores: libro es bueno al inicio pero pierde interés

        Parameters:
        -----------
        book_id : int
            Identificador del libro
        user_id : int
            Identificador del usuario
        num_pages : int
            Número total de páginas del libro
        genero : str
            Género del libro
        rating : int
            Rating medio-bajo (3)

        Returns:
        --------
        list : Lista de sesiones con progreso hasta la mitad
        """
        # Determinar perfil del lector
        perfil = self.determinar_perfil_lector(user_id)

        # SESIONES MODERADAS: entre 5 y 15
        # (Más que abandono temprano pero menos que completado)
        num_sesiones = np.random.randint(5, 15)

        # Fecha de inicio: últimos 10 meses
        # Estos lectores dedican más tiempo antes de abandonar
        dias_atras = np.random.randint(60, 300)
        fecha_inicio = datetime.now() - timedelta(days=dias_atras)

        # Generar timestamps
        timestamps = self.generar_patron_temporal_sesiones(num_sesiones, fecha_inicio)

        # PROGRESO TOTAL: 30-70% del libro
        # El usuario avanzó bastante pero no logró terminar
        progreso_total = int(num_pages * np.random.uniform(0.30, 0.70))
        progreso_acumulado = 0
        sesiones = []

        for i, ts_start in enumerate(timestamps):
            # Cuántas páginas leer en esta sesión
            paginas_restantes = progreso_total - progreso_acumulado

            if i == num_sesiones - 1:
                # ÚLTIMA SESIÓN: leer todas las páginas restantes
                paginas_sesion = paginas_restantes
            else:
                # SESIONES INTERMEDIAS: patrón IRREGULAR
                # A diferencia de completado (patrón suave) y abandono_temprano (patrón decreciente),
                # aquí alternamos entre leer más y leer menos (simulando pérdida de interés)
                if np.random.random() < 0.3:
                    # 30% de las veces: sesión corta (el usuario se cansa)
                    factor = 0.5  # Lee solo 50% del promedio
                else:
                    # 70% de las veces: sesión normal
                    factor = 1.0

                # Calcular páginas promedio
                paginas_promedio = paginas_restantes / (num_sesiones - i)

                # Generar páginas con alta variabilidad (50% para realismo)
                paginas_sesion = max(1, int(np.random.normal(
                    paginas_promedio * factor,
                    paginas_promedio * 0.5
                )))
                paginas_sesion = min(paginas_sesion, paginas_restantes)

            # Calcular duración de la sesión
            duracion_minutos = self.calcular_duracion_sesion(paginas_sesion, perfil, genero)
            ts_end = ts_start + timedelta(minutes=duracion_minutos)

            # Crear registro de sesión
            sesion = {
                'user_id': user_id,
                'book_id': book_id,
                'session_start': ts_start,
                'session_end': ts_end,
                'duration_minutes': duracion_minutos,
                'progress_start': progreso_acumulado,
                'progress_end': progreso_acumulado + paginas_sesion,
                'pages_read': paginas_sesion,
                'completion_pct_start': (progreso_acumulado / num_pages) * 100,
                'completion_pct_end': ((progreso_acumulado + paginas_sesion) / num_pages) * 100
            }

            sesiones.append(sesion)
            progreso_acumulado += paginas_sesion

        return sesiones
    
    def simular_para_interaccion(self, row):
        """
        VERSIÓN MEJORADA CON FEATURES DE REVIEWS 

        Este es el método CENTRAL que integra las características extraídas de reviews
        (abandono_score, engagement_score, complejidad_score, ritmo_score) para
        AJUSTAR las probabilidades de abandono de forma realista.

        LÓGICA:
        1. Si marcó como leído (is_read=1) y rating alto (>=4) → COMPLETÓ el libro
        2. Si rating muy bajo (<=2) → ABANDONO TEMPRANO (apenas leyó)
        3. Si rating medio (3) → Decidir con probabilidad AJUSTADA según features de reviews

        AJUSTES CON REVIEW FEATURES:
        - abandono_score > 0     → +25% probabilidad de abandono (comunidad menciona abandono)
        - engagement_score > 0   → -15% probabilidad de abandono (libro muy engaging)
        - complejidad_score > 0  → +10% probabilidad de abandono (libro difícil = más abandono)
        - ritmo_score < 0        → +8% probabilidad de abandono (libro lento)

        Ejemplo: Un libro con rating=3 tiene 40% abandono base, pero si reviewers
                 dicen es "very complex" y "slow paced", la probabilidad podría subir a 60%.

        Parameters:
        -----------
        row : pandas.Series
            Fila del dataframe con columnas:
            - book_id, user_id, is_read, rating, num_pages, genres
            - abandono_score, engagement_score, complejidad_score, ritmo_score
              (features de reviews, valor por defecto 0 si no existen)

        Returns:
        --------
        list : Lista de diccionarios con sesiones simuladas
               [] si no hay suficiente información para simular
        """
        # Extraer parámetros básicos de la fila
        book_id = row['book_id']
        user_id = row['user_id']
        is_read = row.get('is_read', 0)
        rating = row.get('rating', 3)

        # Convertir num_pages de forma segura (puede venir como NaN, string, float, etc.)
        num_pages = row.get('num_pages', 300)
        try:
            num_pages = int(float(num_pages)) if pd.notna(num_pages) else 300
        except (ValueError, TypeError):
            num_pages = 300
        num_pages = max(1, num_pages)  # Asegurar que sea al menos 1 página

        # Obtener género del libro
        genero = row.get('genres', 'default')

        # ========================================
        # EXTRAER FEATURES DE REVIEWS
        # ========================================
        # Estos features vienen del análisis de sentimientos en 01b_analizar_reviews.py
        # Si no existen (libro sin reviews), el valor por defecto es 0 (neutro)
        abandono_score = row.get('abandono_score', 0)     # % de menciones de abandono
        engagement_score = row.get('engagement_score', 0) # Balance: engagement_alto - engagement_bajo
        complejidad_score = row.get('complejidad_score', 0)  # Balance: complejidad - lectura_fácil
        ritmo_score = row.get('ritmo_score', 0)           # Balance: ritmo_rápido - ritmo_lento

        # CASO 1: Sin información de lectura → no hay sesiones
        if is_read == 0 and pd.isna(rating):
            return []

        # CASO 2: Rating alto y marcó como leído → LIBRO COMPLETADO
        if is_read == 1 and rating >= 4:
            return self.simular_sesiones_completado(book_id, user_id, num_pages, genero, rating)

        # CASO 3: Rating muy bajo → ABANDONO TEMPRANO (apenas empezó)
        elif rating <= 2:
            return self.simular_sesiones_abandono_temprano(book_id, user_id, num_pages, genero, rating)

        # CASO 4: Rating MEDIO (3) → Usar probabilidad AJUSTADA
        else:
            # Obtener probabilidad base según el rating (de la tabla self.prob_abandono)
            prob_aband = self.prob_abandono.get(int(rating), 0.5)

            # ========================================
            # AJUSTES CON FEATURES DE REVIEWS
            # ========================================
            # Modificar la probabilidad base según lo que las reviews dicen del libro

            # 1. ABANDONO SCORE: ¿Las reviews mencionan abandono?
            # Si reviewers dicen "DNF", "gave up", etc. → aumenta probabilidad
            # Peso: máximo +0.25 (25 puntos porcentuales)
            if abandono_score > 0:
                prob_aband += min(abandono_score, 1.0) * 0.25

            # 2. ENGAGEMENT SCORE: ¿Qué tan engaging es el libro?
            # Si reviewers dicen "addictive", "couldn't put down" → reduce probabilidad
            # Peso: máximo -0.15 (15 puntos porcentuales)
            if engagement_score > 0:
                prob_aband -= min(engagement_score, 1.0) * 0.15

            # 3. COMPLEJIDAD SCORE: ¿Qué tan complejo es?
            # Si reviewers dicen "complex", "dense", "confusing" → aumenta probabilidad
            # Peso: máximo +0.10 (10 puntos porcentuales)
            if complejidad_score > 0:
                prob_aband += min(complejidad_score, 1.0) * 0.10

            # 4. RITMO SCORE: ¿Qué tan lento es?
            # Si ritmo_score < 0 significa "lento" (reviewers dijeron "slow", "slow-paced")
            # Libros lentos tienen más abandono → aumenta probabilidad
            # Peso: máximo +0.08 (8 puntos porcentuales)
            if ritmo_score < 0:
                prob_aband += min(abs(ritmo_score), 1.0) * 0.08

            # LIMITAR PROBABILIDAD: debe estar entre 0.0 y 1.0
            prob_aband = max(0.0, min(1.0, prob_aband))

            # DECIDIR: ¿Abandono o completado?
            # Usar la probabilidad ajustada para decidir
            if np.random.random() < prob_aband:
                # DECISIÓN: ABANDONO
                # Pero, ¿es abandono temprano o a mitad?
                # Si engagement_score es MUY NEGATIVO (<-0.2) → abandono temprano
                # En caso contrario → abandono a mitad o irregular
                if engagement_score < -0.2:
                    # Muy bajo engagement → el usuario se desinteresó rápidamente
                    return self.simular_sesiones_abandono_temprano(book_id, user_id, num_pages, genero, rating)
                else:
                    # Engagement neutral o positivo pero igualmente abandonó → a mitad
                    return self.simular_sesiones_abandono_medio(book_id, user_id, num_pages, genero, rating)
            else:
                # DECISIÓN: COMPLETADO (a pesar del rating medio)
                # Ejemplo: Libro complejo pero el usuario persistió y lo terminó
                return self.simular_sesiones_completado(book_id, user_id, num_pages, genero, rating)


def procesar_interacciones_y_generar_sesiones(
    interactions_file,
    books_file,
    reviews_features_file='features_reviews.csv',  
    output_file='datos_sesiones_lectura.csv',
    max_interacciones=50000
):
    """
    ⭐ FUNCIÓN PRINCIPAL: PIPELINE COMPLETO DE SIMULACIÓN ⭐

    Orquesta todo el proceso de generación de datos sintéticos:
    1. Cargar interacciones de usuarios (Goodreads)
    2. Cargar metadatos de libros (número de páginas, géneros)
    3. Cargar features extraídos de reviews (abandono_score, engagement_score, etc.)
    4. Combinar todos los datos
    5. Simular sesiones realistas para cada interacción usuario-libro
    6. Guardar el dataset completo

    PIPELINE:
    --------
    interactions.csv (user_id, book_id, rating, is_read)
                ↓
    books.json (book_id, num_pages, genres)
                ↓
    features_reviews.csv (book_id, abandono_score, engagement_score, ...)
                ↓
    [COMBINACIÓN DE DATOS]
                ↓
    [PARA CADA INTERACCIÓN]
      → Determinar perfil del lector
      → Ajustar probabilidades con review features
      → Generar sesiones realistas
                ↓
    datos_sesiones_lectura.csv (50,000+ sesiones)

    VERSIÓN MEJORADA:
    - Integra features de reviews (abandono_score, engagement_score, etc.)
    - Ajusta probabilidades de abandono según lo que las reviews dicen
    - Genera datos más realistas y correlacionados con sentimiento

    Parameters:
    -----------
    interactions_file : str
        Path a CSV con interacciones de usuarios (user_id, book_id, rating, is_read)
        Ejemplo: "datos_goodreads/goodreads_interactions.csv"

    books_file : str
        Path a JSON Lines con metadatos de libros (book_id, num_pages, genres, etc.)
        Ejemplo: "datos_goodreads/goodreads_books.json"
        Formato: una línea JSON por libro

    reviews_features_file : str
        ⭐ NUEVO: Path a CSV con features extraídos de reviews
        Ejemplo: "features_reviews.csv" (output de 01b_analizar_reviews.py)
        Contiene: abandono_score, engagement_score, complejidad_score, ritmo_score, etc.

    output_file : str
        Path de salida para guardar las sesiones generadas
        Ejemplo: "datos_sesiones_lectura.csv"

    max_interacciones : int
        Número máximo de interacciones a procesar
        Por defecto: 50,000 (puede reducirse para testing)
        Razón: limitar tiempo de ejecución y tamaño de memoria

    Returns:
    --------
    pd.DataFrame : DataFrame con todas las sesiones generadas
    """

    print("="*70)
    print("SIMULACIÓN DE DATOS CON FEATURES DE REVIEWS")
    print("="*70)

    # ========================================
    # PASO 1: Cargar interacciones
    # ========================================
    print("\n[1/6] Cargando interacciones de usuarios...")
    # Leer archivo CSV con todas las interacciones usuario-libro
    interactions = pd.read_csv(interactions_file)
    print(f"   ✓ Cargadas {len(interactions):,} interacciones")

    # Filtrar: solo interacciones con rating (no vacíos)
    # Necesitamos rating para decidir si fue abandono o completado
    interactions = interactions[interactions['rating'].notna()]
    print(f"   ✓ Filtradas a {len(interactions):,} interacciones con rating")

    # Limitar número de interacciones para no saturar la máquina
    if len(interactions) > max_interacciones:
        interactions = interactions.sample(n=max_interacciones, random_state=42)
        print(f"   ✓ Muestreadas {max_interacciones:,} interacciones para simulación")

    # ========================================
    # PASO 2: Cargar metadatos de libros
    # ========================================
    print("\n[2/6] Cargando metadatos de libros...")
    # Leer archivo JSON Lines con información de libros
    books_data = []
    with open(books_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Parsear cada línea como objeto JSON
                book = json.loads(line)

                # Extraer book_id de forma segura (puede venir como string o int)
                book_id_raw = book.get('book_id')
                try:
                    book_id = int(book_id_raw) if book_id_raw else None
                except (ValueError, TypeError):
                    book_id = None

                # Saltar si no tiene book_id válido
                if book_id is None:
                    continue

                # Extraer información relevante del libro
                books_data.append({
                    'book_id': book_id,  # ID del libro
                    'num_pages': book.get('num_pages', 300),  # Número de páginas (default 300 si no existe)
                    # Extraer género del primer "popular_shelf" (categoría más popular)
                    'genres': book.get('popular_shelves', [{}])[0].get('name', 'fiction') if book.get('popular_shelves') else 'fiction'
                })
            except:
                # Ignorar líneas inválidas
                continue

    # Crear DataFrame con información de libros
    books_df = pd.DataFrame(books_data)
    print(f"   ✓ Cargados metadatos de {len(books_df):,} libros")

    # ========================================
    # PASO 3: Cargar features de reviews
    # ========================================
    print("\n[3/6] Cargando features de reviews...")
    try:
        # Leer CSV con features calculados en 01b_analizar_reviews.py
        features_reviews = pd.read_csv(reviews_features_file)
        print(f"   ✓ Features de reviews para {len(features_reviews):,} libros")
        print(f"   ✓ Features disponibles: {len(features_reviews.columns)-1}")
        print(f"      • abandono_score, engagement_score, complejidad_score,")
        print(f"      • ritmo_score, emocional_score, sentimiento_promedio")
        print(f"      • y {len(features_reviews.columns)-7} más...")
    except FileNotFoundError:
        # Si no existe features_reviews.csv, continuar sin él
        # Las simulaciones serán menos realistas pero seguirán funcionando
        print(f"   ⚠️  No se encontró {reviews_features_file}")
        print("   Continuando sin features de reviews...")
        features_reviews = None

    # ========================================
    # PASO 4: Unir datos (interacciones + libros + features)
    # ========================================
    print("\n[4/6] Combinando datos...")
    # Combinar interacciones con metadatos de libros (join por book_id)
    data = interactions.merge(books_df, on='book_id', how='left')

    #  Combinar con features de reviews
    if features_reviews is not None:
        # Convertir book_id de features_reviews a int para que coincida
        print("   • Convirtiendo book_id de features_reviews a int...")
        features_reviews['book_id'] = pd.to_numeric(features_reviews['book_id'], errors='coerce')
        # Eliminar filas con book_id inválido
        features_reviews = features_reviews.dropna(subset=['book_id'])
        # Convertir a int
        features_reviews['book_id'] = features_reviews['book_id'].astype(int)

        # Combinar con features de reviews (left join: mantener todas las interacciones)
        data = data.merge(features_reviews, on='book_id', how='left')
        print(f"   ✓ Integradas features de reviews")

        # Imputar valores faltantes de reviews con 0 (neutro)
        # Si un libro no tiene reviews analizadas, asumir características neutras
        review_cols = [col for col in features_reviews.columns if col != 'book_id']
        for col in review_cols:
            if col in data.columns:
                data[col] = data[col].fillna(0)

    # Imputar otros valores faltantes
    data['num_pages'] = data['num_pages'].fillna(300)  # Default 300 páginas si no existe
    data['genres'] = data['genres'].fillna('fiction')  # Default 'fiction' si no existe

    print(f"   ✓ Dataset combinado: {len(data):,} filas")

    # ========================================
    # PASO 5: Generar sesiones para cada interacción
    # ========================================
    print("\n[5/6] Generando sesiones de lectura...")
    if features_reviews is not None:
        print("   ⭐ Usando features de reviews para simulación más realista")
    print("   (Esto puede tomar unos minutos...)")

    # Crear simulador
    simulador = SimuladorSesionesLectura()

    # Almacenar todas las sesiones generadas
    todas_sesiones = []

    # Para cada interacción usuario-libro, generar sesiones
    for idx, row in data.iterrows():
        # Mostrar progreso cada 5000 interacciones
        if idx % 5000 == 0:
            print(f"   Progreso: {idx:,}/{len(data):,} interacciones procesadas")

        # Simular sesiones para esta interacción
        # La lógica está en simular_para_interaccion() que considera las features de reviews
        sesiones = simulador.simular_para_interaccion(row)
        todas_sesiones.extend(sesiones)

    # ========================================
    # PASO 6: Guardar resultados
    # ========================================
    print("\n[6/6] Guardando datos simulados...")
    # Convertir lista de sesiones a DataFrame
    df_sesiones = pd.DataFrame(todas_sesiones)
    # Guardar como CSV
    df_sesiones.to_csv(output_file, index=False)

    # ========================================
    # REPORTE FINAL
    # ========================================
    print(f"\n{'='*70}")
    print("✓ SIMULACIÓN COMPLETADA CON FEATURES DE REVIEWS")
    print(f"{'='*70}")
    print(f"\nEstadísticas del dataset generado:")
    print(f"  • Total de sesiones: {len(df_sesiones):,}")
    print(f"  • Usuarios únicos: {df_sesiones['user_id'].nunique():,}")
    print(f"  • Libros únicos: {df_sesiones['book_id'].nunique():,}")
    print(f"  • Duración promedio por sesión: {df_sesiones['duration_minutes'].mean():.1f} minutos")
    print(f"  • Páginas promedio por sesión: {df_sesiones['pages_read'].mean():.1f}")
    print(f"\nArchivo guardado: {output_file}")
    print(f"Tamaño: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")

    # Mostrar si se utilizaron features de reviews
    if features_reviews is not None:
        print(f"\n⭐ MEJORA: Simulación ajustada con features de reviews")
        print(f"   Las probabilidades de abandono fueron modificadas según:")
        print(f"   - Menciones de abandono en reviews")
        print(f"   - Nivel de engagement reportado")
        print(f"   - Complejidad del estilo")
        print(f"   - Ritmo narrativo")

    return df_sesiones


if __name__ == "__main__":
    # ========================================
    # PUNTO DE ENTRADA: Ejecutar simulación
    # ========================================
    # Este bloque se ejecuta cuando se corre el script directamente desde línea de comandos
    # Ejemplo: python 01_simular_datos_lectura.py

    # CONFIGURACIÓN DE ARCHIVOS
    # ========================================
    # Ruta al archivo de interacciones usuario-libro (input de Goodreads)
    # Formato: CSV con columnas (user_id, book_id, rating, is_read, ...)
    INTERACTIONS_FILE = "datos_goodreads/goodreads_interactions.csv"

    # Ruta al archivo de metadatos de libros (input de Goodreads)
    # Formato: JSON Lines (una línea JSON por libro)
    # Cada línea contiene: {book_id, num_pages, popular_shelves, ...}
    BOOKS_FILE = "datos_goodreads/goodreads_books.json"

    # Ruta al archivo de features extraídos de reviews
    # ESTA ES LA MEJORA PRINCIPAL
    # Generado por: 01b_analizar_reviews.py
    # Contiene: abandono_score, engagement_score, complejidad_score, ritmo_score, etc.
    REVIEWS_FEATURES_FILE = "features_reviews.csv"

    # Ruta del archivo de salida
    # Se guardará aquí el dataset con todas las sesiones simuladas
    OUTPUT_FILE = "datos_sesiones_lectura.csv"

    # EJECUTAR SIMULACIÓN
    # ========================================
    try:
        # Llamar a la función principal para generar todas las sesiones
        # Esta función orquesta todo el pipeline de simulación
        df_sesiones = procesar_interacciones_y_generar_sesiones(
            interactions_file=INTERACTIONS_FILE,          # Interacciones usuario-libro
            books_file=BOOKS_FILE,                        # Metadatos de libros
            reviews_features_file=REVIEWS_FEATURES_FILE,  # Features de reviews
            output_file=OUTPUT_FILE,                      # Salida
            max_interacciones=50000                       # Limitar a 50K para evitar saturar
            # Nota: Puedes reducir este número si tu computadora tiene limitaciones
        )

    # MANEJO DE ERRORES
    # ========================================
    except FileNotFoundError as e:
        # Si falta algún archivo, mostrar mensaje de error informativo
        print(f"\n❌ Error: No se encontró el archivo {e}")
        print("\nArchivos necesarios:")
        print("  1. datos_goodreads/goodreads_interactions.csv")
        print("     → Descargable desde Goodreads dataset")
        print("  2. datos_goodreads/goodreads_books.json")
        print("     → Descargable desde Goodreads dataset")
        print("  3. features_reviews.csv (generado con 01b_analizar_reviews.py)")
        print("     → Producto del análisis de reviews")
        print("\nPara generar features_reviews.csv:")
        print("  python 01b_analizar_reviews.py")