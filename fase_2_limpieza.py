# -*- coding: utf-8 -*-
"""Fase_2_Limpieza.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FrNitPptpxQUxIvcKlv8Sp6lvgOdLlJt
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

df_tmdb = pd.read_csv('/content/drive/MyDrive/tmdb_5000_movies.csv')
df_imdb = pd.read_csv('/content/drive/MyDrive/IMDB Top 250 Movies.csv')

df_tmdb.head(2)

print(df_tmdb.info())

import matplotlib.pyplot as plt
import seaborn as sns

# 2. Distribución de la Popularidad (Histograma)
sns.histplot(df_tmdb['popularity'], kde=True)
plt.xlabel('Popularidad')
plt.ylabel('Frecuencia')
plt.title('Distribución de la Popularidad')
plt.show()

# 3. Géneros más Populares (Gráfico de Barras)
genres_count = df_tmdb['genres'].str.split(',').apply(len)
sns.countplot(x=genres_count, palette='viridis')
plt.xlabel('Número de Géneros')
plt.ylabel('Número de Películas')
plt.title('Número de Géneros en las Películas')
plt.show()

# 4. Lenguajes Hablados (Gráfico de Barras)
spoken_languages_count = df_tmdb['spoken_languages'].str.split(',').apply(len)
sns.countplot(x=spoken_languages_count, palette='coolwarm')
plt.xlabel('Número de Lenguajes Hablados')
plt.ylabel('Número de Películas')
plt.title('Número de Lenguajes Hablados en las Películas')
plt.show()

# 5. Estado de las Películas (Gráfico de Barras)
sns.countplot(x=df_tmdb['status'], palette='Set1')
plt.xlabel('Estado de la Película')
plt.ylabel('Número de Películas')
plt.title('Estado de las Películas')
plt.show()

# 6. Uso de Página de Inicio (Homepage) (Gráfico de Barras)
df_tmdb['homepage_present'] = df_tmdb['homepage'].apply(lambda x: 0 if pd.isna(x) else 1)
sns.countplot(x=df_tmdb['homepage_present'], palette='rainbow')
plt.xticks([0, 1], ['Sin Página de Inicio', 'Con Página de Inicio'])
plt.ylabel('Número de Películas')
plt.title('Uso de Página de Inicio (Homepage)')
plt.show()

"""## Para el primer dataset"""

# 1. Exploración Inicial de Datos
print(df_tmdb.info())
print(df_tmdb.head())

# 2. Identificación de Datos Faltantes
print(df_tmdb.isnull().sum())

df_tmdb['homepage'].fillna('Desconocido', inplace=True)

df_tmdb.dropna(subset=['overview'], inplace=True)

df_tmdb.dropna(subset=['release_date'], inplace=True)

mean_runtime = df_tmdb['runtime'].mean()
df_tmdb['runtime'].fillna(mean_runtime, inplace=True)

df_tmdb['tagline'].fillna('No disponible', inplace=True)

print(df_tmdb.isnull().sum())

#Los datos estaban en formato JSON, para poder trabajar mejor con ellos, los separé en diferentes columnas
import json

# Analizar la columna "genres" para extraer los géneros
df_tmdb['genres'] = df_tmdb['genres'].apply(json.loads)

# Crear una nueva columna que contenga una lista de nombres de géneros
df_tmdb['genre_names'] = df_tmdb['genres'].apply(lambda x: [genre['name'] for genre in x])

for genre in df_tmdb['genre_names'].explode().unique():
    df_tmdb[genre] = df_tmdb['genre_names'].apply(lambda x: 1 if genre in x else 0)

df_tmdb['keywords'] = df_tmdb['keywords'].apply(json.loads)
df_tmdb['keyword_names'] = df_tmdb['keywords'].apply(lambda x: [keyword['name'] for keyword in x])

# Analizar la columna "production_companies" como objetos JSON
df_tmdb['production_company_names'] = df_tmdb['production_companies']

# Puedes crear una nueva columna que contenga esta lista de países.
df_tmdb['production_country_names'] = df_tmdb['production_countries']

# Crear columnas separadas para cada país de producción
country_columns = pd.get_dummies(df_tmdb['production_country_names'].explode())
df_tmdb = pd.concat([df_tmdb, country_columns], axis=1)

df_tmdb['spoken_language_names'] = df_tmdb['spoken_languages']

from datetime import datetime
df_tmdb['release_date'] = df_tmdb['release_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

# 6. Manejo de Outliers (ejemplo de filtrar películas con calificación de IMDb menor a 5.0)
df_tmdb = df_tmdb[df_tmdb['vote_average'] >= 5.0]

from collections import Counter

# Supongamos que 'keyword_names' es la columna que contiene las listas de palabras clave
keyword_lists = df_tmdb['keyword_names'].dropna()

# Concatena todas las listas de palabras clave en una sola lista
all_keywords = [keyword for keywords in keyword_lists for keyword in keywords]

# Cuenta la frecuencia de cada palabra clave
keyword_counts = Counter(all_keywords)

# Obtén las palabras clave más comunes (puedes ajustar la cantidad según tu preferencia)
top_keywords = keyword_counts.most_common(10)

# Separa las palabras clave y sus conteos
top_keywords, top_counts = zip(*top_keywords)

# Crea un gráfico de barras de las palabras clave más comunes
plt.figure(figsize=(10, 6))
plt.barh(top_keywords, top_counts)
plt.xlabel('Frecuencia')
plt.ylabel('Palabras Clave')
plt.title('Palabras Clave más Comunes')
plt.gca().invert_yaxis()  # Invierte el eje y para mostrar la palabra clave más común en la parte superior
plt.show()

"""## Para el segundo dataset"""

df_imdb.head(2)

print(df_imdb.isnull().sum())

import re

def convertir_a_minutos(duracion):
    match = re.search(r'(\d+)h (\d+)m', duracion)
    if match:
        horas = int(match.group(1))
        minutos = int(match.group(2))
        return horas * 60 + minutos
    return None  # Manejar casos no válidos

# Supongamos que tienes una columna "runtime" con duraciones en "2h 22m"
df_imdb['run_time'] = df_imdb['run_time'].apply(convertir_a_minutos)

"""## Datos limpios"""

# Guardar los datos limpios en nuevos archivos CSV
df_tmdb.to_csv('tmdb_clean.csv', index=False)
df_imdb.to_csv('imdb_clean.csv', index=False)

"""## Tercer Dataset"""

import pandas as pd
df_rt = pd.read_csv('/content/drive/MyDrive/ratings_small.csv')

df_rt.head(5)

print(df_rt.isnull().sum())

import matplotlib.pyplot as plt

#diagrama de caja para la columna 'rating'
plt.figure(figsize=(8, 6))
plt.boxplot(df_rt['rating'], vert=False)
plt.title('Diagrama de Caja de la Calificación (Rating)')
plt.xlabel('Calificación')
plt.show()

# Calcular los percentiles 25 y 75 para el rango intercuartílico
Q1 = df_rt['rating'].quantile(0.25)
Q3 = df_rt['rating'].quantile(0.75)
IQR = Q3 - Q1

# Definir el umbral para considerar valores atípicos
outlier_threshold = 1.5

# Identificar y tratar valores atípicos
df = df_rt[(df_rt['rating'] >= Q1 - outlier_threshold * IQR) & (df_rt['rating'] <= Q3 + outlier_threshold * IQR)]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_rt['rating_normalizada'] = scaler.fit_transform(df_rt[['rating']])

# Guardar los datos limpios en nuevos archivos CSV
df_rt.to_csv('ratingsc.csv', index=False)