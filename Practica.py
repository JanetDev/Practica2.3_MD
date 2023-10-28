import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import warnings
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings('ignore')


df_tmdb = pd.read_csv('Data/tmdb_5000_movies.csv')
df_imdb = pd.read_csv('Data/IMDB Top 250 Movies.csv')

df_tmdb.head(2)

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

# Para el segundo dataset

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

df_imdb['comb'] = df_imdb['directors'] +' '+ df_imdb['genre']

df_cache= df_imdb[['name', 'tagline', 'genre']]
df_cache.to_csv('df_cache.csv', index=False)

# Datos limpios

# Guardar los datos limpios en nuevos archivos CSV
df_tmdb.to_csv('Data/tmdb_clean.csv', index=False)
df_imdb.to_csv('Data/imdb_clean.csv', index=False)
df_cache.to_csv('Data/cache_data.csv', index=False)
#Modelo de desarrollo
#Recomendador basado en popularidad
p_df = df_imdb[['name', 'rating']]
p_df.head(5)

# Finalmente, ordenemos el DataFrame según la puntuación de la popularidad
popular_movies = p_df.sort_values(by='rating',ascending=False)

popular_movies.head(10)
# Recomendador basado en contenido
# concatenar todas estas columnas y crear una columna separada para ellas
df_imdb['Caracteristicas'] = df_imdb['directors'] +' '+ df_imdb['genre']
df_imdb['Caracteristicas'].head(5)

# Crear matriz de recuento a partir de esta nueva columna combinada

cv = CountVectorizer()
count_matrix = cv.fit_transform(df_imdb['Caracteristicas'])

# Ahora calcula la similitud del coseno según count_matrix
cosine_sim = cosine_similarity(count_matrix)
# Esta función toma el título de la película como entrada y devuelve las 5 películas más similares.
# Esta función también es capaz de corregir la busqueda ingresado por el usuario.

def get_recomandation_contentBase(title):
    
   # Convertir mayúsculas a minúsculas
    title = title.lower()
    
    # Corrección del busqueda de entrada del usuario (coincidencia cercana de nuestra lista de películas)
    title = get_close_matches(title, df_imdb['name'].values, n=3, cutoff=0.6)[0]
    
    # Obtener el índice de la película que coincide con el título.
    idx = df_imdb['name'][df_imdb['name']==title].index[0]
    
    # Obtenemos las puntuaciones de similitud de pares de todas las películas con esa película
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordena las películas según las puntuaciones de similitud.
    sim_scores = sorted(sim_scores, key=lambda x:x[1],reverse=True)
    
    # Obtén las puntuaciones de las 15 películas más similares.
    sim_scores = sim_scores[0:6]
    
    for i in sim_scores:
        movie_index = i[0]
        print(df_imdb['name'].iloc[movie_index])


# Ahora hagamos predicciones.
get_recomandation_contentBase("The Shawshank Redemption")

# Filtración colaborativa
data = df_imdb[['name', 'rating', 'genre']]
data.head(5)
