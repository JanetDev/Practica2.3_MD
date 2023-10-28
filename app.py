# Importing the necessary libraries
from flask import Flask, request, render_template
from flask_cors import cross_origin
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from tmdbv3api import TMDb, Movie
import requests

# Initialising flask app

app = Flask(__name__, static_url_path='/static')

# load the data
df = pd.read_csv('Data/imdb_clean.csv')
# load cache data
df_cache = pd.read_csv('Data/cache_data.csv')
# storing movie title into list
movie_list = list(df['name'])



# creating TMDB Api Object
tmdb = TMDb()
tmdb.api_key = 'e955d66146c91573e52a09a5566459d4'


def get_poster_link(title_list):
    tmdb_movie = Movie()
    dic_data = {"name": [], "genre": [], "tagline": []}

    for title in title_list:
        r_df = df_cache[df_cache['name'] == title]
        try:
            if len(r_df) >= 1:
                dic_data["name"].append(r_df['name'].values[0])
                dic_data["genre"].append(r_df['genre'].values[0])
                dic_data["tagline"].append(r_df['tagline'].values[0])
            else:
                result = tmdb_movie.search(title)
                movie_id = result[0].id
                response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id))
                data_json = response.json()

                movie_title = data_json['title']
                movie_genre = data_json['genres'][0]['name']
                movie_tag_line = data_json['tagline']

                dic_data['name'].append(movie_title)
                dic_data['genre'].append(movie_genre)
                dic_data['tagline'].append(movie_tag_line)
        except Exception as e:
            print(str(e))
            pass
    
    print(dic_data)

    return dic_data



@app.route('/', methods=['GET'])  # ruta para mostrar la página de inicio
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET']) # ruta para mostrar la recomendación en la interfaz de usuario web
@cross_origin()
def recommendation():
    if request.method == 'POST':
        try:
            # Lectura de las entradas dadas por el usuario.
            title = request.form['search']
            title = title.lower()
            # crear una matriz de recuento a partir de esta nueva columna combinada
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(df['tagline'])

            # calculamos la similitud del coseno
            cosine_sim = cosine_similarity(count_matrix)

            #Corregimos la busqueda de entrada del usuario (coincidencia cercana de nuestra lista de películas)
            correct_title = get_close_matches(title, movie_list, n=3, cutoff=0.6)[0]

            # obtener el valor del índice del título de la película dada
            idx = df['name'][df['name'] == correct_title].index[0]

            # obtener las puntuaciones de similitud por pares de todas las películas con esa película
            sim_score = list(enumerate(cosine_sim[idx]))

            # ordenar la película según puntuaciones de similitud
            sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)[0:15]

            #Las películas sugeridas se almacenan en una lista
            suggested_movie_list = []
            for i in sim_score:
                movie_index = i[0]
                suggested_movie_list.append(df['name'][movie_index])

            # llamando a la función get_poster_link para obtener su título y enlace del cartel.
            poster_title_link = get_poster_link(suggested_movie_list)
            return render_template('Recomendacion.html', output=poster_title_link)

        except Exception as e:
            print(str(e))
            return render_template("error.html")


if __name__ == '__main__':
    print("App is running")
    app.run(debug=True)
