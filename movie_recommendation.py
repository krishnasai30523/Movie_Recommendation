import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.read_csv("tmdb_5000_movies.csv")


movies = movies[['title', 'overview', 'genres']]
movies.dropna(inplace=True)
movies['content'] = movies['overview'] + " " + movies['genres']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def recommend_movie(movie_title, n=5):
    if movie_title not in movies['title'].values:
        return "Movie not found!"

    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]

    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]
print("Recommended movies:")
print(recommend_movie("Avatar"))
