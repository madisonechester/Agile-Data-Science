# SVD MODEL FOR PRODUCT-BASED RECOMMENDATION 

# !pip install surprise 

import warnings
warnings.filterwarnings("ignore")

import zipfile
import numpy as np
import pandas as pd 
from surprise import SVD
from surprise import accuracy
from surprise import Reader, Dataset
from urllib.request import urlretrieve
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split
from recommender_flow import get_last_model_movie, get_last_model_user

# download MovieLens data
urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", "movielens.zip")
zip_ref = zipfile.ZipFile('movielens.zip', "r")
zip_ref.extractall()


# load users and ratings dataset
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

# load movies dataset
genre_cols = ["genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
              "Fantasy", "Film-Noir", "Horror","Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols

movies = pd.read_csv('ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

# movies per genre
genre_occurences = movies[genre_cols].sum().to_dict()

# some movies can belong to more than one genre
# all_genres: all the active genres of the movie
# genre: randomly sampled from the active genres

def mark_genres(movies, genres):
  def get_random_genre(gs):
    active = [genre for genre, g in zip(genres, gs) if g==1]
    if len(active) == 0:
      return 'Other'
    return np.random.choice(active)
  def get_all_genres(gs):
    active = [genre for genre, g in zip(genres, gs) if g==1]
    if len(active) == 0:
      return 'Other'
    return '-'.join(active)
  movies['genre'] = [get_random_genre(gs) for gs in zip(*[movies[genre] for genre in genres])]
  movies['all_genres'] = [get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genres])]

mark_genres(movies, genre_cols)

# merged DataFrame containing all the movielens data
movielens = ratings.merge(movies, on='movie_id').merge(users, on='user_id')

# split the data into training and test sets
def split_dataframe(df, holdout_fraction=0.2):
  test = df.sample(frac=holdout_fraction, replace=False)
  train = df[~df.index.isin(test.index)]
  return train, test

df = movielens[['user_id','title','rating']]
df['user_id'] = df['user_id'].astype(int)

# data preparations
# set rating_scale parameter
reader = Reader(rating_scale=(1, 5))

# the columns must correspond to user, product, and ratings
data = Dataset.load_from_df(df[['user_id', 'title', 'rating']], reader)

#train test split
trainset, testset = train_test_split(data, test_size = 0.2)

algo = get_last_model_movie()

# list of all users
unique_users = df['user_id'].unique()
# list of all movies
unique_movies = df['title'].unique()

users_list = df['user_id'].tolist()
movie_list = df['title'].tolist()

ratings_list = df['rating'].tolist()

movies_dict = {unique_movies[i] : i for i in range(len(unique_movies))}

# creating a utility matrix for the available data
# creating an empty array with (number of rows = number of movies) and (number of columns = number of users)
# rows as movies, columns as users

utility_matrix = np.asarray([[np.nan for j in range(len(unique_users))] for i in range(len(unique_movies))])

for i in range(len(ratings_list)):

  # ith entry in users list and subtract 1 to get the index, we do the same for movies but we already defined a dictionary to get the index
  utility_matrix[movies_dict[movie_list[i]]][users_list[i]-1] = ratings_list[i]


mask = np.isnan(utility_matrix)
masked_arr = np.ma.masked_array(utility_matrix, mask)
temp_mask = masked_arr.T
rating_means = np.mean(temp_mask, axis=0)

filled_matrix = temp_mask.filled(rating_means)
filled_matrix = filled_matrix.T
filled_matrix = filled_matrix - rating_means.data[:,np.newaxis]

filled_matrix = filled_matrix.T / np.sqrt(len(movies_dict)-1)

# computing the SVD of the input matrix
U, S, V = np.linalg.svd(filled_matrix)

case_insensitive_movies_list = [i.lower() for i in unique_movies]

# function to calculate the cosine similarity (sorting by most similar and returning the top N)
def top_cosine_similarity(data, movie_id, top_n=10):
  index = movie_id
  movie_row = data[index, :]
  magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
  similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
  sort_indexes = np.argsort(-similarity)
  return sort_indexes[:top_n]

# k-principal components to represent movies, movie_id to find recommendations, top_n print n results
def get_similar_movies(movie_name,top_n,k = 50):
  # k = 50
  # movie_id = 1
  # top_n = 10

  sliced = V.T[:, :k]
  movie_id = movies_dict[movie_name]
  indexes = top_cosine_similarity(sliced, movie_id, top_n)
  return unique_movies[indexes]
    
# function which takes input and returns suggestions for the user
def get_possible_movies(movie):

    temp = ''
    possible_movies = case_insensitive_movies_list.copy()
    for i in movie :
      out = []
      temp += i
      for j in possible_movies:
        if temp in j:
          out.append(j)
      if len(out) == 0:
          return possible_movies
      out.sort()
      possible_movies = out.copy()

    return possible_movies

def recommender_movie(movie_id, num_recom):
    return get_similar_movies(unique_movies[movie_id],num_recom)

algo2 = get_last_model_user()

def recommender_user(user_id, num_recom):
  # make predictions for the specified user
  user_predictions = [algo2.predict(user_id, item_id) for item_id in unique_movies]
  # sort the predictions by the estimated ratings
  user_predictions.sort(key=lambda x: x.est, reverse=True)
  # recommend the top N items
  recommended_items = user_predictions[:num_recom]
  return np.array(recommended_items)[:,1].tolist()