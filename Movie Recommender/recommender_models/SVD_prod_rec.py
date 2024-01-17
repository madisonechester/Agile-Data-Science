# SVD MODEL FOR PRODUCT-BASED RECOMMENDATION 

# !pip install surprise python-dotenv azure-ai-ml mlflow azureml-mlflow

import warnings
warnings.filterwarnings("ignore")

import json
import requests
import mlflow
import os
import zipfile
import numpy as np
import pandas as pd 
from surprise import SVD
from surprise import accuracy
from surprise import Reader, Dataset
from urllib.request import urlretrieve
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split
from mlflow.tracking import MlflowClient
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
load_dotenv()

# Set Azure variables and set mlflow tracking uri to Azure
if 'AZURE_CLIENT_ID' not in os.environ:
  raise Exception("It seems that Azure environment variables are not set. Please ask alex to provide you with the .env file.")

subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
workspace = os.environ.get("AZURE_WORKSPACE")

# Model parameters
EXPERIMENT_NAME = "SVD-Product-Based-Recommendation-Experiment"
MODEL_NAME = "SVD-Product-Based-Recommendation-Model"
ARTIFACT_PATH = "artifact-path-model"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

azureml_tracking_uri = ml_client.workspaces.get(
    ml_client.workspace_name
).mlflow_tracking_uri
mlflow.set_tracking_uri(azureml_tracking_uri)

###############################################################################

# download MovieLens data
urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", "movielens.zip")
zip_ref = zipfile.ZipFile('movielens.zip', "r")
zip_ref.extractall()
print("Dataset contains:")
print(zip_ref.read('ml-100k/u.info'))

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

# grid search for the best parameters
#param_grid = {'n_factors':[40, 60],
 #             'n_epochs': [20, 40, 60],
  #            'lr_all': [0.002, 0.005],
   #           'reg_all': [0.1, 0.4, 0.6]}
#gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

# {'n_factors': 40, 'n_epochs': 60, 'lr_all': 0.005, 'reg_all': 0.1}

#gs.fit(data)
# best RMSE score
#print(gs.best_score['rmse'])
# combination of parameters that gave the best RMSE score
#print(gs.best_params['rmse'])

# Initialize a MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()
with mlflow.start_run() as run:
  # run the SVD model
  #algo = gs.best_estimator['rmse']
  params = {
      "n_factors": 40,
      "n_epochs": 60,
      "lr_all": 0.005,
      "reg_all": 0.1,
  }
  algo = SVD(**params)
  algo.fit(trainset)
  test_pred = algo.test(testset)
  acc = accuracy.rmse(test_pred)

  # log parameters and metrics
  mlflow.log_params(params)
  mlflow.log_metric("test_rmse", acc)
  mlflow.sklearn.log_model(
      sk_model=algo,
      artifact_path=ARTIFACT_PATH,
      registered_model_name=MODEL_NAME, # Note: This will create a new registered model if it doesn't exist
  )
  mlflow.end_run()

# list of all users
unique_users = df['user_id'].unique()
# list of all movies
unique_movies = df['title'].unique()

len(unique_movies),len(unique_users)

users_list = df['user_id'].tolist()
movie_list = df['title'].tolist()
len(users_list),len(movie_list)

ratings_list = df['rating'].tolist()
len(ratings_list)

movies_dict = {unique_movies[i] : i for i in range(len(unique_movies))}

# creating a utility matrix for the available data
# creating an empty array with (number of rows = number of movies) and (number of columns = number of users)
# rows as movies, columns as users

utility_matrix = np.asarray([[np.nan for j in range(len(unique_users))] for i in range(len(unique_movies))])
print("Shape of Utility matrix: ",utility_matrix.shape)

for i in range(len(ratings_list)):

  # ith entry in users list and subtract 1 to get the index, we do the same for movies but we already defined a dictionary to get the index
  utility_matrix[movies_dict[movie_list[i]]][users_list[i]-1] = ratings_list[i]

utility_matrix

mask = np.isnan(utility_matrix)
masked_arr = np.ma.masked_array(utility_matrix, mask)
temp_mask = masked_arr.T
rating_means = np.mean(temp_mask, axis=0)

filled_matrix = temp_mask.filled(rating_means)
filled_matrix = filled_matrix.T
filled_matrix = filled_matrix - rating_means.data[:,np.newaxis]

filled_matrix = filled_matrix.T / np.sqrt(len(movies_dict)-1)
filled_matrix

filled_matrix.shape

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
  print(" ")
  print("Top",top_n,"movies which are very much similar to the Movie-",movie_name, "are: ")
  print(" ")
  for i in indexes[0:]:
    print(unique_movies[i])
    
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

class invalid(Exception):
    pass

def recommender():

    try:
      movie_name = input("Enter the Movie name: ")
      movie_name_lower = movie_name.lower()
      if movie_name_lower not in case_insensitive_movies_list :
        raise invalid
      else :
        try:
            num_recom = int(input("Enter Number of movie recommendations needed: ").strip())
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        get_similar_movies(unique_movies[case_insensitive_movies_list.index(movie_name_lower)],num_recom)

    except invalid:
      possible_movies = get_possible_movies(movie_name_lower)

      if len(possible_movies) == len(unique_movies) :
        print("Movie name entered is does not exist in the list ")
      else :
        indices = [case_insensitive_movies_list.index(i) for i in possible_movies]
        print("Entered Movie name is not matching with any movie from the dataset . Please check the below suggestions :\n",[unique_movies[i] for i in indices])
        print("")
        recommender()
        
recommender()