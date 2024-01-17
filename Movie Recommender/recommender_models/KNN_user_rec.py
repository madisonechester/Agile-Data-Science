# KNN MODEL FOR USER-BASED RECOMMENDATION

# !pip install surprise 

import warnings
warnings.filterwarnings("ignore")

import json
import requests
import mlflow
import os
import zipfile
import numpy as np
import pandas as pd 
from surprise import KNNBasic
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
EXPERIMENT_NAME = "KNN-User-Based-Recommendation-Experiment"
MODEL_NAME = "KNN-User-Based-Recommendation-Model"
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

# load data into a Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)

# split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize a MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()
with mlflow.start_run() as run:
  params = {
    'k': 40
  }
  algo = KNNBasic(
    k=params['k'],
    sim_options={'name': 'pearson_baseline', 'user_based': False})
  algo.fit(trainset)

  # run the trained model against the test-set
  test_pred = algo.test(testset)

  # get RMSE
  acc = accuracy.rmse(test_pred, verbose=True)
  
  # log parameters and metrics
  mlflow.log_params(params)
  mlflow.log_params(sim_options)
  mlflow.log_metric("test_rmse", acc)
  mlflow.sklearn.log_model(
      sk_model=algo,
      artifact_path=ARTIFACT_PATH,
      registered_model_name=MODEL_NAME, # Note: This will create a new registered model if it doesn't exist
  )
  mlflow.end_run()


# input user ID
user_id = input("Enter your user ID: ")

# input number of recommendations
num_recommendations = int(input("Enter Number of movie recommendations needed: ").strip())

# make predictions for the specified user
user_predictions = [algo.predict(user_id, item_id, verbose=False) for item_id in df['title']]

# sort the predictions by the estimated ratings
user_predictions.sort(key=lambda x: x.est, reverse=True)

# recommend the top N items
top_N = num_recommendations
recommended_items = user_predictions[:top_N]

# print recommendations with similarity scores
print(f"Top {num_recommendations} recommendations for user {user_id}:")

for prediction in recommended_items:
    item_title = prediction.iid
    similarity_score = prediction.est
    print(f"Recommendation: '{item_title}' with a similarity score of {similarity_score:.2f}")
