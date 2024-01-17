from fastapi import FastAPI
from models import User, Movie
from typing import List
from dotenv import load_dotenv
load_dotenv()
import warnings
from ddb_management import DDB_manager
from recommender_sys.recommender_systems import recommender_movie, recommender_user
warnings.filterwarnings("ignore")
app = FastAPI()
ddb_manager = DDB_manager()

@app.get("/")
async def root():
    return {"message": "Recommender API"}

@app.post("/user/recommend/{num_recommendations}")
async def recommend_for_user(user: User, num_recommendations:int):
    return recommender_user(user.user_id, num_recommendations)

@app.post("/user/bulk/{num_recommendations}")
async def recommendation_bulk(users: List[User], num_recommendations:int):
    tmp = []
    for user in users:
        tmp.append(recommender_user(user.user_id, num_recommendations))
    return tmp

@app.post("/movie/recommend/{num_recommendations}")
async def recommend_for_movie(movie: Movie,num_recommendations:int):
    return recommender_movie(movie.movie_id,num_recommendations).tolist()

@app.post("/movie/bulk/{num_recommendations}")
async def recommendation_bulk_movies(movies: List[Movie],num_recommendations:int):
    tmp = []
    for movie in movies:
        tmp.append(recommender_movie(movie.movie_id,num_recommendations).tolist())
    return tmp

@app.get("/movies/")
async def get_movies():
    return ddb_manager.get_unique_movies()