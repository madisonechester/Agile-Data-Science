from pydantic import BaseModel
from typing import List, Optional


class Movie(BaseModel):
    movie_id: int
    title: Optional[str] = ""
    release_date: Optional[int] = 0
    video_release_date: Optional[int] = 0
    imdb_url: Optional[str] = ""
    genres: Optional[List[str]] = []

class Rating(BaseModel):
    movie_id: int
    rating: int
    unix_timestamp: int

class User(BaseModel):
    name: Optional[str] = "Anonymous"
    age: Optional[int] = 0
    occupation: Optional[str] = "Not Available"
    zip_code: Optional[int] = 0
    user_id: int
    ratings: Optional[List[Rating]] = []
