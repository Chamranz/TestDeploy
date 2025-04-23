from pydantic import BaseModel, Field
from typing import List

class Song(BaseModel):
    entry_word: str = Field(..., example="Time")
    words_predict: str = Field(..., example="Pink Floyd")

class RecommendationResponse(BaseModel):
    requested_track: str = Field(..., example="Time")
    recommendations: List[Song]

    