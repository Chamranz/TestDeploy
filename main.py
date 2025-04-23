from fastapi import FastAPI, HTTPException, Query
from recommender import load_model, recommend_words
from schemas import RecommendationResponse
from typing import List

app = FastAPI(
    title="Words in Politic API",
    description="Узнай слова, политически связанные с вашим словом.",
    version="1.0.0",
    contact={
        "name": "Your Name",  # Используйте общее имя или оставьте placeholder
        "email": "your_email@example.com"  # Общий email
    }
)

model = load_model()

@app.get("/", tags=["Health Check"])
def read_root():
    return {"message": "🎶 Poltic associations is running!"}

@app.get(
    "/api/recommend/{track_title}",
    response_model=RecommendationResponse,
    summary="Получить связанные слова",
    description="Возвращает список слов, связанных с введеным, используя косинусное сходство по признакам.",
    tags=["Recommendations"]
)
def get_recommendations(
    track_title: str,
    N: int = Query(5, alias="n", ge=1, le=20, description="Количество выдаваемых слов (от 1 до 20)")
):
    recommendations = recommend_words(model, track_title, N)
    
    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail=f"Слова '{track_title}' нет в базе данных."
        )
    
    return {"requested_track": track_title, "recommendations": recommendations}