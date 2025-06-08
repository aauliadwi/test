from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
import json

app = FastAPI()

# Load model user embedding
user_model = tf.keras.models.load_model("saved_model/user_embedding_model")

# Load item embeddings dari JSON
with open("app/item_embeddings.json", "r") as f:
    item_data = json.load(f)

# Request model untuk user-based
class UserRecommendationRequest(BaseModel):
    user_id: int

# Request model untuk content-based
class FoodRecommendationRequest(BaseModel):
    food_id: str
    title: str

# Response model
class RecommendedItem(BaseModel):
    food_id: str
    title: str
    score: Optional[float] = None

# 1. Content-based recommendation (replacing /recommend)
@app.post("/recommend", response_model=List[RecommendedItem])
async def recommend_by_food(request: FoodRecommendationRequest):
    input_food_id = request.food_id
    input_title = request.title

    # Cari makanan input
    input_item = next((item for item in item_data if item["food_id"] == input_food_id), None)
    if input_item is None:
        raise HTTPException(status_code=404, detail="food_id tidak ditemukan")

    if input_item["title"] != input_title:
        raise HTTPException(status_code=400, detail="title tidak cocok dengan food_id")

    input_embedding = np.array(input_item["embedding"])

    # Hitung similarity dengan semua item lain
    scores = []
    for item in item_data:
        if item["food_id"] == input_food_id:
            continue
        item_embedding = np.array(item["embedding"])
        similarity = np.dot(input_embedding, item_embedding) / (
            np.linalg.norm(input_embedding) * np.linalg.norm(item_embedding) + 1e-10
        )
        scores.append({
            "food_id": item["food_id"],
            "title": item["title"],
            "score": float(similarity)
        })

    top_items = sorted(scores, key=lambda x: x["score"], reverse=True)[:10]
    return top_items

# 2. User-based recommendation (pindah ke /recommend_by_user)
@app.post("/recommend_by_user", response_model=List[RecommendedItem])
async def recommend_by_user(request: UserRecommendationRequest):
    user_id = request.user_id
    user_embedding = user_model(tf.constant([user_id])).numpy()[0]

    scores = []
    for item in item_data:
        item_vec = np.array(item["embedding"])
        score = np.dot(user_embedding, item_vec)
        scores.append({
            "food_id": item["food_id"],
            "title": item["title"],
            "score": float(score)
        })

    top_items = sorted(scores, key=lambda x: x["score"], reverse=True)[:10]
    return top_items