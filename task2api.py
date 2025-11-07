import httpx
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import ORJSONResponse, JSONResponse 
from pydantic import BaseModel, Field
from typing import List
from task1 import findpricegappair

# ===================================================
# CONFIGURATION
# ===================================================
OMDB_API_KEY = "ee55d25a"  # ✅ Your working OMDb API key
OMDB_BASE_URL = "http://www.omdbapi.com/"  # ✅ Correct OMDb base URL

app = FastAPI(
    title="Price Gap Pair & Movie API",
    version="1.0.0",
    default_response_class=ORJSONResponse  # ✅ Enables clean JSON output
)

# ===================================================
# DATA MODELS
# ===================================================
class PriceGapInput(BaseModel):
    nums: List[int] = Field(..., description="List of integers")
    k: int = Field(..., ge=0, description="Non-negative integer")


# ===================================================
# ENDPOINT 1: PRICE GAP PAIR (Task 1 Logic)
# ===================================================
@app.post("/api/price-gap-pair", response_class=ORJSONResponse)
async def price_gap_pair(data: PriceGapInput):
    
    try:
        result = findpricegappair(data.nums, data.k)
        if result is None:
            return {"pair": None, "values": None}

        i, j = result
        # ✅ Cleanly formatted JSON output
        return ORJSONResponse(
            content={
                "pair": [i, j],
                "values": [data.nums[i], data.nums[j]]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad Request: {str(e)}")


@app.get("/api/movies")
async def get_movies(
    q: str = Query(..., description="Movie title keyword"),
    page: int = Query(1, ge=1, description="Page number (≥1)"),
):
    params = {"s": q, "page": page, "apikey": OMDB_API_KEY}

    try:
        response = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="OMDb service error")

        data = response.json()

        if data.get("Response") == "False":
            # Gracefully handle no results
            return JSONResponse(content={
                "page": page,
                "total_pages": 0,
                "total_results": 0,
                "movies": []
            })

        total_results = int(data.get("totalResults", 0))
        total_pages = (total_results + 9) // 10

        movie_list = []
        for item in data.get("Search", []):
            imdb_id = item.get("imdbID")
            movie_details = requests.get(
                OMDB_BASE_URL, params={"i": imdb_id, "apikey": OMDB_API_KEY}
            ).json()

            # Skip if movie details are invalid
            if movie_details.get("Response") == "False":
                continue

            movie_list.append({
                "title": movie_details.get("Title"),
                "director": movie_details.get("Director")
            })

        return JSONResponse(content={
            "page": page,
            "total_pages": total_pages,
            "total_results": total_results,
            "movies": movie_list
        })

    except requests.exceptions.RequestException:
        raise HTTPException(status_code=502, detail="OMDb service error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")