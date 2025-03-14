import os
from typing import List
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastembed import ImageEmbedding
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class SearchRequest(BaseModel):
    query: str
    limit: int = 5  # Default value of 5

# Initialize FastEmbed models
image_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
text_model = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")

# Initialize Qdrant client with cloud credentials
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
COLLECTION_NAME = "house_images"

# Verify Qdrant connection at startup
try:
    collections = qdrant_client.get_collections()
    print(f"Successfully connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
    if COLLECTION_NAME not in [c.name for c in collections.collections]:
        print(f"Warning: Collection '{COLLECTION_NAME}' not found!")
except Exception as e:
    print(f"Error connecting to Qdrant: {str(e)}")
    print(f"QDRANT_URL: {os.getenv('QDRANT_URL')}")
    # Don't print the actual API key for security
    print(f"QDRANT_API_KEY present: {'Yes' if os.getenv('QDRANT_API_KEY') else 'No'}")

def get_text_embedding(text: str) -> np.ndarray:
    """Generate FastEmbed embedding for text."""
    try:
        # Use FastEmbed to generate text embedding
        embeddings = list(text_model.embed([text]))
        return embeddings[0]
    except Exception as e:
        print(f"Error in get_text_embedding: {str(e)}")
        raise

@app.post("/search")
async def search_images(request: SearchRequest):
    """Search for similar images using text query."""
    try:
        # Generate text embedding
        query_embedding = get_text_embedding(request.query)
        
        print(f"Generated embedding for query: {request.query}")
        print(f"Embedding shape: {query_embedding.shape}")
        
        # Search in Qdrant
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=request.limit
        )
        
        print(f"Found {len(search_results.points)} results")
        
        # Format results
        results = [
            {
                "filename": hit.payload["filename"],
                "score": hit.score
            }
            for hit in search_results.points
        ]
        
        return {"results": results}
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        # Check if collection exists
        try:
            collections = qdrant_client.get_collections()
            print(f"Available collections: {[c.name for c in collections.collections]}")
            if COLLECTION_NAME not in [c.name for c in collections.collections]:
                return HTTPException(
                    status_code=404, 
                    detail=f"Collection '{COLLECTION_NAME}' not found. Available collections: {[c.name for c in collections.collections]}"
                )
        except Exception as ce:
            print(f"Error checking collections: {str(ce)}")

        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 