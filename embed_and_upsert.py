import os
import torch
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
from fastembed import ImageEmbedding
import argparse

# Load environment variables
load_dotenv()

# Get Qdrant connection details from environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "house_images"

# Initialize FastEmbed model
model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")

def get_image_embedding(image_path):
    """Generate embedding for a single image using FastEmbed"""
    try:
        # FastEmbed handles the image loading and preprocessing
        embeddings = list(model.embed([image_path]))
        return embeddings[0]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def setup_qdrant():
    """Initialize Qdrant client and create collection if it doesn't exist"""
    client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    
    # Check if collection exists, if not create it
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=512,  # CLIP embedding size
                distance=models.Distance.COSINE
            )
        )
        print(f"Created new collection: {COLLECTION_NAME}")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")
    
    return client

def embed_and_upload_images(image_folder, batch_size=100):
    """Process all images in the folder and upload to Qdrant"""
    client = setup_qdrant()
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        image_files.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_embeddings = []
        batch_payloads = []
        
        for image_path in tqdm(batch_files, desc=f"Processing batch {i//batch_size + 1}"):
            embedding = get_image_embedding(image_path)
            if embedding is not None:
                # Create relative path for storage
                rel_path = os.path.relpath(image_path, start=os.path.dirname(image_folder))
                # Remove 'images/' prefix if it exists to avoid duplication
                if rel_path.startswith('images/'):
                    rel_path = rel_path[7:]
                    
                batch_embeddings.append(embedding)
                batch_payloads.append({
                    "image_path": rel_path,
                    "filename": os.path.basename(image_path)
                })
        
        if batch_embeddings:
            # Upload batch to Qdrant
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=models.Batch(
                    ids=list(range(i, i + len(batch_embeddings))),
                    vectors=batch_embeddings,
                    payloads=batch_payloads
                )
            )
            print(f"Uploaded batch of {len(batch_embeddings)} embeddings")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed images and upload to Qdrant")
    parser.add_argument("--image_folder", type=str, default="./images", 
                        help="Path to folder containing images")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for processing")
    
    args = parser.parse_args()
    
    embed_and_upload_images(args.image_folder, args.batch_size)
    print("Image embedding and uploading complete!") 