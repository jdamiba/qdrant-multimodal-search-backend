# Multi-modal Image Search API with CLIP and Qdrant Cloud

This FastAPI server provides an endpoint for performing semantic similarity search using OpenAI's CLIP model and Qdrant Cloud vector database.

## Prerequisites

- Python 3.8+
- Qdrant Cloud account (sign up at https://cloud.qdrant.io/)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your Qdrant Cloud credentials:

   - Create an account at https://cloud.qdrant.io/
   - Create a new cluster
   - Copy your cluster URL and API key
   - Create a `.env` file in the project root with the following content:

   ```
   QDRANT_URL=your-cluster-url.qdrant.tech
   QDRANT_API_KEY=your-api-key
   ```

3. Prepare your images:

   - Create an `images` directory in the project root
   - Place your image files (`.jpg`, `.jpeg`, or `.png`) in this directory
   - Run `python embed_and_upsert.py` to create embeddings of your images and upsert them into a Qdrant collection.

4. Start the server:

```bash
python main.py
```

The server will run at `http://localhost:8000`

## API Endpoints

### Search Images

- **URL**: `/search`
- **Method**: `POST`
- **Body**:

```json
{
    "query": "your search text here",
    "limit": 5 (optional, default=5)
}
```

- **Response**: List of matching images with similarity scores

## Example Usage

### Search Images

```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "a red car"}' http://localhost:8000/search
```

## Deployment

This FastAPI server can be deployed to Render, making it accessible to a deployed frontend application. Instructions for how to deploy a FastAPI app on Render can be found at https://render.com/docs/deploy-fastapi?utm_campaign=fastapi&utm_medium=referral&utm_source=deploydoc.

## Security Note

Make sure to keep your `.env` file secure and never commit it to version control. Add `.env` to your `.gitignore` file.
