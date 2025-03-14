# Image Search API with CLIP and Qdrant Cloud

This FastAPI server provides endpoints for uploading images and performing semantic similarity search using OpenAI's CLIP model and Qdrant Cloud vector database.

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
   - These images will be automatically uploaded to Qdrant when starting the server for the first time

4. Start the server:

```bash
python main.py
```

The server will run at `http://localhost:8000`

When starting the server for the first time:

- A new collection will be created in Qdrant Cloud
- All images from the `images` directory will be automatically processed and uploaded
- Progress information will be printed to the console

## API Endpoints

### Upload Images

- **URL**: `/upsert`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**: Form data with one or more image files
- **Response**: List of processed images with status

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

### Upload Additional Images

```bash
curl -X POST -F "files=@/path/to/image1.jpg" -F "files=@/path/to/image2.jpg" http://localhost:8000/upsert
```

### Search Images

```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "a red car"}' http://localhost:8000/search
```

## Security Note

Make sure to keep your `.env` file secure and never commit it to version control. Add `.env` to your `.gitignore` file.
