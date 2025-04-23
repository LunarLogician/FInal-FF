import openai
import os
from typing import List

async def getEmbedding(text: str) -> List[float]:
    try:
        # Initialize the client with the API key
        client = openai.AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create embedding using text-embedding-3-small model
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        
        # Return the embedding vector
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        raise

def splitTextIntoChunks(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for the space
        if current_size + word_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks 