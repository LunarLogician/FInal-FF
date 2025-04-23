from typing import List, Dict, Optional
from pydantic import BaseModel
from pinecone import Pinecone
import os
from .embeddings import getEmbedding

class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict

# Initialize Pinecone client once
pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
)

async def storeDocument(text: str, metadata: Optional[Dict] = None) -> int:
    try:
        # Get the index
        index = pc.Index(os.getenv('PINECONE_INDEX', 'embed-upload'))
        
        # Create embedding
        embedding = await getEmbedding(text)
        
        # Prepare vector
        vector = {
            'id': metadata.get('documentId', str(hash(text))),
            'values': embedding,
            'metadata': {
                **(metadata or {}),
                'text': text
            }
        }
        
        # Upload to Pinecone
        index.upsert(vectors=[vector])
        
        return 1
    except Exception as e:
        print(f"Error storing document: {e}")
        raise

async def searchDocument(query: str, documentId: Optional[str] = None) -> List[DocumentChunk]:
    try:
        # Get the index
        index = pc.Index(os.getenv('PINECONE_INDEX', 'embed-upload'))
        
        # Create query embedding
        query_embedding = await getEmbedding(query)
        
        # Search in Pinecone
        search_response = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True,
            filter={'documentId': documentId} if documentId else None
        )
        
        # Format results
        results = []
        for match in search_response.matches:
            chunk = DocumentChunk(
                id=match.id,
                text=match.metadata.get('text', ''),
                metadata={
                    'documentId': match.metadata.get('documentId', ''),
                    'fileName': match.metadata.get('fileName', ''),
                    'chunkIndex': match.metadata.get('chunkIndex', 0)
                }
            )
            results.append(chunk)
        
        return results
    except Exception as e:
        print(f"Error searching document: {e}")
        raise 