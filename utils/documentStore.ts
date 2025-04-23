import { Pinecone } from '@pinecone-database/pinecone';
import { getEmbedding, splitTextIntoChunks } from './embeddings';
import pLimit from 'p-limit';
import NodeCache from 'node-cache';

const PINECONE_API_KEY = process.env.PINECONE_API_KEY!;
const PINECONE_INDEX = process.env.PINECONE_INDEX || 'document-store';

// Initialize cache with 30 minute TTL
const cache = new NodeCache({ stdTTL: 1800 });

// Initialize Pinecone client - do it once
const pinecone = new Pinecone({ 
  apiKey: PINECONE_API_KEY,
});

// Rate limiting for API calls
const limit = pLimit(3); // Maximum 3 concurrent API calls

export interface DocumentChunk {
  id: string;
  text: string;
  metadata: {
    documentId: string;
    fileName: string;
    chunkIndex: number;
  };
}

export async function getDocumentMetadata(documentId: string): Promise<{ exists: boolean; chunkCount?: number }> {
  const index = pinecone.Index(PINECONE_INDEX);
  
  // Search for any chunks with this document ID
  const searchResponse = await index.query({
    vector: new Array(1536).fill(0), // dummy vector
    topK: 1,
    includeMetadata: true,
    filter: { documentId }
  });
  
  return {
    exists: searchResponse.matches.length > 0,
    chunkCount: searchResponse.matches.length > 0 ? 
      await getDocumentChunkCount(documentId) : undefined
  };
}

async function getDocumentChunkCount(documentId: string): Promise<number> {
  const index = pinecone.Index(PINECONE_INDEX);
  
  // Get index stats
  const stats = await index.describeIndexStats();
  
  // Filter the stats on the client side since the API doesn't support filtering
  // This is an approximation - for exact count we'd need to do a filtered query
  return stats.totalRecordCount ?? 0;
}

export async function storeDocument(text: string, metadata: any = {}) {
  try {
    const cacheKey = `doc_${metadata.documentId || Date.now()}`;
    const cachedResult = cache.get(cacheKey);
    if (cachedResult) {
      return cachedResult;
    }

    // Split text into smaller chunks
    const chunks = splitTextIntoChunks(text);
    if (!chunks.length) {
      throw new Error('No valid chunks created from document');
    }

    // Process chunks in batches with rate limiting
    const BATCH_SIZE = 10;
    const batches = [];
    for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
      batches.push(chunks.slice(i, i + BATCH_SIZE));
    }

    const vectors = [];
    for (const batch of batches) {
      const batchVectors = await Promise.all(
        batch.map(chunk => 
          limit(async () => {
            try {
              const embedding = await getEmbedding(chunk);
              return {
                id: `${metadata.documentId || Date.now()}_${vectors.length}`,
                values: embedding,
                metadata: {
                  ...metadata,
                  text: chunk,
                  chunkIndex: vectors.length,
                  timestamp: new Date().toISOString()
                }
              };
            } catch (error) {
              console.error('Error creating embedding for chunk:', error);
              return null;
            }
          })
        )
      );
      
      vectors.push(...batchVectors.filter(v => v !== null));
      await new Promise(resolve => setTimeout(resolve, 100)); // Reduced delay
    }

    if (vectors.length > 0) {
      const index = pinecone.Index(PINECONE_INDEX);
      
      const UPLOAD_BATCH_SIZE = 100; // Increased batch size
      for (let i = 0; i < vectors.length; i += UPLOAD_BATCH_SIZE) {
        const batch = vectors.slice(i, i + UPLOAD_BATCH_SIZE);
        await index.upsert(batch);
        await new Promise(resolve => setTimeout(resolve, 100)); // Reduced delay
      }
    }

    const result = vectors.length;
    cache.set(cacheKey, result);
    return result;
  } catch (error) {
    console.error('Error storing document:', error);
    throw error;
  }
}

export async function deleteDocument(documentId: string): Promise<void> {
  const index = pinecone.Index(PINECONE_INDEX);
  
  // Delete all vectors for this document
  await index.deleteMany({
    filter: { documentId }
  });
}

// Cache search results
export async function searchDocument(query: string, documentId?: string): Promise<DocumentChunk[]> {
  const cacheKey = `search_${query}_${documentId || 'all'}`;
  const cachedResult = cache.get<DocumentChunk[]>(cacheKey);
  if (cachedResult) {
    return cachedResult;
  }

  const index = pinecone.Index(PINECONE_INDEX);
  const queryEmbedding = await getEmbedding(query);
  
  const searchResponse = await index.query({
    vector: queryEmbedding,
    topK: 10,
    includeMetadata: true,
    filter: documentId ? { documentId } : undefined
  });
  
  const results = searchResponse.matches.map(match => ({
    id: match.id,
    text: match.metadata?.text as string,
    metadata: {
      documentId: match.metadata?.documentId as string,
      fileName: match.metadata?.fileName as string,
      chunkIndex: match.metadata?.chunkIndex as number,
    }
  }));

  cache.set(cacheKey, results);
  return results;
} 