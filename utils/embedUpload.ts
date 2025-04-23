import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

const PINECONE_API_KEY = process.env.PINECONE_API_KEY!;
const PINECONE_INDEX = process.env.PINECONE_INDEX || 'embed-upload';

export async function embedAndUpload(text: string, fileName: string) {
  // Initialize Pinecone client
  const pinecone = new Pinecone({ 
    apiKey: PINECONE_API_KEY,
  });

  const index = pinecone.Index(PINECONE_INDEX);

  // Split text into chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const docs = await splitter.createDocuments([text]);

  // Create embeddings
  const embeddings = new OpenAIEmbeddings();
  
  // Process chunks and upload to Pinecone
  const vectors = await Promise.all(
    docs.map(async (doc, i) => {
      const embedding = await embeddings.embedQuery(doc.pageContent);
      return {
        id: `${fileName}-${i}`,
        values: embedding,
        metadata: {
          text: doc.pageContent,
          fileName: fileName,
          chunk_index: i,
        },
      };
    })
  );

  // Upsert vectors to Pinecone
  await index.upsert(vectors);

  return vectors.length;
}

export async function searchDocument(query: string, fileName?: string) {
  const pinecone = new Pinecone({ 
    apiKey: PINECONE_API_KEY,
  });

  const index = pinecone.Index(PINECONE_INDEX);
  const embeddings = new OpenAIEmbeddings();
  
  const queryEmbedding = await embeddings.embedQuery(query);
  
  const searchResponse = await index.query({
    vector: queryEmbedding,
    topK: 5,
    includeMetadata: true,
    filter: fileName ? { fileName: fileName } : undefined
  });

  return searchResponse.matches.map(match => ({
    text: match.metadata?.text,
    score: match.score,
    fileName: match.metadata?.fileName,
    chunkIndex: match.metadata?.chunk_index,
  }));
}
