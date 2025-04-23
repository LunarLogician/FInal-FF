// app/api/embed/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { OpenAIEmbeddings } from '@langchain/openai';
import { PineconeStore } from '@langchain/community/vectorstores/pinecone'; // ✅ Correct now
import pinecone from '@/utils/pinecone';
import { v4 as uuidv4 } from 'uuid';
import { OpenAI } from 'openai';

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Function to get embeddings
async function getEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    input: text,
    model: "text-embedding-3-small"
  });
  return response.data[0].embedding;
}

interface EmbeddingResult {
  embedding: number[];
}

export async function POST(req: NextRequest) {
  const { text, namespace } = await req.json();

  if (!text || !namespace) {
    return NextResponse.json({ error: 'Missing text or namespace' }, { status: 400 });
  }

  try {
    const embeddings = new OpenAIEmbeddings({
      modelName: 'text-embedding-3-small',
      openAIApiKey: process.env.OPENAI_API_KEY,
    });    
    const index = pinecone.Index("embed-upload");

    // Define chunk size
    const chunkSize = 500;
    const chunks = text.match(new RegExp(`.{1,${chunkSize}}`, 'g')) || [];
    
    // Get embeddings for each chunk
    const chunkEmbeddings = await Promise.all(
      chunks.map(async (chunk: string, i: number) => {
        const embedding = await getEmbedding(chunk);
        return embedding;
      })
    );

    const docs = chunks.map((chunk: string, i: number) => ({
      pageContent: chunk,
      metadata: {
        id: uuidv4(),
        namespace,
        chunkIndex: i,
        page_number: i + 1,
      },
    }));

    console.log(`Uploading ${docs.length} chunks to Pinecone under namespace "${namespace}"...`);

    await PineconeStore.fromDocuments(docs, {
      embedDocuments: async (texts: string[]) => {
        const embeddings = await Promise.all(
          texts.map(text => getEmbedding(text))
        );
        return embeddings;
      },
      embedQuery: async (text: string) => {
        return getEmbedding(text);
      }
    }, {
      pineconeIndex: index,
      namespace,
    });

    return NextResponse.json({ message: 'Uploaded to Pinecone' });
  } catch (error: any) {
    console.error('🔥 Pinecone upload error:', error.message, error);
    return NextResponse.json({ error: 'Failed to upload to Pinecone' }, { status: 500 });
  }
}
