import { NextResponse } from 'next/server';
import { embedAndUpload } from '@/utils/embedUpload';

export async function POST(req: Request) {
  try {
    const { text, namespace } = await req.json();

    if (!text || !namespace) {
      return NextResponse.json(
        { error: 'Missing required fields: text and namespace' },
        { status: 400 }
      );
    }

    // Embed and upload to Pinecone
    const numChunks = await embedAndUpload(text, namespace);

    return NextResponse.json({
      message: 'Successfully embedded and uploaded to Pinecone',
      chunks: numChunks
    });
  } catch (error) {
    console.error('Error in Pinecone upload:', error);
    return NextResponse.json(
      { error: 'Failed to upload to Pinecone' },
      { status: 500 }
    );
  }
} 