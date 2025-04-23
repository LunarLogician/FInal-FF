import { OpenAI } from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function getEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float"
  });
  return response.data[0].embedding;
}

export const splitTextIntoChunks = (text: string, maxChunkSize: number = 1000): string[] => {
  if (!text) return [];
  
  // Safeguard against extremely large texts
  const MAX_TOTAL_LENGTH = 100000; // 100K characters max
  const truncatedText = text.slice(0, MAX_TOTAL_LENGTH);
  
  const chunks: string[] = [];
  let currentChunk = '';
  
  // Split by sentences or paragraphs
  const sentences = truncatedText.split(/(?<=[.!?])\s+/);
  
  for (const sentence of sentences) {
    if ((currentChunk + sentence).length <= maxChunkSize) {
      currentChunk += (currentChunk ? ' ' : '') + sentence;
    } else {
      if (currentChunk) chunks.push(currentChunk);
      currentChunk = sentence;
    }
  }
  
  if (currentChunk) chunks.push(currentChunk);
  
  return chunks;
}; 