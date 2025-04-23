import { OpenAI } from 'openai';
import { searchDocument, storeDocument } from '@/utils/documentStore';
import { documentCache } from '@/utils/documentCache';

export const runtime = 'nodejs';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export async function POST(req: Request) {
  try {
    const { messages, docText, docName, docId } = await req.json();
    
    // Get the last user message
    const lastUserMessage = messages[messages.length - 1];
    console.log('Processing chat request with message:', lastUserMessage.content);

    let documentId = docId;
    let context = '';

    // If docText is provided but no docId, store the document
    if (docText && !docId) {
      try {
        const fileName = docName || `doc-${Date.now()}.txt`;
        documentId = fileName;
        // Store in both Pinecone and cache
        await storeDocument(docText, fileName);
        documentCache.store(documentId, docText, fileName);
        console.log('Stored new document with ID:', documentId);
      } catch (error) {
        console.error('Error storing document:', error);
      }
    }

    // If we have a document ID, try to get context
    if (documentId) {
      // First try to get from cache
      const cachedDoc = documentCache.get(documentId);
      if (cachedDoc) {
        // Use full document as context
        context = cachedDoc.text;
      } else {
        // Fall back to vector search if not in cache
        const relevantChunks = await searchDocument(lastUserMessage.content, documentId);
        context = relevantChunks
          .map(chunk => chunk.text)
          .join('\n\n');
      }
    }

    // If no context available, respond without it
    if (!context) {
      const response = await openai.chat.completions.create({
        model: 'gpt-4-turbo-preview',
        messages: [
          { role: 'system', content: 'You are a helpful assistant. Respond clearly and concisely.' },
          ...messages,
        ],
        stream: true,
      });

      return streamResponse(response);
    }

    // Create chat completion with document context
    const response = await openai.chat.completions.create({
      model: 'gpt-4-turbo-preview',
      messages: [
        {
          role: 'system',
          content: `You are a helpful assistant analyzing documents with expertise in ESG reporting standards (including GRI, SASB, TCFD, and others). 

When analyzing documents:
1. Pay special attention to document titles, headers, and metadata that indicate which reporting standards are being followed
2. Look for explicit mentions of reporting frameworks in both the document title and content
3. When asked about reporting standards, check both:
   - If the document itself is a specific standard's report (e.g. TCFD report)
   - If the document mentions following or aligning with any standards

Use the following document context to answer questions accurately and cite specific parts when possible:

Document Context:
${context}

If the context doesn't contain relevant information for a question, explicitly state that the document doesn't mention it.`
        },
        ...messages
      ],
      stream: true,
    });

    return streamResponse(response);

  } catch (error) {
    console.error('Error in chat endpoint:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to process request' }),
      { status: 500 }
    );
  }
}

// Helper function to stream the response
function streamResponse(response: any) {
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      for await (const chunk of response) {
        const content = chunk.choices[0]?.delta?.content;
        if (content) {
          controller.enqueue(encoder.encode(content));
        }
      }
      controller.close();
    },
  });

  return new Response(stream, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' }
  });
}
