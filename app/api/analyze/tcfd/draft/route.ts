import { OpenAI } from 'openai';

export const runtime = 'nodejs';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (magA * magB);
}

function chunkText(text: string, maxWords = 200): string[] {
  const clean = text
    .replace(/\n{2,}/g, '\n')
    .replace(/Page\s+\d+/gi, '')
    .replace(/ {2,}/g, ' ')
    .trim();

  const sentences = clean.split(/(?<=[.?!])\s+/);
  const chunks: string[] = [];
  let current: string[] = [];

  for (const sentence of sentences) {
    current.push(sentence);
    const wordCount = current.join(' ').split(/\s+/).length;
    if (wordCount > maxWords) {
      chunks.push(current.join(' '));
      current = [];
    }
  }
  if (current.length > 0) chunks.push(current.join(' '));
  return chunks;
}

export async function POST(req: Request) {
  try {
    const { question, docText } = await req.json();

    if (!question || !docText) {
      return new Response(JSON.stringify({ error: 'Missing question or document text' }), { status: 400 });
    }

    const docChunks = chunkText(docText, 200);
    const embeddings = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: [question, ...docChunks],
      encoding_format: 'float',
    });

    const questionEmbedding = embeddings.data[0].embedding;
    const chunkEmbeddings = embeddings.data.slice(1);

    const scored = chunkEmbeddings.map((e, i) => ({
      chunk: docChunks[i],
      score: cosineSimilarity(questionEmbedding, e.embedding),
    }));

    const topMatch = scored.sort((a, b) => b.score - a.score)[0];
    const isLowMatch = topMatch.score < 0.40;

    const topChunks = isLowMatch
      ? ''
      : scored
          .sort((a, b) => b.score - a.score)
          .slice(0, 2)
          .map((c) => c.chunk)
          .join('\n\n');

    const systemPrompt = `
You are an ESG and climate disclosure expert helping companies improve alignment with TCFD.

Given a specific TCFD-aligned requirement (e.g. "Describe the board’s oversight of climate-related risks and opportunities") and the content of a draft ESG report, generate a short paragraph that could be included in the company’s report to better meet that requirement.

Use a professional, objective, and report-style tone. The draft should sound like part of an actual disclosure, and should be based on the document provided.

Respond ONLY with a JSON object like:
{ "draft": "..." }
    `.trim();

    const completion = await openai.chat.completions.create({
      model: 'gpt-4-turbo-preview',
      response_format: { type: 'json_object' },
      max_tokens: 600,
      temperature: 0.5,
      messages: [
        { role: 'system', content: systemPrompt },
        {
          role: 'user',
          content: isLowMatch
            ? `Requirement: ${question}\n\nThe uploaded document does not contain relevant information. Please invent a plausible paragraph that would help meet this requirement.`
            : `Requirement: ${question}\n\nDocument:\n${topChunks}`,
        },
      ],
    });

    const raw = completion.choices[0]?.message?.content || '{}';
    const parsed = JSON.parse(raw);

    return new Response(JSON.stringify(parsed), { status: 200 });
  } catch (err) {
    console.error("TCFD draft generation error:", err);
    return new Response(JSON.stringify({ error: "Internal server error" }), { status: 500 });
  }
}
