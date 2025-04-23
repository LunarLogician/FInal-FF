import { OpenAI } from 'openai';
import gri_checklist from '@/public/data/gri_checklist.json';

export const runtime = 'nodejs';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

function cosineSimilarity(a: number[], b: number[]) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (magA * magB);
}

function chunkTextBySentence(text: string, maxWords = 200): string[] {
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
  const { docText } = await req.json();
  if (!docText) return new Response(JSON.stringify({ error: 'Missing document text' }), { status: 400 });

  try {
    const checklist = gri_checklist;

    const flatChecklist = Object.entries(checklist).flatMap(([section, items]) =>
      items.flatMap((item: any) => {
        const variants = item.alternatives || [item.question];
        return variants.map((variant: string) => ({
          id: item.id,
          section,
          question: item.question,
          variant,
          threshold: item.threshold ?? 0.60  // 👈 fallback to default
        }));
      })
    );
    

    const docChunks = chunkTextBySentence(docText, 200);
    const chunkEmbeddings = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: docChunks,
      encoding_format: 'float',
    });

    const variantTexts = flatChecklist.map((item) => item.variant);
    const variantEmbeddings = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: variantTexts,
      encoding_format: 'float',
    });

    const results: any[] = [];

    for (let i = 0; i < flatChecklist.length; i++) {
      const variant = flatChecklist[i];
      const variantEmbedding = variantEmbeddings.data[i].embedding;

      let maxSim = 0;
      let topChunk = '';

      for (let j = 0; j < chunkEmbeddings.data.length; j++) {
        const chunk = chunkEmbeddings.data[j];
        const score = cosineSimilarity(chunk.embedding, variantEmbedding);
        if (score > maxSim) {
          maxSim = score;
          topChunk = docChunks[j];
        }
      }

      results.push({
        id: variant.id,
        section: variant.section,
        question: variant.question,
        similarity: maxSim,
        topChunk,
      });
    }

    const grouped: Record<string, { section: string; question: string; best: number; topChunk: string }> = {};
    for (const r of results) {
      if (!grouped[r.id] || r.similarity > grouped[r.id].best) {
        grouped[r.id] = { section: r.section, question: r.question, best: r.similarity, topChunk: r.topChunk };
      }
    }

    const final = await Promise.all(
      Object.entries(grouped).map(async ([id, entry]) => {
        const threshold = flatChecklist.find((item) => item.id === id)?.threshold ?? 0.60;
        const similarity = entry.best;
    
        // Normalize the score
        const matchScore = Math.min(1, similarity / threshold);
        const matchScorePercent = Math.round(matchScore * 100);
    
        let status = '❌ Not Met';
        if (matchScorePercent >= 100) status = '✅ Likely Met';
        else if (matchScorePercent >= 80) status = '⚠️ Unclear';
    
        let advice = '';
        if (status !== '✅ Likely Met') {
          const chat = await openai.chat.completions.create({
            model: 'gpt-4-turbo-preview',
            messages: [
              {
                role: 'system',
                content: `You are a sustainability reporting advisor. Provide clear, actionable advice (2–3 sentences) to help a company improve its GRI disclosure to meet the following guideline.`,
              },
              {
                role: 'user',
                content: `GRI Guideline: "${entry.question}"\n\nDocument Excerpt:\n"${entry.topChunk.slice(0, 1000)}"\n\nHow can the company improve its disclosure to meet the requirement?`,
              },
            ],
            response_format: { type: 'text' },
          });
    
          advice = chat.choices[0]?.message?.content?.trim() || '';
        }
    
        return {
          id,
          section: entry.section,
          question: entry.question,
          similarity,
          threshold,
          matchScore: matchScorePercent,
          status,
          advice,
        };
      })
    );
    

    return new Response(JSON.stringify({ results: final }), { status: 200 });
  } catch (err) {
    console.error('GRI error:', err);
    return new Response(JSON.stringify({ error: 'Internal server error' }), { status: 500 });
  }
}
