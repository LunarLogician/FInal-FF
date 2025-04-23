import { OpenAI } from 'openai';
import fs from 'fs';
import path from 'path';

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

function chunkTextBySentence(text: string, maxWords = 200): string[] {
  const sentences = text.replace(/\n{2,}/g, '\n').split(/(?<=[.?!])\s+/);
  const chunks: string[] = [];
  let current: string[] = [];
  for (const sentence of sentences) {
    current.push(sentence);
    if (current.join(' ').split(/\s+/).length > maxWords) {
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
    const checklistPath = path.join(process.cwd(), 'public', 'data', 'tcfd_checklist.json');
    const checklistJson = fs.readFileSync(checklistPath, 'utf8');
    const checklist = JSON.parse(checklistJson);

    const flatChecklist = Object.entries(checklist).flatMap(([section, items]) =>
      (items as any[]).flatMap((item) => {
        const variants = item.alternatives || [item.question];
        return variants.map((variant: string) => ({
          id: item.id,
          section,
          question: item.question,
          variant,
          threshold: item.threshold ?? 0.60, // 👈 pulled directly from the JSON
        }));
      })
    );

    const docChunks = chunkTextBySentence(docText);
    const chunkEmbeddings = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: docChunks,
      encoding_format: 'float',
    });

    const variantEmbeddings = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: flatChecklist.map((q) => q.variant),
      encoding_format: 'float',
    });

    const scores: Record<string, { section: string; question: string; similarity: number; topChunk: string; threshold: number }> = {};

    for (let i = 0; i < flatChecklist.length; i++) {
      const item = flatChecklist[i];
      const variantEmbedding = variantEmbeddings.data[i].embedding;

      let bestScore = 0;
      let topChunk = '';

      for (let j = 0; j < chunkEmbeddings.data.length; j++) {
        const score = cosineSimilarity(variantEmbedding, chunkEmbeddings.data[j].embedding);
        if (score > bestScore) {
          bestScore = score;
          topChunk = docChunks[j];
        }
      }

      const current = scores[item.id];
      if (!current || bestScore > current.similarity) {
        scores[item.id] = {
          section: item.section,
          question: item.question,
          similarity: bestScore,
          topChunk,
          threshold: item.threshold ?? 0.60,
        };
      }
    }

    const results = Object.entries(scores).map(([id, entry]) => {
      const threshold = flatChecklist.find(item => item.id === id)?.threshold ?? 0.60;
      const similarity = entry.similarity;
    
      // ✅ Normalize similarity score to create match score percentage
      const matchScore = Math.min(1, similarity / threshold);
      const matchScorePercent = Math.round(matchScore * 100);
    
      let status = '❌ Not Met';
      if (similarity >= threshold) status = '✅ Likely Met';
      else if (similarity >= threshold - 0.10) status = '⚠️ Unclear';
    
      return {
        id,
        section: entry.section,
        question: entry.question,
        similarity,
        threshold,
        matchScore: matchScorePercent,
        status,
        topChunk: entry.topChunk,
      };
    });
    

    return new Response(JSON.stringify({ results }), { status: 200 });
  } catch (err) {
    console.error('TCFD checker error:', err);
    return new Response(JSON.stringify({ error: 'Internal server error' }), { status: 500 });
  }
}
