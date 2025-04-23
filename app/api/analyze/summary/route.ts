// File: app/api/analyze/summary/route.ts
import { OpenAI } from 'openai';

export const runtime = 'nodejs';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: Request) {
  try {
    const { docText } = await req.json();

    if (!docText) {
      return new Response(JSON.stringify({ error: 'Missing document text' }), { status: 400 });
    }

    const systemPrompt = `
You are a corporate analyst summarizing companies sustainability, ESG, and annual reports.

Summarize the uploaded report in 3–5 bullet points. Focus on tone, key themes, and sustainability, ESG, climate relevance.
Respond ONLY with a JSON object like:
{ "summary": ["point 1", "point 2", "point 3"] }
    `.trim();

    const completion = await openai.chat.completions.create({
      model: 'gpt-4-turbo-preview',
      temperature: 0.4,
      max_tokens: 400,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Document:
${docText}` }
      ]
    });

    const raw = completion.choices[0]?.message?.content || '{}';
    const parsed = JSON.parse(raw);
    return new Response(JSON.stringify(parsed), { status: 200 });

  } catch (err) {
    console.error("Summary generation error:", err);
    return new Response(JSON.stringify({ error: 'Internal server error' }), { status: 500 });
  }
}
