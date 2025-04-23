import { OpenAI } from 'openai';

export const runtime = 'nodejs';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
,
});


export async function POST(req: Request) {
  try {
    const { question } = await req.json();
    if (!question) {
      return new Response(JSON.stringify({ error: 'Missing question' }), { status: 400 });
    }

    const systemPrompt = `
You are a climate disclosure and ESG expert helping companies improve CSRD alignment.

Given a CSRD-aligned question (e.g. "Does the report include a double materiality assessment?) respond with a brief, actionable improvement tip.

Your advice should:
- Be 1–3 sentences long
- Be specific and easy to follow
- Focus on how to improve alignment with CSRD expectations and requirements

Respond ONLY with the improvement advice as a JSON object like:
{ "advice": "..." }
    `.trim();

    const completion = await openai.chat.completions.create({
      model: 'gpt-4-turbo-preview',
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: question }
      ]
    });

    const raw = completion.choices[0]?.message?.content || '{}';

    try {
      const parsed = JSON.parse(raw);
      return new Response(JSON.stringify(parsed), { status: 200 });
    } catch (err) {
      console.error("Failed to parse GPT JSON:", raw);
      return new Response(JSON.stringify({ error: "OpenAI response format error" }), { status: 500 });
    }

  } catch (err) {
    console.error("CSRD improvement error:", err);
    return new Response(JSON.stringify({ error: "Internal server error" }), { status: 500 });
  }
}
