import { OpenAI } from 'openai';

export const runtime = 'nodejs';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  // 🔐 For production: use `process.env.OPENAI_API_KEY`
});

export async function POST(req: Request) {
  try {
    const { question } = await req.json();
    if (!question) {
      return new Response(JSON.stringify({ error: 'Missing question' }), { status: 400 });
    }

    const systemPrompt = `
You are a sustainability reporting expert helping companies align with SASB standards.

Given a SASB-style question (e.g. "Does the report disclose GHG emissions, including Scope 1 and 2?"), respond with a brief, actionable improvement tip.

Your advice should:
- Be 1–3 sentences long
- Be specific and easy to follow
- Focus on how to improve alignment with SASB expectations

Respond ONLY with the improvement advice in a JSON object:
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
      console.error("❌ Failed to parse GPT JSON:", raw);
      return new Response(JSON.stringify({ error: "OpenAI response format error" }), { status: 500 });
    }

  } catch (err) {
    console.error("🔥 SASB improvement error:", err);
    return new Response(JSON.stringify({ error: "Internal server error" }), { status: 500 });
  }
}
