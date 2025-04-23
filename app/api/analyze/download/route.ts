import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export const runtime = 'nodejs';

export async function GET(req: NextRequest) {
  const filePath = req.nextUrl.searchParams.get('file');
  if (!filePath) {
    return NextResponse.json({ error: 'Missing file param' }, { status: 400 });
  }

  // E.g. your code wrote the file to “/Users/…”
  // For safety, you might keep your excel files in “/tmp” or project folder.
  try {
    const fileContents = fs.readFileSync(filePath);
    // Set headers for a file download
    return new NextResponse(fileContents, {
      headers: {
        'Content-Type':
          'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'Content-Disposition': `attachment; filename="${path.basename(filePath)}"`,
      },
    });
  } catch (err: any) {
    console.error('Download error:', err);
    return NextResponse.json({ error: 'File not found or read error' }, { status: 404 });
  }
}
