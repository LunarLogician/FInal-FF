export const runtime = 'nodejs';

import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';
import { NextResponse } from 'next/server';

interface Row {
  pdf_name: string;
  company_name: string;
  report_name: string;
}

function groupByCompany(rows: Row[]) {
  const grouped: Record<string, Row[]> = {};

  for (const row of rows) {
    const { company_name } = row;
    if (!grouped[company_name]) {
      grouped[company_name] = [];
    }
    grouped[company_name].push(row);
  }

  return grouped;
}

export async function GET() {
  const filePath = path.join(process.cwd(), 'data', 'mapping file.csv');
  const file = fs.readFileSync(filePath, 'utf8');
  const parsed = Papa.parse<Row>(file, {
    header: true,
    skipEmptyLines: true,
  });

  const grouped = groupByCompany(parsed.data);

  return NextResponse.json(grouped);
}
