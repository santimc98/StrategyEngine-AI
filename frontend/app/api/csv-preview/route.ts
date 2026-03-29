import { NextResponse } from "next/server";
import fs from "fs";

function parseCsvLine(text: string, delimiter: string) {
  const result: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (char === '"' && text[i+1] === '"') {
          cur += '"';
          i++;
      } else if (char === '"') {
          inQuotes = !inQuotes;
      } else if (char === delimiter && !inQuotes) {
          result.push(cur.trim());
          cur = '';
      } else {
          cur += char;
      }
  }
  result.push(cur.trim());
  return result;
}

export async function POST(req: Request) {
  try {
    const { csvPath } = await req.json();
    if (!csvPath) return NextResponse.json({ error: "No path" }, { status: 400 });

    if (!fs.existsSync(csvPath)) {
      return NextResponse.json({ error: "File not found on disk" }, { status: 404 });
    }
    
    const stream = fs.createReadStream(csvPath, { start: 0, end: 12000, encoding: 'utf-8' });
    let content = '';
    for await (const chunk of stream) {
        content += chunk;
    }
    const lines = content.split('\n');
    if (lines.length < 2) return NextResponse.json({ error: "Empty or invalid CSV" }, { status: 400 });
    
    const headerLine = lines[0];
    const sep = headerLine.includes(';') && !headerLine.includes(',') ? ';' : ',';
    
    const columns = parseCsvLine(headerLine, sep);
    const rows = [];
    
    for (let i = 1; i < Math.min(lines.length, 10); i++) {
        if (!lines[i].trim()) continue;
        const vals = parseCsvLine(lines[i], sep);
        const obj: Record<string, string> = {};
        columns.forEach((col, idx) => { obj[col] = vals[idx] || ""; });
        rows.push(obj);
    }
    
    return NextResponse.json({ rows, columns });
  } catch(e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
