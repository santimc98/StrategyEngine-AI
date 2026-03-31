"use client";

import React from "react";
import { CopyButton } from "@/components/copy-button";
import type { RunReportResponse } from "@/types/api";

type RunReportProps = {
  runId: string;
  report: RunReportResponse;
};

export function RunReport({ runId, report }: RunReportProps) {
  const renderBlock = (block: any, idx: number) => {
    switch (block.type) {
      case "heading":
        const HTag = `h${block.level || 2}` as keyof React.JSX.IntrinsicElements;
        const fontSizes: Record<number, string> = { 1: "2.5rem", 2: "1.8rem", 3: "1.4rem" };
        const fs = fontSizes[block.level] || "1.2rem";
        return React.createElement(HTag, { 
          key: idx, 
          style: { marginTop: "32px", marginBottom: "16px", fontSize: fs, fontWeight: 700, color: "var(--text)" } 
        }, block.text);
      
      case "paragraph":
        return <p key={idx} style={{ lineHeight: 1.7, marginBottom: "20px", fontSize: "1.05rem", color: "var(--text)" }}>{block.text}</p>;
      
      case "bullet_list":
        return (
          <ul key={idx} style={{ marginBottom: "24px", paddingLeft: "24px", color: "var(--text)", fontSize: "1.05rem" }}>
            {block.items.map((item: string, i: number) => (
              <li key={i} style={{ marginBottom: "8px", lineHeight: 1.6 }}>{item}</li>
            ))}
          </ul>
        );
      
      case "numbered_list":
        return (
          <ol key={idx} style={{ marginBottom: "24px", paddingLeft: "24px", color: "var(--text)", fontSize: "1.05rem" }}>
            {block.items.map((item: string, i: number) => (
              <li key={i} style={{ marginBottom: "8px", lineHeight: 1.6 }}>{item}</li>
            ))}
          </ol>
        );

      case "artifact":
        if (block.artifact_type === "html_table") {
          return (
            <div key={idx} style={{ margin: "32px 0 40px 0" }}>
              <h4 style={{ fontSize: "1.2rem", marginBottom: "8px", fontWeight: 700 }}>{block.title}</h4>
              <p className="muted-copy" style={{ marginBottom: "16px", fontStyle: "italic" }}>{block.lead_in}</p>
              
              <div 
                className="table-wrap report-table" 
                style={{ overflow: "hidden", marginBottom: "16px" }}
                dangerouslySetInnerHTML={{ __html: block.content_html || "" }} 
              />
              
              {block.analysis && block.analysis.length > 0 && (
                <div style={{ paddingLeft: "16px", borderLeft: "4px solid var(--accent)", color: "var(--muted)", fontStyle: "italic", fontSize: "0.95rem" }}>
                   <ul style={{ margin: 0, paddingLeft: "20px" }}>
                     {block.analysis.map((a: string, i: number) => <li key={i} style={{ marginBottom: "4px" }}>{a}</li>)}
                   </ul>
                </div>
              )}
            </div>
          );
        }
        
        if (block.artifact_type === "chart" || block.artifact_type === "image") {
          const filename = block.path ? block.path.split('/').pop() : "";
          return (
            <div key={idx} style={{ margin: "40px 0 48px 0" }}>
              <h4 style={{ fontSize: "1.2rem", marginBottom: "8px", fontWeight: 700 }}>{block.title}</h4>
              <p className="muted-copy" style={{ marginBottom: "24px", fontStyle: "italic" }}>{block.lead_in}</p>
              
              <figure style={{ margin: "0 0 16px 0" }}>
                 <img 
                   src={`/api/runs/${runId}/report/plots/${filename}`} 
                   alt={block.title} 
                   style={{ maxWidth: "100%", maxHeight: "500px", objectFit: "contain", borderRadius: "8px", border: "1px solid var(--border)" }}
                 />
              </figure>

              {block.analysis && block.analysis.length > 0 && (
                <div style={{ paddingLeft: "16px", borderLeft: "4px solid var(--accent-2)", color: "var(--muted)", fontStyle: "italic", fontSize: "0.95rem" }}>
                   <ul style={{ margin: 0, paddingLeft: "20px" }}>
                     {block.analysis.map((a: string, i: number) => <li key={i} style={{ marginBottom: "4px" }}>{a}</li>)}
                   </ul>
                </div>
              )}
            </div>
          );
        }
        return null;

      case "markdown":
        // For 'markdown' block we shouldn't necessarily put it in a huge code frame, just a pre block that blends.
        return (
          <div key={idx} style={{ position: "relative", marginTop: "40px", padding: "24px", paddingTop: "36px", background: "var(--bg-soft)", borderRadius: "12px", border: "1px solid var(--border)" }}>
            <div style={{ position: "absolute", top: "8px", right: "8px", zIndex: 2 }}>
              <CopyButton text={block.content || ""} variant="light" />
            </div>
            <pre style={{ whiteSpace: "pre-wrap", color: "var(--text)", fontSize: "0.9rem", fontFamily: "var(--font-body)", lineHeight: 1.6, margin: 0 }}>
              {block.content}
            </pre>
          </div>
        );

      default:
        return null;
    }
  };

  // Build full report text for the "copy all" button
  const fullReportText = React.useMemo(() => {
    if (!report.blocks || report.blocks.length === 0) {
      return report.markdown || "";
    }
    return report.blocks
      .map((block: any) => {
        switch (block.type) {
          case "heading":
            return `${"#".repeat(block.level || 2)} ${block.text || ""}`;
          case "paragraph":
            return block.text || "";
          case "bullet_list":
            return (block.items || []).map((item: string) => `- ${item}`).join("\n");
          case "numbered_list":
            return (block.items || []).map((item: string, i: number) => `${i + 1}. ${item}`).join("\n");
          case "markdown":
            return block.content || "";
          case "artifact":
            if (block.artifact_type === "html_table") {
              return `[Tabla: ${block.title || ""}]\n${block.lead_in || ""}`;
            }
            if (block.artifact_type === "chart" || block.artifact_type === "image") {
              return `[Figura: ${block.title || ""}]\n${block.lead_in || ""}`;
            }
            return "";
          default:
            return "";
        }
      })
      .filter(Boolean)
      .join("\n\n");
  }, [report.blocks, report.markdown]);

  return (
    <div className="animate-fade-in" style={{ maxWidth: "860px", color: "var(--text)" }}>
      {/* Action bar: PDF + Copy report */}
      <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px", marginBottom: "24px", flexWrap: "wrap" }}>
        {report.pdf_available && report.pdf_url && (
          <a className="history-link" href={`/api${report.pdf_url}`} target="_blank" rel="noreferrer" style={{ background: "var(--bg-soft)", color: "var(--accent)", border: "1px solid var(--border-strong)" }}>
            + Descargar PDF Original
          </a>
        )}
        {fullReportText && (
          <CopyButton text={fullReportText} variant="light" label="Copiar Reporte" />
        )}
      </div>

      <div style={{ background: "#ffffff", padding: "64px 80px", borderRadius: "8px", border: "1px solid var(--border)", boxShadow: "0 4px 20px rgba(0,0,0,0.03)" }}>
        {report.blocks && report.blocks.length > 0 ? (
          report.blocks.map((block, idx) => renderBlock(block, idx))
        ) : (
          <div className="markdown-frame">
            <pre>{report.markdown || "No hay reporte estructurado disponible."}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
