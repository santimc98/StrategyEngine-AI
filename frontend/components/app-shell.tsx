"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { StatusPill } from "@/components/status-pill";
import { formatShortText } from "@/lib/format";
import type { RunListItem, RunListResponse } from "@/types/api";

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [historyRuns, setHistoryRuns] = useState<RunListItem[]>([]);

  useEffect(() => {
    let cancelled = false;

    async function loadRuns(): Promise<void> {
      try {
        const response = await fetch("/api/runs?limit=12", { cache: "no-store" });
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as RunListResponse;
        if (!cancelled) {
          setHistoryRuns(payload.items || []);
        }
      } catch {
        if (!cancelled) {
          setHistoryRuns([]);
        }
      }
    }

    void loadRuns();
    return () => {
      cancelled = true;
    };
  }, [pathname]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-top">
          <Link href="/" className="brand-block">
            <p className="brand-kicker">StrategyEngine AI</p>
            <h1>Asistente multiagente para equipos de datos</h1>
            <p className="brand-copy">
              Define el objetivo, adjunta el dataset y deja que el sistema ejecute la run con
              trazabilidad completa.
            </p>
          </Link>

          <Link
            href="/settings"
            className={`settings-link${pathname.startsWith("/settings") ? " active" : ""}`}
          >
            Settings
          </Link>
        </div>

        <section className="history-section">
          <div className="history-section-head">
            <p className="sidebar-title">Historial de runs</p>
            <Link href="/runs" className="history-link">
              Ver todo
            </Link>
          </div>

          <div className="history-list">
            {historyRuns.length ? (
              historyRuns.map((run) => {
                const active = pathname === `/runs/${run.run_id}`;
                
                const getRunTitle = () => {
                  if (run.strategy && run.strategy.trim().length > 0) return run.strategy;
                  if (run.business_objective) {
                     const lines = run.business_objective.split("\n").filter(l => l.trim().length > 0 && !l.toLowerCase().includes("dataset") && !l.toLowerCase().includes("file:") && !l.includes("KEY COLUMNS"));
                     if (lines.length > 0) {
                         return lines[0].slice(0, 45) + (lines[0].length > 45 ? "..." : "");
                     }
                  }
                  return `Análisis de Datos (${run.run_id.slice(0, 6)})`;
                };

                return (
                  <Link
                    key={run.run_id}
                    href={`/runs/${run.run_id}`}
                    className={`history-item${active ? " active" : ""}`}
                    title={`Run ID: ${run.run_id}\nObjetivo: ${run.business_objective}`}
                  >
                    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                       <strong style={{ 
                          fontSize: "0.92rem", 
                          lineHeight: 1.3, 
                          color: active ? "#ffffff" : "#c7d0ea", 
                          display: "-webkit-box", 
                          WebkitLineClamp: 2, 
                          WebkitBoxOrient: "vertical", 
                          overflow: "hidden",
                          wordBreak: "break-word"
                       }}>
                          {getRunTitle()}
                       </strong>
                       <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                          <div style={{ transform: "scale(0.85)", transformOrigin: "bottom left" }}>
                             <StatusPill value={run.verdict || run.status} />
                          </div>
                          <span style={{ fontSize: "0.75rem", color: "#88a7ff", opacity: 0.7 }}>
                             {run.started_str?.split(" ")[0] || ""}
                          </span>
                       </div>
                    </div>
                  </Link>
                );
              })
            ) : (
              <p className="history-empty">Aún no hay runs indexadas en la API.</p>
            )}
          </div>
        </section>
      </aside>

      <main className="main-content">
        <div className="content-inner">{children}</div>
      </main>
    </div>
  );
}
