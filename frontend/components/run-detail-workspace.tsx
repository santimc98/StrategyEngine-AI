"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { CopyableFrame } from "@/components/copy-button";
import { RunLiveConsole } from "@/components/run-live-console";
import { RunReport } from "@/components/run-report";
import { StatusPill } from "@/components/status-pill";
import { formatDate, formatNumber, formatShortText } from "@/lib/format";
import type {
  ArtifactManifestResponse,
  JsonRecord,
  RunDetailResponse,
  RunReportResponse,
} from "@/types/api";

type RunDetailWorkspaceProps = {
  runId: string;
  detail: RunDetailResponse;
  report: RunReportResponse | null;
  manifest: ArtifactManifestResponse | null;
};

type AgentStepKey =
  | "initial"
  | "steward"
  | "strategist"
  | "planner"
  | "data_engineer"
  | "ml_engineer"
  | "translator";

const agentSteps: Array<{ key: AgentStepKey; label: string }> = [
  { key: "initial", label: "01. Estado Inicial" },
  { key: "steward", label: "02. Data Steward" },
  { key: "strategist", label: "03. Strategist" },
  { key: "planner", label: "04. Planner" },
  { key: "data_engineer", label: "05. Data Engineer" },
  { key: "ml_engineer", label: "06. ML Engineer" },
  { key: "translator", label: "07. Translator" },
];

export function RunDetailWorkspace({
  runId,
  detail,
  report,
  manifest,
}: RunDetailWorkspaceProps) {
  const router = useRouter();
  const [activeStep, setActiveStep] = useState<AgentStepKey>("initial");

  const result = detail.result || {};
  const input = detail.input || {};
  const serverStatus = detail.status || {};
  const runSummary = (report?.run_summary || {}) as Record<string, unknown>;

  // --------------- Live status polling ---------------
  // The server component loads detail.status once. To track a running run in
  // real-time we poll /status independently so isRunning stays accurate even
  // if the page was rendered before the worker wrote its first status file.
  const initialIsRunning = String(serverStatus.status || "").toLowerCase() === "running";
  const [liveStatus, setLiveStatus] = useState<JsonRecord | null>(
    initialIsRunning ? serverStatus : null,
  );
  const [isRunning, setIsRunning] = useState(initialIsRunning);
  const wasRunningRef = useRef(initialIsRunning);

  const pollStatus = useCallback(async () => {
    try {
      const res = await fetch(`/api/runs/${runId}/status`, { cache: "no-store" });
      if (!res.ok) return;
      const payload = (await res.json()) as JsonRecord;
      const running = String(payload.status || "").toLowerCase() === "running";
      setLiveStatus(payload);
      setIsRunning(running);

      if (wasRunningRef.current && !running) {
        // Run just finished — refresh server data so the page shows final results
        wasRunningRef.current = false;
        router.refresh();
      }
      if (running) {
        wasRunningRef.current = true;
      }
    } catch {
      // Silently ignore — next poll will retry
    }
  }, [runId, router]);

  useEffect(() => {
    // Always do an initial status check to catch race conditions where the
    // server-rendered page loaded before worker_status.json existed.
    void pollStatus();

    // Keep polling while running (or until we confirm it's not running).
    // We poll at a slower cadence here (5s) because RunLiveConsole does its
    // own fast polling (3s) for logs + status when visible.
    const id = window.setInterval(() => { void pollStatus(); }, 5000);
    return () => window.clearInterval(id);
  }, [pollStatus]);

  // Derive the display status: prefer live-polled status, fall back to server
  const displayStatus = liveStatus || serverStatus;
  const runStatus = String(displayStatus.status || "N/A");

  const businessObjective = String(result.business_objective || input.business_objective || "");
  const dataSummary = String(result.data_summary || "No hay summary generado.");
  const selectedStrategy = (result.selected_strategy || {}) as Record<string, unknown>;
  const executionContract = (result.execution_contract || {}) as Record<string, unknown>;
  const cleaningCode = String(result.cleaning_code || "").trim();
  const generatedCode = String(result.generated_code || result.last_generated_code || "").trim();
  const executionOutput = String(result.execution_output || "").trim();
  const reviewBoardVerdict = (result.review_board_verdict || {}) as Record<string, unknown>;

  const requiredActions = Array.isArray(reviewBoardVerdict.required_actions)
    ? reviewBoardVerdict.required_actions.map(String)
    : [];

  const transformPandasSplit = (inputStr: unknown) => {
    try {
      if (typeof inputStr === "string") {
        const parsed = JSON.parse(inputStr);
        if (parsed && Array.isArray(parsed.data) && Array.isArray(parsed.columns)) {
           return parsed.data.map((row: any[]) => {
              const obj: Record<string, unknown> = {};
              parsed.columns.forEach((col: string, idx: number) => { obj[col] = row[idx]; });
              return obj;
           });
        }
      } else if (Array.isArray(inputStr)) {
        return inputStr as Array<Record<string, unknown>>;
      }
    } catch(e) {}
    return [];
  };

  const rawPreview = useMemo(() => transformPandasSplit(result.raw_data_preview), [result.raw_data_preview]);
  const cleanedPreview = useMemo(() => transformPandasSplit(result.cleaned_data_preview), [result.cleaned_data_preview]);

  const finalVerdict = String(
    reviewBoardVerdict.final_review_verdict ||
      reviewBoardVerdict.status ||
      result.review_verdict ||
      detail.status?.status ||
      "N/A",
  );

  const [dynamicRawPreview, setDynamicRawPreview] = useState<Array<Record<string, unknown>>>([]);
  const [loadingPreview, setLoadingPreview] = useState(false);

  // Dynamic fallback for missing raw preview!
  useEffect(() => {
    if (activeStep === "initial") {
      const p = String(input.csv_path || "");
      if (p && dynamicRawPreview.length === 0) {
        setLoadingPreview(true);
        fetch("/api/csv-preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ csvPath: p })
        }).then(r => r.json()).then(data => {
           if (data.rows && data.rows.length) {
              setDynamicRawPreview(data.rows);
           }
        }).catch(e => console.error("Preview fetch failed", e)).finally(() => setLoadingPreview(false));
      }
    }
  }, [activeStep, input.csv_path, dynamicRawPreview.length]);

  // runStatus and isRunning are now derived from live polling above

  const renderTable = (rows: Array<Record<string, unknown>>) => {
    if (!rows || rows.length === 0) return <p className="muted-copy">Vista previa no disponible en este momento.</p>;
    const columns = Object.keys(rows[0]);
    return (
      <div className="table-wrap" style={{ background: "var(--panel)", borderRadius: "12px", border: "1px solid var(--border)", overflow: "auto", boxShadow: "0 8px 32px rgba(48, 69, 120, 0.04)" }}>
        <table className="data-table" style={{ margin: 0, width: "100%", whiteSpace: "nowrap" }}>
          <thead style={{ background: "rgba(0, 0, 0, 0.02)" }}>
            <tr>
              {columns.map((col) => (
                <th key={col} style={{ padding: "10px 14px", textAlign: "left", fontSize: "0.8rem", color: "var(--text)" }}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 6).map((row, idx) => (
              <tr key={idx} style={{ borderTop: "1px solid var(--border)" }}>
                {columns.map((col) => (
                  <td key={col} style={{ padding: "8px 14px", fontSize: "0.85rem", color: "var(--muted)" }}>{String(row[col] ?? "")}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };
  
  const displayPreview = dynamicRawPreview;

  const [dynamicCleanedPreview, setDynamicCleanedPreview] = useState<Array<Record<string, unknown>>>([]);
  const [loadingCleanedPreview, setLoadingCleanedPreview] = useState(false);

  useEffect(() => {
    if (activeStep === "data_engineer") {
      // Find the cleaned dataset path from manifest
      const cleanedItem = manifest?.items?.find(item => item.path.endsWith("dataset_cleaned.csv") || item.path.endsWith("cleaned_full.csv"));
      let p = cleanedItem?.path;
      
      let absolutePath = "";
      if (p && input.csv_path) {
         const basePath = String(input.csv_path).replace(/[/\\]?data[\\/].*?$/, "");
         const sep = String(input.csv_path).includes("\\") ? "\\" : "/";
         absolutePath = `${basePath}${sep}runs${sep}${runId}${sep}${p.replace(/\//g, sep)}`;
      }

      if (absolutePath && dynamicCleanedPreview.length === 0) {
        setLoadingCleanedPreview(true);
        fetch("/api/csv-preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ csvPath: absolutePath })
        }).then(r => r.json()).then(data => {
           if (data.rows && data.rows.length) {
              setDynamicCleanedPreview(data.rows);
           } else if (cleanedPreview.length > 0) {
              // fallback to static JSON if file missing or un-parsable
              setDynamicCleanedPreview(cleanedPreview);
           }
        }).catch(e => {
           console.error("Cleaned preview fetch failed", e);
           setDynamicCleanedPreview(cleanedPreview); // fallback
        }).finally(() => setLoadingCleanedPreview(false));
      } else if (!absolutePath && cleanedPreview.length > 0 && dynamicCleanedPreview.length === 0) {
        setDynamicCleanedPreview(cleanedPreview); // fallback immediately
      }
    }
  }, [activeStep, manifest, input.csv_path, dynamicCleanedPreview.length, cleanedPreview]);

  const finalCleanedPreview = dynamicCleanedPreview.length > 0 ? dynamicCleanedPreview : cleanedPreview;

  return (
    <div className="stack-xl run-document-flow" style={{ width: "100%", maxWidth: "1400px", margin: "0 auto" }}>
      {/* HEADER: META RUN INFO */}
      <header style={{ borderBottom: "1px solid var(--border)", paddingBottom: "24px", display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
        <div>
          <p className="eyebrow">Run ID</p>
          <h1 style={{ fontSize: "2.5rem", margin: "4px 0 12px 0", lineHeight: 1 }}>{runId}</h1>
          <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
            <StatusPill value={runStatus} />
            <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
               Veredicto Final: <strong style={{ color: "var(--text)" }}>{finalVerdict.replace("_", " ")}</strong>
            </span>
          </div>
        </div>
        {!isRunning && (
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            <a
              className="secondary-button"
              href={`/api/runs/${runId}/artifacts/zip`}
              download
              style={{ fontSize: "0.85rem", whiteSpace: "nowrap" }}
            >
              Descargar Artefactos (.zip)
            </a>
          </div>
        )}
      </header>

      {/* AGENT TIMELINE / WORKFLOW REVIEW UI */}
      <div style={{ display: "grid", gridTemplateColumns: "260px minmax(0, 1fr)", gap: "48px", alignItems: "start" }}>
        
        {/* LEFT NAV: PIPELINE STEPPER */}
        <aside style={{ position: "sticky", top: "24px", display: "grid", gap: "8px" }}>
          <div>
            <p className="eyebrow" style={{ marginBottom: "16px" }}>Pipeline de Agentes</p>
          </div>
          {agentSteps.map((step) => {
            if (isRunning && step.key !== "initial") return null;
            const isActive = activeStep === step.key;
            return (
              <button
                key={step.key}
                onClick={() => setActiveStep(step.key)}
                style={{
                  display: "block",
                  width: "100%",
                  textAlign: "left",
                  background: isActive ? "var(--panel)" : "transparent",
                  boxShadow: isActive ? "0 4px 12px rgba(48, 69, 120, 0.05)" : "none",
                  color: isActive ? "var(--accent)" : "var(--text)",
                  border: "1px solid",
                  borderColor: isActive ? "var(--border-strong)" : "transparent",
                  padding: "12px 16px",
                  borderRadius: "12px",
                  fontWeight: isActive ? 700 : 500,
                  fontSize: "1rem",
                  cursor: "pointer",
                  transition: "all 0.2s ease"
                }}
              >
                {step.label}
              </button>
            );
          })}
        </aside>

        {/* RIGHT CONTENT: AGENT WORK VIEWER */}
        <main style={{ minHeight: "60vh", paddingBottom: "100px" }}>
          
          {isRunning && (
            <div style={{ marginBottom: "32px" }}>
              <RunLiveConsole runId={runId} initialStatus={displayStatus} />
            </div>
          )}

          <div style={{ background: "var(--panel)", borderRadius: "24px", padding: "40px", border: "1px solid var(--border)", boxShadow: "0 24px 70px rgba(48, 69, 120, 0.08)", minHeight: "100%" }}>

          {/* STEP 1: INITIAL STATE */}
          {activeStep === "initial" && (
            <div className="stack-lg animate-fade-in">
              <h2 style={{ fontSize: "2rem" }}>Estado Inicial (Input)</h2>
              <article className="stack-sm">
                <p className="eyebrow">Vista Previa del Dataset Original ({String(input.csv_path || result.csv_path || "N/A").split('\\').pop()})</p>
                {loadingPreview ? (
                   <p className="muted-copy">Cargando vista previa desde disco...</p>
                ) : displayPreview.length > 0 ? (
                   renderTable(displayPreview)
                ) : (
                  <div style={{ padding: "16px", background: "var(--panel)", borderRadius: "12px", border: "1px solid var(--border)", fontFamily: "monospace", fontSize: "0.95rem" }}>
                    Ruta: {String(input.csv_path || result.csv_path || "N/A")}
                    <br/><br/>
                    <span className="muted-copy" style={{ fontFamily: "var(--font-body)" }}>Nota: Error al intentar cargar la vista previa directa vía disco, o el archivo no es un CSV válido.</span>
                  </div>
                )}
              </article>
              <article className="stack-sm">
                <p className="eyebrow">Objetivo de Negocio</p>
                {businessObjective ? (
                  <CopyableFrame copyText={businessObjective} variant="light" copyLabel="Copiar">
                    <div style={{ padding: "24px", paddingTop: "40px", background: "var(--panel)", borderRadius: "16px", border: "1px solid var(--border)", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
                      {businessObjective}
                    </div>
                  </CopyableFrame>
                ) : (
                  <div style={{ padding: "24px", background: "var(--panel)", borderRadius: "16px", border: "1px solid var(--border)", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
                    No se especificó un objetivo de negocio para esta ejecución.
                  </div>
                )}
              </article>
            </div>
          )}

          {/* STEP 2: STEWARD */}
          {activeStep === "steward" && (
            <div className="stack-lg animate-fade-in">
              <h2 style={{ fontSize: "2rem" }}>Auditoría del Data Steward</h2>
              <article className="stack-sm">
                <p className="eyebrow">Data Summary & Profile</p>
                <CopyableFrame copyText={dataSummary} variant="light" copyLabel="Copiar">
                  <div style={{ padding: "32px", paddingTop: "40px", background: "var(--panel)", borderRadius: "16px", border: "1px solid var(--border)", whiteSpace: "pre-wrap", lineHeight: 1.6, fontSize: "0.95rem" }}>
                    {dataSummary}
                  </div>
                </CopyableFrame>
              </article>
            </div>
          )}

          {/* STEP 3: STRATEGIST */}
          {activeStep === "strategist" && (
            <div className="stack-lg animate-fade-in">
              <h2 style={{ fontSize: "2rem" }}>Diseño del AI Strategist</h2>
              <div className="overview-grid">
                <article className="stack-sm" style={{ padding: "24px", background: "var(--panel)", borderRadius: "16px", border: "1px solid var(--border)", gridColumn: "span 2" }}>
                  <p className="eyebrow">Estrategia Seleccionada</p>
                  <strong style={{ fontSize: "1.25rem", display: "block", marginBottom: "8px" }}>
                    {String(selectedStrategy.title || "Sin título")}
                  </strong>
                  <p className="muted-copy">{String(selectedStrategy.objective_reasoning || "")}</p>
                </article>
                <article className="stack-sm" style={{ padding: "24px", background: "var(--panel)", borderRadius: "16px", border: "1px solid var(--border)", gridColumn: "span 2" }}>
                  <p className="eyebrow">Hipótesis Principal</p>
                  <p style={{ lineHeight: 1.6, margin: 0 }}>
                    {String(selectedStrategy.hypothesis || "No data")}
                  </p>
                </article>
                <article className="stack-sm" style={{ padding: "24px", background: "var(--panel)", borderRadius: "16px", border: "1px solid var(--border)", gridColumn: "span 2" }}>
                  <p className="eyebrow">Razón de Validación</p>
                  <p style={{ lineHeight: 1.6, margin: 0 }}>
                    {String(selectedStrategy.validation_rationale || "No data")}
                  </p>
                </article>
              </div>
            </div>
          )}

          {/* STEP 4: PLANNER */}
          {activeStep === "planner" && (
            <div className="stack-lg animate-fade-in">
              <h2 style={{ fontSize: "2rem" }}>Contrato del Execution Planner</h2>
              <article className="stack-sm">
                <p className="eyebrow">Scope & Semantics</p>
                <div className="stack-sm" style={{ padding: "24px", background: "var(--panel)", borderRadius: "16px", border: "1px solid var(--border)" }}>
                   <div className="metric-row"><span>Scope</span><strong>{String(executionContract.scope || "N/A")}</strong></div>
                   <div className="metric-row"><span>Contract Version</span><strong>{String(detail.manifest?.contract_version || executionContract.version || "N/A")}</strong></div>
                   {!!executionContract.task_semantics && (
                      <div className="metric-row" style={{ marginTop: "12px" }}>
                        <span>Problem Family:</span>
                        <strong>{String((executionContract.task_semantics as Record<string, unknown>).problem_family || "")}</strong>
                      </div>
                   )}
                </div>
              </article>
              <article className="stack-sm">
                <p className="eyebrow">Raw Contract (JSON Object)</p>
                <CopyableFrame copyText={JSON.stringify(executionContract, null, 2)} variant="dark" copyLabel="Copiar">
                  <div className="code-frame" style={{ paddingTop: "36px" }}>
                    <pre>{JSON.stringify(executionContract, null, 2)}</pre>
                  </div>
                </CopyableFrame>
              </article>
            </div>
          )}

          {/* STEP 5: DATA ENGINEER */}
          {activeStep === "data_engineer" && (
            <div className="stack-lg animate-fade-in">
              <h2 style={{ fontSize: "2rem" }}>Trabajo del Data Engineer</h2>
              
              {loadingCleanedPreview ? (
                <article className="stack-sm">
                   <p className="eyebrow">Preview del CSV Limpio (Head)</p>
                   <p className="muted-copy">Cargando la vista previa del archivo físico limpiado...</p>
                </article>
              ) : finalCleanedPreview.length > 0 ? (
                <article className="stack-sm">
                  <p className="eyebrow">Preview del CSV Limpio (Head)</p>
                  {renderTable(finalCleanedPreview)}
                </article>
              ) : (
                <p className="muted-copy">No hay preview del dataset limpio para esta run.</p>
              )}

              <article className="stack-sm" style={{ marginTop: "16px" }}>
                <p className="eyebrow">Script de Limpieza Autogenerado</p>
                {cleaningCode ? (
                  <CopyableFrame copyText={cleaningCode} variant="dark" copyLabel="Copiar">
                    <div className="code-frame" style={{ paddingTop: "36px" }}>
                      <pre>{cleaningCode}</pre>
                    </div>
                  </CopyableFrame>
                ) : (
                   <p className="muted-copy">No se encontró el script cleaning_code.</p>
                )}
              </article>
            </div>
          )}

          {/* STEP 6: ML ENGINEER */}
          {activeStep === "ml_engineer" && (
            <div className="stack-lg animate-fade-in">
              <h2 style={{ fontSize: "2rem" }}>Trabajo del ML Engineer & Review Board</h2>
              
              <div style={{ padding: "24px", border: "1px solid var(--border)", borderRadius: "16px", background: "var(--panel)" }}>
                <div className="stack-sm">
                  <span className="insight-label">Veredicto del Review Board</span>
                  <div><StatusPill value={finalVerdict} /></div>
                  {requiredActions.length > 0 && (
                    <div style={{ marginTop: "16px" }}>
                      <p className="eyebrow">Acciones Mandatorias / Problemas Encontrados</p>
                      <ul className="fact-list" style={{ display: "grid", gap: "8px", paddingLeft: "16px" }}>
                        {requiredActions.map((action, idx) => (
                          <li key={idx}><span style={{ color: "var(--danger)", marginRight: "8px" }}>!</span>{action}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {!!reviewBoardVerdict.summary && (
                     <p style={{ marginTop: "16px", lineHeight: 1.6, fontSize: "0.95rem" }}>
                       {String(reviewBoardVerdict.summary)}
                     </p>
                  )}
                </div>
              </div>

              <article className="stack-sm">
                <p className="eyebrow">Script Final Elegido (Tras Iteraciones)</p>
                {generatedCode ? (
                  <CopyableFrame copyText={generatedCode} variant="dark" copyLabel="Copiar">
                    <details style={{ background: "#0d1530", borderRadius: "12px", overflow: "hidden" }}>
                      <summary style={{ padding: "16px 24px", color: "#edf2ff", cursor: "pointer", fontWeight: 600 }}>
                        Ver código fuente completo ({generatedCode.split("\\n").length} líneas)
                      </summary>
                      <div className="code-frame" style={{ margin: 0, border: "none", borderRadius: 0, background: "transparent" }}>
                        <pre style={{ margin: 0, padding: "24px", color: "#b4bfdc" }}>{generatedCode}</pre>
                      </div>
                    </details>
                  </CopyableFrame>
                ) : (
                   <p className="muted-copy">No se encontró script generado.</p>
                )}
              </article>

              {executionOutput && (
                <article className="stack-sm">
                  <p className="eyebrow">Salida Stdout Final</p>
                  <CopyableFrame copyText={executionOutput} variant="dark" copyLabel="Copiar">
                    <div className="code-frame" style={{ paddingTop: "36px" }}>
                      <pre>{executionOutput}</pre>
                    </div>
                  </CopyableFrame>
                </article>
              )}
            </div>
          )}

          {/* STEP 7: TRANSLATOR */}
          {activeStep === "translator" && (
            <div className="stack-lg animate-fade-in">
              <h2 style={{ fontSize: "2rem", marginBottom: "16px" }}>Reporte del Business Translator</h2>
              {report ? (
                 <RunReport runId={runId} report={report} />
              ) : (
                 <p className="muted-copy">El reporte ejecutivo aún no ha sido generado o no está disponible.</p>
              )}
            </div>
          )}

          </div>
        </main>
      </div>
    </div>
  );
}
