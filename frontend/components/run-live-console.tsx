"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { StatusPill } from "@/components/status-pill";
import type { JsonRecord, RunEventLogResponse, RunLogsResponse } from "@/types/api";

type RunLiveConsoleProps = {
  runId: string;
  initialStatus: JsonRecord;
};

type LogEntry = {
  ts: string;
  agent: string;
  msg: string;
  level: string;
  source: "worker" | "event";
};

export function RunLiveConsole({ runId, initialStatus }: RunLiveConsoleProps) {
  const router = useRouter();
  const [status, setStatus] = useState<JsonRecord>(initialStatus || {});
  const [workerLogs, setWorkerLogs] = useState<LogEntry[]>([]);
  const [eventLogs, setEventLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [aborting, setAborting] = useState(false);
  const nextWorkerLineRef = useRef(0);
  const nextEventLineRef = useRef(0);
  const finishedRef = useRef(false);
  const consoleEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    async function poll(): Promise<void> {
      if (finishedRef.current) return;
      try {
        const [statusResponse, logsResponse, eventResponse] = await Promise.all([
          fetch(`/api/runs/${runId}/status`, { cache: "no-store" }),
          fetch(`/api/runs/${runId}/logs?after_line=${nextWorkerLineRef.current}`, {
            cache: "no-store",
          }),
          fetch(`/api/runs/${runId}/activity?after_line=${nextEventLineRef.current}`, {
            cache: "no-store",
          }),
        ]);

        if (!statusResponse.ok) {
          throw new Error("No se pudo refrescar el estado");
        }

        const statusPayload = (await statusResponse.json()) as JsonRecord;
        setStatus(statusPayload);

        if (logsResponse.ok) {
          const logsPayload = (await logsResponse.json()) as Partial<RunLogsResponse>;
          const incoming = Array.isArray(logsPayload?.entries) ? logsPayload.entries : [];
          if (incoming.length) {
            const newEntries: LogEntry[] = incoming.map((e) => ({
              ts: String(e?.ts ?? ""),
              agent: String(e?.agent ?? "Sistema"),
              msg: String(e?.msg ?? ""),
              level: String(e?.level ?? "info"),
              source: "worker" as const,
            }));
            setWorkerLogs((current) => [...current, ...newEntries].slice(-300));
          }
          if (typeof logsPayload?.next_after_line === "number") {
            nextWorkerLineRef.current = logsPayload.next_after_line;
          }
        }

        if (eventResponse.ok) {
          const eventPayload = (await eventResponse.json()) as Partial<RunEventLogResponse>;
          const incoming = Array.isArray(eventPayload?.entries) ? eventPayload.entries : [];
          if (incoming.length) {
            const newEntries: LogEntry[] = incoming.map((e) => ({
              ts: String(e?.ts ?? ""),
              agent: String(e?.agent ?? "Sistema"),
              msg: String(e?.msg ?? ""),
              level: String(e?.level ?? "info"),
              source: "event" as const,
            }));
            setEventLogs((current) => [...current, ...newEntries].slice(-300));
          }
          if (typeof eventPayload?.next_after_line === "number") {
            nextEventLineRef.current = eventPayload.next_after_line;
          }
        }

        setError(null);

        const currentIsRunning = String(statusPayload.status || "").toLowerCase() === "running";
        if (!currentIsRunning && !finishedRef.current) {
          finishedRef.current = true;
          router.refresh();
        }

      } catch (err) {
        setError(err instanceof Error ? err.message : "Error refrescando la run");
      } finally {
        setLoading(false);
      }
    }

    const interval = window.setInterval(() => {
      void poll();
    }, 3000);
    void poll();

    return () => window.clearInterval(interval);
  }, [runId, router]);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [workerLogs, eventLogs]);

  async function handleAbort(): Promise<void> {
    setAborting(true);
    setError(null);
    try {
      const response = await fetch(`/api/runs/${runId}/abort`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ force_kill: false }),
      });
      if (!response.ok) {
        throw new Error("No se pudo solicitar el aborto de la run");
      }
      const statusResponse = await fetch(`/api/runs/${runId}/status`, { cache: "no-store" });
      if (statusResponse.ok) {
        const statusPayload = (await statusResponse.json()) as JsonRecord;
        setStatus(statusPayload);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo abortar la run");
    } finally {
      setAborting(false);
    }
  }

  // Merge and sort both log streams by timestamp (defensive against missing fields)
  const mergedLogs = [...workerLogs, ...eventLogs]
    .filter((e): e is LogEntry => Boolean(e) && typeof e.ts === "string")
    .sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
  // Deduplicate: if worker and event have the same ts+agent, keep worker only
  const deduped = mergedLogs.filter((entry, index, arr) => {
    if (entry.source !== "event") return true;
    const start = Math.max(0, index - 3);
    const end = Math.min(arr.length, index + 4);
    for (let i = start; i < end; i++) {
      if (i === index) continue;
      const other = arr[i];
      if (other && other.source === "worker" && other.ts === entry.ts && other.agent === entry.agent) {
        return false;
      }
    }
    return true;
  });

  const stage = String(status.stage_name || status.stage || "N/A");
  const metricName = String(status.metric_name || "");
  const metricValue = String(status.metric_value || "");
  const completedSteps = Array.isArray(status.completed_steps) ? status.completed_steps : [];
  const isRunning = String(status.status || "").toLowerCase() === "running";

  function levelClass(level: string): string {
    switch (level) {
      case "ok":
      case "success":
        return "log-level-ok";
      case "warn":
      case "warning":
        return "log-level-warn";
      case "error":
        return "log-level-error";
      default:
        return "";
    }
  }

  return (
    <div className="stack-lg">
      <div className="section-head" style={{ marginBottom: "16px" }}>
        <div>
          <p className="eyebrow">Seguimiento en vivo</p>
          <h2>Estado operativo y consola de eventos</h2>
        </div>
        <div className="panel-actions">
          <StatusPill value={String(status.status || "N/A")} />
          {isRunning ? (
            <button className="secondary-button" onClick={handleAbort} disabled={aborting}>
              {aborting ? "Abortando..." : "Abortar run"}
            </button>
          ) : null}
        </div>
      </div>

      {/* Progress Bar */}
      <div style={{ width: "100%", height: "6px", background: "var(--border)", borderRadius: "4px", overflow: "hidden", marginBottom: "8px" }}>
        <div
          style={{
            height: "100%",
            width: `${Math.min(100, Math.max(0, Number(status.progress) || 0))}%`,
            background: "linear-gradient(90deg, var(--accent), var(--accent-2))",
            transition: "width 0.4s cubic-bezier(0.4, 0, 0.2, 1)"
          }}
        />
      </div>

      <div className="overview-grid live-grid">
        <article className="stack-sm">
          <div className="stack-sm">
            <div className="metric-row">
              <span>Stage</span>
              <strong>{stage}</strong>
            </div>
            <div className="metric-row">
              <span>Progreso</span>
              <strong>{String(status.progress ?? "N/A")}%</strong>
            </div>
            <div className="metric-row">
              <span>Iteracion</span>
              <strong>
                {String(status.iteration ?? 0)} / {String(status.max_iterations ?? "N/A")}
              </strong>
            </div>
            <div className="metric-row">
              <span>Metrica</span>
              <strong>{metricName ? `${metricName} ${metricValue}` : "N/A"}</strong>
            </div>
          </div>
        </article>

        <article className="stack-sm">
          <div className="section-head">
            <div>
              <p className="eyebrow">Steps completados</p>
              <h2>Pipeline visible</h2>
            </div>
          </div>
          {completedSteps.length ? (
            <ul className="fact-list">
              {completedSteps.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ul>
          ) : (
            <p className="muted-copy">
              {loading ? "Cargando estado..." : "Aun no hay pasos completados reportados."}
            </p>
          )}
        </article>
      </div>

      {error ? <p className="form-error">{error}</p> : null}

      <div className="stack-sm">
        <div className="section-head">
          <div>
            <p className="eyebrow">Consola de eventos</p>
            <h2>Timeline completo del pipeline</h2>
          </div>
          <span className="muted-copy" style={{ fontSize: "0.8rem" }}>
            {deduped.length} entradas
          </span>
        </div>
        <div className="logs-frame">
          {deduped.length ? (
            <>
              {deduped.map((entry, index) => (
                <div
                  key={`${entry.ts}-${entry.agent}-${entry.source}-${index}`}
                  className={`log-entry ${levelClass(entry.level)}`}
                >
                  <span className="log-time">{entry.ts}</span>
                  <span className="log-agent">{entry.agent}</span>
                  <span className="log-message">{entry.msg}</span>
                </div>
              ))}
              <div ref={consoleEndRef} />
            </>
          ) : (
            <p className="muted-copy">{loading ? "Cargando logs..." : "Todavia no hay logs visibles."}</p>
          )}
        </div>
      </div>
    </div>
  );
}
