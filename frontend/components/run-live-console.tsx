"use client";

import { useEffect, useRef, useState } from "react";

import { StatusPill } from "@/components/status-pill";
import type { JsonRecord, RunLogsResponse } from "@/types/api";

type RunLiveConsoleProps = {
  runId: string;
  initialStatus: JsonRecord;
};

type LogEntry = RunLogsResponse["entries"][number];

export function RunLiveConsole({ runId, initialStatus }: RunLiveConsoleProps) {
  const [status, setStatus] = useState<JsonRecord>(initialStatus || {});
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [aborting, setAborting] = useState(false);
  const nextAfterLineRef = useRef(0);

  useEffect(() => {
    async function poll(): Promise<void> {
      try {
        const [statusResponse, logsResponse] = await Promise.all([
          fetch(`/api/runs/${runId}/status`, { cache: "no-store" }),
          fetch(`/api/runs/${runId}/logs?after_line=${nextAfterLineRef.current}`, {
            cache: "no-store",
          }),
        ]);

        if (!statusResponse.ok) {
          throw new Error("No se pudo refrescar el estado");
        }

        const statusPayload = (await statusResponse.json()) as JsonRecord;
        setStatus(statusPayload);

        if (logsResponse.ok) {
          const logsPayload = (await logsResponse.json()) as RunLogsResponse;
          if (logsPayload.entries.length) {
            setLogs((current) => [...current, ...logsPayload.entries].slice(-200));
          }
          nextAfterLineRef.current = logsPayload.next_after_line;
        }

        setError(null);
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
  }, [runId]);

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

  const stage = String(status.stage_name || status.stage || "N/A");
  const metricName = String(status.metric_name || "");
  const metricValue = String(status.metric_value || "");
  const completedSteps = Array.isArray(status.completed_steps) ? status.completed_steps : [];
  const isRunning = String(status.status || "").toLowerCase() === "running";

  return (
    <div className="stack-lg">
      <div className="section-head">
        <div>
          <p className="eyebrow">Seguimiento en vivo</p>
          <h2>Estado operativo y últimos logs</h2>
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
              <span>Iteración</span>
              <strong>
                {String(status.iteration ?? 0)} / {String(status.max_iterations ?? "N/A")}
              </strong>
            </div>
            <div className="metric-row">
              <span>Métrica</span>
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
              {loading ? "Cargando estado..." : "Aún no hay pasos completados reportados."}
            </p>
          )}
        </article>
      </div>

      {error ? <p className="form-error">{error}</p> : null}

      <div className="logs-frame">
        {logs.length ? (
          logs.map((entry, index) => (
            <div key={`${entry.ts}-${entry.agent}-${index}`} className="log-entry">
              <span className="log-time">{entry.ts}</span>
              <span className="log-agent">{entry.agent}</span>
              <span className="log-message">{entry.msg}</span>
            </div>
          ))
        ) : (
          <p className="muted-copy">{loading ? "Cargando logs..." : "Todavía no hay logs visibles."}</p>
        )}
      </div>
    </div>
  );
}
