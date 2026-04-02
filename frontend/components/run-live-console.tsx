"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { StatusPill } from "@/components/status-pill";
import type { JsonRecord, RunActivityEntry, RunActivityResponse, RunLogsResponse } from "@/types/api";

type RunLiveConsoleProps = {
  runId: string;
  initialStatus: JsonRecord;
};

type LogEntry = RunLogsResponse["entries"][number];

export function RunLiveConsole({ runId, initialStatus }: RunLiveConsoleProps) {
  const router = useRouter();
  const [status, setStatus] = useState<JsonRecord>(initialStatus || {});
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [activity, setActivity] = useState<RunActivityEntry[]>([]);
  const [activitySnapshot, setActivitySnapshot] = useState<JsonRecord>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [aborting, setAborting] = useState(false);
  const nextAfterLineRef = useRef(0);
  const nextAfterActivityRef = useRef(0);
  const finishedRef = useRef(false);

  useEffect(() => {
    async function poll(): Promise<void> {
      if (finishedRef.current) return;
      try {
        const [statusResponse, logsResponse, activityResponse] = await Promise.all([
          fetch(`/api/runs/${runId}/status`, { cache: "no-store" }),
          fetch(`/api/runs/${runId}/logs?after_line=${nextAfterLineRef.current}`, {
            cache: "no-store",
          }),
          fetch(`/api/runs/${runId}/activity?after_line=${nextAfterActivityRef.current}`, {
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

        if (activityResponse.ok) {
          const activityPayload = (await activityResponse.json()) as RunActivityResponse;
          if (activityPayload.entries.length) {
            setActivity((current) => [...current, ...activityPayload.entries].slice(-60));
          }
          setActivitySnapshot(activityPayload.snapshot || {});
          nextAfterActivityRef.current = activityPayload.next_after_line;
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
  const latestActivity = activity[activity.length - 1] || null;
  const currentTitle = String(
    activitySnapshot.latest_title ||
      latestActivity?.title ||
      (isRunning ? "La run sigue avanzando" : "La run ya terminó"),
  );
  const currentSummary = String(
    activitySnapshot.latest_summary ||
      latestActivity?.summary ||
      (isRunning
        ? "La plataforma sigue procesando pasos internos del pipeline."
        : "Ya no hay pasos activos en ejecución."),
  );
  const currentPhase = String(activitySnapshot.latest_phase || latestActivity?.phase || stage || "");
  const currentTs = String(activitySnapshot.latest_ts || latestActivity?.ts || "");
  const currentDetails = latestActivity?.details || [];

  return (
    <div className="stack-lg">
      <div className="section-head" style={{ marginBottom: "16px" }}>
        <div>
          <p className="eyebrow">Seguimiento en vivo</p>
          <h2>Estado operativo, actividad interna y logs</h2>
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
      
      {/* ProgressBar Element */}
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
              <p className="eyebrow">Ahora mismo</p>
              <h2>{currentTitle}</h2>
            </div>
          </div>
          <p className="muted-copy">{currentSummary}</p>
          <div className="status-inline-meta">
            {currentPhase ? <span className="status-inline-chip">{currentPhase}</span> : null}
            {currentTs ? <span className="status-inline-chip">{currentTs}</span> : null}
          </div>
          {currentDetails.length ? (
            <div className="activity-detail-list">
              {currentDetails.map((detail) => (
                <span key={`${detail.label}-${detail.value}`} className="activity-detail">
                  <strong>{detail.label}</strong>
                  <span>{detail.value}</span>
                </span>
              ))}
            </div>
          ) : null}
        </article>
      </div>

      <article className="stack-sm">
        <div className="section-head">
          <div>
            <p className="eyebrow">Pipeline visible</p>
            <h2>Pasos reportados por el worker</h2>
          </div>
        </div>
        {completedSteps.length ? (
          <div className="status-inline-meta">
            {completedSteps.map((step) => (
              <span key={step} className="status-inline-chip">
                {step}
              </span>
            ))}
          </div>
        ) : (
          <p className="muted-copy">
            {loading ? "Cargando estado..." : "Aún no hay pasos completados reportados."}
          </p>
        )}
      </article>

      {error ? <p className="form-error">{error}</p> : null}

      <div className="activity-frame">
        <div className="section-head">
          <div>
            <p className="eyebrow">Actividad interna</p>
            <h2>Lo que está ocurriendo dentro del sistema</h2>
          </div>
        </div>
        {activity.length ? (
          activity
            .slice()
            .reverse()
            .map((entry) => (
              <div key={`${entry.index}-${entry.event}`} className={`activity-entry ${entry.level || "info"}`}>
                <div className="activity-entry-top">
                  <div>
                    <p className="activity-kicker">
                      {entry.ts} · {entry.phase || entry.event}
                    </p>
                    <h3 className="activity-title">{entry.title}</h3>
                  </div>
                  <span className="status-inline-chip">{entry.event}</span>
                </div>
                <p className="activity-summary">{entry.summary}</p>
                {entry.details.length ? (
                  <div className="activity-detail-list">
                    {entry.details.map((detail) => (
                      <span key={`${entry.index}-${detail.label}-${detail.value}`} className="activity-detail">
                        <strong>{detail.label}</strong>
                        <span>{detail.value}</span>
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            ))
        ) : (
          <p className="muted-copy">
            {loading ? "Cargando actividad interna..." : "Todavía no hay actividad interna visible."}
          </p>
        )}
      </div>

      <div className="stack-sm">
        <div className="section-head">
          <div>
            <p className="eyebrow">Logs del worker</p>
            <h2>Salida cruda de ejecución</h2>
          </div>
        </div>
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
    </div>
  );
}
