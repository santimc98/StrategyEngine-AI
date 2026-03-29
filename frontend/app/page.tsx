import Link from "next/link";

import { CreateRunPanel } from "@/components/create-run-panel";
import { StatusPill } from "@/components/status-pill";
import { fetchApiJson } from "@/lib/api";
import { formatShortText } from "@/lib/format";
import type { ActiveRunResponse, RunListResponse } from "@/types/api";

export default async function HomePage() {
  let activeRun: ActiveRunResponse | null = null;
  let recentRuns: RunListResponse | null = null;
  let loadError: string | null = null;

  try {
    [activeRun, recentRuns] = await Promise.all([
      fetchApiJson<ActiveRunResponse>("/runs/active"),
      fetchApiJson<RunListResponse>("/runs?limit=5"),
    ]);
  } catch (error) {
    loadError = error instanceof Error ? error.message : "No se pudo cargar la API";
  }

  return (
    <div className="stack-xl">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Producto enterprise</p>
          <h1>Una interfaz real para lanzar, seguir y defender cada run.</h1>
          <p className="hero-text">
            Esta nueva capa frontend desacopla la experiencia de producto del backend multiagente
            y permite evolucionar la plataforma sin depender de Streamlit.
          </p>
        </div>
        <div className="hero-metrics">
          <div className="metric-card">
            <span>API backend</span>
            <strong>FastAPI activa</strong>
            <small>Runs, reportes, settings e integraciones ya expuestos</small>
          </div>
          <div className="metric-card">
            <span>Frontend objetivo</span>
            <strong>Next.js App Router</strong>
            <small>Shell profesional, routing real y componentes reutilizables</small>
          </div>
        </div>
      </section>

      {loadError ? (
        <section className="panel">
          <h2>No se pudo cargar la API</h2>
          <p className="form-error">{loadError}</p>
        </section>
      ) : null}

      <section className="overview-grid">
        <article className="panel nested">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Run activa</p>
              <h2>Estado operativo actual</h2>
            </div>
          </div>
          {activeRun?.active_run_id ? (
            <div className="stack-sm">
              <div className="metric-row">
                <span>Run ID</span>
                <strong>{activeRun.active_run_id}</strong>
              </div>
              <div className="metric-row">
                <span>Estado</span>
                <StatusPill value={String(activeRun.status?.status || "running")} />
              </div>
              <div className="metric-row">
                <span>Stage</span>
                <strong>{String(activeRun.status?.stage_name || activeRun.status?.stage || "N/A")}</strong>
              </div>
              <Link className="secondary-button" href={`/runs/${activeRun.active_run_id}`}>
                Abrir detalle de la run
              </Link>
            </div>
          ) : (
            <p className="muted-copy">Ahora mismo no hay ninguna run activa.</p>
          )}
        </article>

        <article className="panel nested">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Histórico reciente</p>
              <h2>Últimas runs</h2>
            </div>
            <Link className="secondary-button" href="/runs">
              Ver todas
            </Link>
          </div>
          <div className="stack-sm">
            {(recentRuns?.items || []).map((run) => (
              <div key={run.run_id} className="list-card">
                <div className="list-card-top">
                  <Link href={`/runs/${run.run_id}`}>{run.run_id}</Link>
                  <StatusPill value={run.verdict || run.status} />
                </div>
                <p>{formatShortText(run.business_objective, 110)}</p>
              </div>
            ))}
          </div>
        </article>
      </section>

      <CreateRunPanel />
    </div>
  );
}
