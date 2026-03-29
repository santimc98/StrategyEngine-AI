import { CreateRunPanel } from "@/components/create-run-panel";
import { StatusPill } from "@/components/status-pill";
import { fetchApiJson } from "@/lib/api";
import type { ActiveRunResponse } from "@/types/api";

export default async function HomePage() {
  let activeRun: ActiveRunResponse | null = null;
  let loadError: string | null = null;

  try {
    activeRun = await fetchApiJson<ActiveRunResponse>("/runs/active");
  } catch (error) {
    loadError = error instanceof Error ? error.message : "No se pudo cargar la API";
  }

  return (
    <div className="chat-home">
      <section className="chat-home-header">
        <div className="chat-home-copy">
          <p className="eyebrow">Nueva run</p>
          <h1>Plantea el objetivo de negocio y lanza una run gobernada desde una interfaz tipo chat.</h1>
          <p className="hero-text">
            Escribe lo que quieres resolver, adjunta un CSV local o usa un CSV generado desde el
            CRM de la empresa. El sistema coordina estrategia, ingenieria de datos, modelado,
            gobernanza e informe ejecutivo dentro de la misma run.
          </p>
        </div>

        {loadError ? <p className="inline-message danger">{loadError}</p> : null}

        <div className={`active-run-banner${activeRun?.active_run_id ? "" : " idle"}`}>
          {activeRun?.active_run_id ? (
            <>
              <div className="active-run-copy">
                <span className="active-run-label">Run activa</span>
                <strong>{activeRun.active_run_id}</strong>
                <p>
                  {String(activeRun.status?.stage_name || activeRun.status?.stage || "En ejecucion")}
                </p>
              </div>
              <div className="active-run-meta">
                <StatusPill value={String(activeRun.status?.status || "running")} />
                <a className="secondary-button" href={`/runs/${activeRun.active_run_id}`}>
                  Abrir run
                </a>
              </div>
            </>
          ) : (
            <div className="active-run-copy">
              <span className="active-run-label">Sin run activa</span>
              <strong>Todo listo para lanzar una nueva ejecucion</strong>
              <p>Escribe el objetivo y adjunta el dataset para empezar.</p>
            </div>
          )}
        </div>
      </section>

      <CreateRunPanel />

      <section className="chat-support-strip">
        <article className="support-card">
          <strong>CSV local</strong>
          <p>Sube el dataset directamente desde tu equipo y arranca la run sin salir de la pantalla principal.</p>
        </article>
        <article className="support-card">
          <strong>Conexion CRM</strong>
          <p>Tambien puedes lanzar la run con un CSV generado desde una integracion corporativa preparada en Settings.</p>
        </article>
        <article className="support-card">
          <strong>Historial y trazabilidad</strong>
          <p>El lateral mantiene el acceso a runs anteriores mientras la zona principal sigue centrada en la conversacion.</p>
        </article>
      </section>
    </div>
  );
}
