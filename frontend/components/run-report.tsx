import type { RunReportResponse } from "@/types/api";

type RunReportProps = {
  runId: string;
  report: RunReportResponse;
};

export function RunReport({ runId, report }: RunReportProps) {
  return (
    <div className="stack-lg">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Informe ejecutivo</p>
            <h2>Reporte final de la run</h2>
          </div>
          {report.pdf_available && report.pdf_url ? (
            <a className="secondary-button" href={`/api${report.pdf_url}`} target="_blank" rel="noreferrer">
              Abrir PDF
            </a>
          ) : null}
        </div>
        <div className="markdown-frame">
          <pre>{report.markdown || "No hay reporte disponible."}</pre>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Plots curados</p>
            <h2>Artefactos visuales de la run</h2>
          </div>
        </div>
        <div className="plot-grid">
          {report.plots.map((plot) => (
            <article key={plot.filename} className="plot-card">
              <img
                src={`/api/runs/${runId}/report/plots/${plot.filename}`}
                alt={plot.title}
                className="plot-image"
              />
              <div className="plot-card-body">
                <h3>{plot.title}</h3>
                {plot.facts.length ? (
                  <ul className="fact-list">
                    {plot.facts.slice(0, 4).map((fact) => (
                      <li key={fact}>{fact}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted-copy">Plot disponible sin facts resumidos.</p>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
