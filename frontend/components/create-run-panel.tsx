"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

type CreateRunResponse = {
  run_id: string;
};

export function CreateRunPanel() {
  const router = useRouter();
  const [csvPath, setCsvPath] = useState("");
  const [objective, setObjective] = useState("");
  const [replaceActiveRun, setReplaceActiveRun] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          csv_path: csvPath,
          business_objective: objective,
          replace_active_run: replaceActiveRun,
        }),
      });

      if (!response.ok) {
        const payload = await response.text();
        throw new Error(payload || "No se pudo crear la run");
      }

      const payload = (await response.json()) as CreateRunResponse;
      router.push(`/runs/${payload.run_id}`);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Nueva ejecución</p>
          <h2>Lanzar una run desde el frontend nuevo</h2>
        </div>
      </div>

      <form className="form-grid" onSubmit={handleSubmit}>
        <label className="field field-full">
          <span>Ruta al CSV</span>
          <input
            value={csvPath}
            onChange={(event) => setCsvPath(event.target.value)}
            placeholder="C:\data\dataset.csv"
            required
          />
        </label>

        <label className="field field-full">
          <span>Objetivo de negocio</span>
          <textarea
            value={objective}
            onChange={(event) => setObjective(event.target.value)}
            placeholder="Describe el objetivo de negocio que debe resolver la run."
            rows={5}
            required
          />
        </label>

        <label className="checkbox-row field-full">
          <input
            type="checkbox"
            checked={replaceActiveRun}
            onChange={(event) => setReplaceActiveRun(event.target.checked)}
          />
          <span>Abortar y reemplazar una run activa si existe</span>
        </label>

        {error ? <p className="form-error">{error}</p> : null}

        <div className="form-actions">
          <button className="primary-button" type="submit" disabled={submitting}>
            {submitting ? "Lanzando..." : "Lanzar run"}
          </button>
        </div>
      </form>
    </section>
  );
}
