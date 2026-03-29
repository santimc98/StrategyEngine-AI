"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import { useRouter } from "next/navigation";

type CreateRunResponse = {
  run_id: string;
};

type AttachmentMode = "upload" | "path" | "crm";

export function CreateRunPanel() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [objective, setObjective] = useState("");
  const [replaceActiveRun, setReplaceActiveRun] = useState(false);
  const [attachmentMode, setAttachmentMode] = useState<AttachmentMode>("upload");
  const [csvPath, setCsvPath] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function resolveCsvPath(): Promise<string> {
    if (attachmentMode === "upload") {
      if (!selectedFile) {
        throw new Error("Adjunta un CSV local antes de lanzar la run.");
      }

      const uploadResponse = await fetch("/api/datasets/upload", {
        method: "POST",
        headers: {
          "x-filename": selectedFile.name,
          "content-type": selectedFile.type || "text/csv",
        },
        body: selectedFile,
      });

      if (!uploadResponse.ok) {
        throw new Error(await uploadResponse.text());
      }

      const uploadPayload = (await uploadResponse.json()) as { csv_path: string };
      return uploadPayload.csv_path;
    }

    const trimmed = csvPath.trim();
    if (!trimmed) {
      throw new Error("Indica la ruta del CSV antes de lanzar la run.");
    }
    return trimmed;
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const resolvedCsvPath = await resolveCsvPath();
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          csv_path: resolvedCsvPath,
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

  const attachmentLabel = selectedFile
    ? selectedFile.name
    : csvPath.trim() || "Ningun dataset adjunto";

  const modeDescription =
    attachmentMode === "upload"
      ? "Sube un CSV local desde tu equipo y lo guardaremos en el workspace de la plataforma."
      : attachmentMode === "path"
        ? "Usa una ruta de CSV ya existente dentro del entorno donde corre el producto."
        : "Lanza la run con un CSV generado previamente desde una integracion CRM configurada en Settings.";

  return (
    <section className="composer-panel">
      <div className="composer-surface">
        <div className="composer-header">
          <p className="eyebrow">Prompt inicial</p>
          <h2>Describe el objetivo de negocio y adjunta el dataset de partida.</h2>
          <p className="helper-copy">
            El sistema utilizara este mensaje como punto de partida de la run y conservara toda la
            trazabilidad tecnica y ejecutiva.
          </p>
        </div>

        <div className="composer-mode-row">
          <button
            type="button"
            className={`mode-chip${attachmentMode === "upload" ? " active" : ""}`}
            onClick={() => setAttachmentMode("upload")}
          >
            CSV local
          </button>
          <button
            type="button"
            className={`mode-chip${attachmentMode === "path" ? " active" : ""}`}
            onClick={() => setAttachmentMode("path")}
          >
            Ruta CSV
          </button>
          <button
            type="button"
            className={`mode-chip${attachmentMode === "crm" ? " active" : ""}`}
            onClick={() => setAttachmentMode("crm")}
          >
            CRM
          </button>
        </div>

        <p className="mode-helper">{modeDescription}</p>

        {attachmentMode === "upload" ? (
          <div className="attachment-panel">
            <button
              type="button"
              className="secondary-button"
              onClick={() => fileInputRef.current?.click()}
            >
              Adjuntar CSV local
            </button>
            <span className="attachment-chip">{selectedFile ? selectedFile.name : "Sin archivo"}</span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              className="hidden-input"
              onChange={(event) => {
                const file = event.target.files?.[0] || null;
                setSelectedFile(file);
                if (file) {
                  setCsvPath("");
                }
              }}
            />
          </div>
        ) : null}

        {attachmentMode === "path" ? (
          <label className="field field-full">
            <span>Ruta del CSV</span>
            <input
              value={csvPath}
              onChange={(event) => {
                setCsvPath(event.target.value);
                setSelectedFile(null);
              }}
              placeholder="C:\\data\\dataset.csv"
            />
          </label>
        ) : null}

        {attachmentMode === "crm" ? (
          <div className="crm-attach-box">
            <p className="helper-copy">
              Usa aqui la ruta de un CSV generado desde una integracion CRM corporativa. Puedes
              preparar esa extraccion desde <Link href="/settings">Settings</Link>.
            </p>
            <label className="field field-full">
              <span>Ruta CSV generada desde CRM</span>
              <input
                value={csvPath}
                onChange={(event) => {
                  setCsvPath(event.target.value);
                  setSelectedFile(null);
                }}
                placeholder="C:\\Users\\santi\\Projects\\Hackathon_Gemini_Agents\\data\\crm_deals.csv"
              />
            </label>
          </div>
        ) : null}

        <form className="chat-form" onSubmit={handleSubmit}>
          <textarea
            value={objective}
            onChange={(event) => setObjective(event.target.value)}
            className="prompt-composer"
            placeholder="Ejemplo: Queremos construir un modelo prescriptivo de pricing para contratos de mantenimiento, maximizando conversion y preservando coherencia entre segmentos de cliente."
            rows={8}
            required
          />

          <div className="composer-toolbar">
            <div className="composer-meta">
              <span className="attachment-chip subtle">{attachmentLabel}</span>
              <label className="checkbox-inline">
                <input
                  type="checkbox"
                  checked={replaceActiveRun}
                  onChange={(event) => setReplaceActiveRun(event.target.checked)}
                />
                <span>Reemplazar run activa</span>
              </label>
            </div>

            <button className="primary-button send-button" type="submit" disabled={submitting}>
              {submitting ? "Lanzando..." : "Iniciar run"}
            </button>
          </div>

          {error ? <p className="form-error">{error}</p> : null}
        </form>
      </div>
    </section>
  );
}
