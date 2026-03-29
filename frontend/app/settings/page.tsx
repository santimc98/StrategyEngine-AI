import { SettingsWorkspace } from "@/components/settings-workspace";
import { fetchApiJson } from "@/lib/api";
import type {
  ApiKeysResponse,
  ConnectorSpecsResponse,
  ModelsConfigResponse,
  SandboxConfigResponse,
} from "@/types/api";

export default async function SettingsPage() {
  let models: ModelsConfigResponse | null = null;
  let sandbox: SandboxConfigResponse | null = null;
  let apiKeys: ApiKeysResponse | null = null;
  let connectors: ConnectorSpecsResponse | null = null;
  let loadError: string | null = null;

  try {
    [models, sandbox, apiKeys, connectors] = await Promise.all([
      fetchApiJson<ModelsConfigResponse>("/config/models"),
      fetchApiJson<SandboxConfigResponse>("/config/sandbox"),
      fetchApiJson<ApiKeysResponse>("/config/api-keys"),
      fetchApiJson<ConnectorSpecsResponse>("/integrations/connectors"),
    ]);
  } catch (error) {
    loadError = error instanceof Error ? error.message : "No se pudo cargar la configuración";
  }

  return (
    <div className="stack-xl">
      <section className="hero compact">
        <div className="hero-copy">
          <p className="eyebrow">Settings</p>
          <h1>Configuración operativa del producto.</h1>
          <p className="hero-text">
            Esta vista resume el estado efectivo de modelos, sandbox, proveedores y conectores sin
            depender del sidebar de Streamlit.
          </p>
        </div>
      </section>

      {loadError ? (
        <section className="panel">
          <p className="form-error">{loadError}</p>
        </section>
      ) : null}

      {models && sandbox && apiKeys && connectors ? (
        <SettingsWorkspace
          initialModels={models}
          initialSandbox={sandbox}
          initialApiKeys={apiKeys}
          initialConnectors={connectors}
        />
      ) : null}
    </div>
  );
}
