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
    loadError = error instanceof Error ? error.message : "No se pudo cargar la configuracion";
  }

  return (
    <div className="stack-xl">
      <section className="page-intro">
        <div className="page-intro-copy">
          <p className="eyebrow">Settings</p>
          <h1>Configuracion operativa del producto.</h1>
          <p className="hero-text">
            Controla modelos por agente, sandbox, credenciales e integraciones desde una vista
            unica pensada como superficie de producto y no como consola auxiliar.
          </p>
        </div>
        {models && apiKeys ? (
          <div className="page-intro-side">
            <span className="page-kpi-label">Agentes configurables</span>
            <strong>{models.agents.length}</strong>
            <p>{apiKeys.items.filter((item) => item.configured).length} credenciales ya configuradas.</p>
          </div>
        ) : null}
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
