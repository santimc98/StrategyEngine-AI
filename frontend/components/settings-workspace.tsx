"use client";

import { useMemo, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

import { StatusPill } from "@/components/status-pill";
import type {
  ApiKeysResponse,
  ConnectorSpecsResponse,
  ModelsConfigResponse,
  SandboxConfigResponse,
} from "@/types/api";

type SettingsWorkspaceProps = {
  initialModels: ModelsConfigResponse;
  initialSandbox: SandboxConfigResponse;
  initialApiKeys: ApiKeysResponse;
  initialConnectors: ConnectorSpecsResponse;
};

type MessageState = {
  tone: "success" | "warning" | "danger" | "neutral";
  text: string;
} | null;

type SettingsTabKey = "models" | "sandbox" | "keys" | "connectors";

const settingsTabs: Array<{ key: SettingsTabKey; label: string }> = [
  { key: "models", label: "Modelos" },
  { key: "sandbox", label: "Sandbox" },
  { key: "keys", label: "Credenciales" },
  { key: "connectors", label: "Conectores CRM" },
];

function sanitizeObject(payload: Record<string, unknown>): Record<string, string> {
  const cleaned: Record<string, string> = {};
  Object.entries(payload).forEach(([key, value]) => {
    if (value !== null && typeof value === "object") {
      return;
    }
    const text = String(value ?? "").trim();
    if (text) {
      cleaned[key] = text;
    }
  });
  return cleaned;
}

export function SettingsWorkspace({
  initialModels,
  initialSandbox,
  initialApiKeys,
  initialConnectors,
}: SettingsWorkspaceProps) {
  const router = useRouter();
  const [isRefreshing, startTransition] = useTransition();
  const [activeTab, setActiveTab] = useState<SettingsTabKey>("models");

  const [modelsState, setModelsState] = useState(initialModels);
  const [sandboxState, setSandboxState] = useState(initialSandbox);
  const [apiKeysState, setApiKeysState] = useState(initialApiKeys);

  const [modelInputs, setModelInputs] = useState<Record<string, string>>(initialModels.effective_models);
  const [customModeAgents, setCustomModeAgents] = useState<Set<string>>(new Set());
  const [provider, setProvider] = useState(initialSandbox.provider);
  const [providerSettings, setProviderSettings] = useState<Record<string, string>>(
    sanitizeObject(
      Object.fromEntries(
        Object.entries(
          ((initialSandbox.config.settings as Record<string, unknown> | undefined) || {}) as Record<
            string,
            unknown
          >,
        ).filter(([key]) => key !== "execution_backend"),
      ),
    ),
  );
  const [executionBackend, setExecutionBackend] = useState<Record<string, string>>(
    sanitizeObject((initialSandbox.execution_backend as Record<string, unknown>) || {}),
  );
  const [apiKeyInputs, setApiKeyInputs] = useState<Record<string, string>>({});
  const [selectedConnectorId, setSelectedConnectorId] = useState(initialConnectors.items[0]?.id || "");
  const [selectedAuthModeId, setSelectedAuthModeId] = useState(
    initialConnectors.items[0]?.auth_modes[0]?.id || "",
  );
  const [connectorCredentials, setConnectorCredentials] = useState<Record<string, string>>({});
  const [connectorObjectName, setConnectorObjectName] = useState("");
  const [connectorObjects, setConnectorObjects] = useState<string[]>([]);
  const [connectorPreview, setConnectorPreview] = useState<Record<string, unknown>[]>([]);
  const [connectorCsvPath, setConnectorCsvPath] = useState("");

  const [modelsMessage, setModelsMessage] = useState<MessageState>(null);
  const [sandboxMessage, setSandboxMessage] = useState<MessageState>(null);
  const [apiKeysMessage, setApiKeysMessage] = useState<MessageState>(null);
  const [connectorMessage, setConnectorMessage] = useState<MessageState>(null);

  const [savingModels, setSavingModels] = useState(false);
  const [savingSandbox, setSavingSandbox] = useState(false);
  const [savingApiKey, setSavingApiKey] = useState<string | null>(null);
  const [connectorBusyAction, setConnectorBusyAction] = useState<string | null>(null);

  const groupedAgents = useMemo(() => {
    return {
      primary: modelsState.agents.filter((item) => item.section === "primary"),
      advanced: modelsState.agents.filter((item) => item.section === "advanced"),
    };
  }, [modelsState.agents]);

  const customModelOption = modelsState.custom_option || "__custom_model__";
  const presetModels = useMemo(() => {
    return new Map(modelsState.presets.map((preset) => [preset.id, preset.label]));
  }, [modelsState.presets]);

  const selectedProviderSpec = useMemo(() => {
    return sandboxState.providers.find((item) => item.name === provider) || sandboxState.providers[0];
  }, [provider, sandboxState.providers]);

  const selectedConnector = useMemo(() => {
    return initialConnectors.items.find((item) => item.id === selectedConnectorId) || null;
  }, [initialConnectors.items, selectedConnectorId]);

  const selectedAuthMode = useMemo(() => {
    if (!selectedConnector) {
      return null;
    }
    return (
      selectedConnector.auth_modes.find((mode) => mode.id === selectedAuthModeId) ||
      selectedConnector.auth_modes[0] ||
      null
    );
  }, [selectedAuthModeId, selectedConnector]);

  const configuredRequiredKeys = useMemo(() => {
    return apiKeysState.items.filter((item) => item.required && item.configured).length;
  }, [apiKeysState.items]);

  const totalRequiredKeys = useMemo(() => {
    return apiKeysState.items.filter((item) => item.required).length;
  }, [apiKeysState.items]);

  function refreshServerState(): void {
    startTransition(() => {
      router.refresh();
    });
  }

  function resolveModelSelectValue(agentKey: string): string {
    const currentValue = String(modelInputs[agentKey] || "").trim();
    if (customModeAgents.has(agentKey)) {
      // Mientras se edita, el select siempre muestra "Modelo personalizado…",
      // independientemente de lo que haya en el input (pegar, escribir, etc.).
      return customModelOption;
    }
    if (!currentValue) {
      return "";
    }
    return presetModels.has(currentValue) ? currentValue : currentValue;
  }

  function commitCustomModel(agentKey: string, explicitValue?: string): void {
    const fromState = String(modelInputs[agentKey] || "").trim();
    const value = (explicitValue !== undefined ? explicitValue : fromState).trim();
    if (!value) {
      return;
    }
    // Asegura que el valor del input quede normalizado (trim) en el estado.
    setModelInputs((current) => ({ ...current, [agentKey]: value }));
    setCustomModeAgents((current) => {
      if (!current.has(agentKey)) {
        return current;
      }
      const next = new Set(current);
      next.delete(agentKey);
      return next;
    });
  }

  function updateModelPreset(agentKey: string, nextValue: string): void {
    if (nextValue === customModelOption) {
      setCustomModeAgents((current) => {
        const next = new Set(current);
        next.add(agentKey);
        return next;
      });
      setModelInputs((current) => {
        const currentValue = String(current[agentKey] || "").trim();
        // If the current value is a preset, clear it so the user can type freely.
        // If it's already a custom string, preserve it.
        if (presetModels.has(currentValue)) {
          return { ...current, [agentKey]: "" };
        }
        return current;
      });
      return;
    }
    setCustomModeAgents((current) => {
      if (!current.has(agentKey)) {
        return current;
      }
      const next = new Set(current);
      next.delete(agentKey);
      return next;
    });
    setModelInputs((current) => ({ ...current, [agentKey]: nextValue || "" }));
  }

  async function saveModels(): Promise<void> {
    setSavingModels(true);
    setModelsMessage(null);
    try {
      const response = await fetch("/api/config/models", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ models: modelInputs }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as Partial<ModelsConfigResponse>;
      const nextState: ModelsConfigResponse = {
        ...modelsState,
        persisted_models: payload.saved_models || modelsState.persisted_models,
        effective_models: payload.effective_models || modelsState.effective_models,
      };
      setModelsState(nextState);
      setModelInputs(nextState.effective_models);
      setModelsMessage({ tone: "success", text: "Configuracion de modelos guardada." });
      refreshServerState();
    } catch (error) {
      setModelsMessage({
        tone: "danger",
        text: error instanceof Error ? error.message : "No se pudieron guardar los modelos.",
      });
    } finally {
      setSavingModels(false);
    }
  }

  async function resetModels(): Promise<void> {
    setSavingModels(true);
    setModelsMessage(null);
    try {
      const response = await fetch("/api/config/models/reset", { method: "POST" });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as Partial<ModelsConfigResponse>;
      const nextEffective = (payload.effective_models || {}) as Record<string, string>;
      setModelsState((current) => ({
        ...current,
        persisted_models: (payload.saved_models || {}) as Record<string, string>,
        effective_models: nextEffective,
      }));
      setModelInputs(nextEffective);
      setModelsMessage({ tone: "success", text: "Modelos reseteados al estado base." });
      refreshServerState();
    } catch (error) {
      setModelsMessage({
        tone: "danger",
        text: error instanceof Error ? error.message : "No se pudieron resetear los modelos.",
      });
    } finally {
      setSavingModels(false);
    }
  }

  async function saveSandbox(): Promise<void> {
    setSavingSandbox(true);
    setSandboxMessage(null);
    try {
      const response = await fetch("/api/config/sandbox", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          config: {
            provider,
            settings: {
              ...sanitizeObject(providerSettings),
              execution_backend: sanitizeObject(executionBackend),
            },
          },
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as SandboxConfigResponse;
      setSandboxState(payload);
      setProvider(payload.provider);
      setProviderSettings(
        sanitizeObject(
          Object.fromEntries(
            Object.entries(
              (((payload.config.settings as Record<string, unknown> | undefined) || {}) as Record<
                string,
                unknown
              >) || {},
            ).filter(([key]) => key !== "execution_backend"),
          ),
        ),
      );
      setExecutionBackend(
        sanitizeObject((payload.execution_backend as Record<string, unknown>) || {}),
      );
      setSandboxMessage({ tone: "success", text: "Sandbox guardado correctamente." });
      refreshServerState();
    } catch (error) {
      setSandboxMessage({
        tone: "danger",
        text: error instanceof Error ? error.message : "No se pudo guardar el sandbox.",
      });
    } finally {
      setSavingSandbox(false);
    }
  }

  async function updateApiKey(envVar: string): Promise<void> {
    const value = String(apiKeyInputs[envVar] || "").trim();
    if (!value) {
      setApiKeysMessage({
        tone: "warning",
        text: `Introduce un valor para actualizar ${envVar}.`,
      });
      return;
    }
    setSavingApiKey(envVar);
    setApiKeysMessage(null);
    try {
      const response = await fetch("/api/config/api-keys", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keys: { [envVar]: value } }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as ApiKeysResponse;
      setApiKeysState(payload);
      setApiKeyInputs((current) => ({ ...current, [envVar]: "" }));
      setApiKeysMessage({ tone: "success", text: `${envVar} actualizada.` });
      refreshServerState();
    } catch (error) {
      setApiKeysMessage({
        tone: "danger",
        text: error instanceof Error ? error.message : "No se pudo actualizar la clave.",
      });
    } finally {
      setSavingApiKey(null);
    }
  }

  async function testApiKey(envVar: string): Promise<void> {
    setSavingApiKey(envVar);
    setApiKeysMessage(null);
    try {
      const response = await fetch("/api/config/api-keys/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          env_var: envVar,
          value: String(apiKeyInputs[envVar] || "").trim() || undefined,
        }),
      });
      const payload = (await response.json()) as { ok: boolean; message: string };
      setApiKeysMessage({
        tone: payload.ok ? "success" : "warning",
        text: `${envVar}: ${payload.message}`,
      });
    } catch (error) {
      setApiKeysMessage({
        tone: "danger",
        text: error instanceof Error ? error.message : "No se pudo testear la clave.",
      });
    } finally {
      setSavingApiKey(null);
    }
  }

  async function runConnectorAction(action: "test" | "objects" | "fetch"): Promise<void> {
    if (!selectedConnector || !selectedAuthMode) {
      return;
    }
    const credentials = sanitizeObject(connectorCredentials);
    setConnectorBusyAction(action);
    setConnectorMessage(null);
    try {
      const body =
        action === "fetch"
          ? {
              credentials,
              object_name: connectorObjectName,
              preview_rows: 12,
              max_records: 2000,
              save_to_data: true,
            }
          : { credentials };

      const response = await fetch(
        `/api/integrations/connectors/${selectedConnector.id}/${action}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || JSON.stringify(payload));
      }

      if (action === "test") {
        setConnectorMessage({
          tone: payload.ok ? "success" : "warning",
          text: payload.message || "Conexion verificada.",
        });
      } else if (action === "objects") {
        const items = Array.isArray(payload.items)
          ? payload.items
              .map((item: unknown) => {
                if (item && typeof item === "object") {
                  const record = item as Record<string, unknown>;
                  return String(record.name || record.id || record.label || "");
                }
                return String(item || "");
              })
              .filter((item: string) => item.trim())
          : [];
        setConnectorObjects(items);
        if (items.length && !connectorObjectName) {
          setConnectorObjectName(items[0]);
        }
        setConnectorMessage({
          tone: "success",
          text: `Se cargaron ${items.length} objetos del conector.`,
        });
      } else {
        setConnectorPreview(Array.isArray(payload.preview) ? payload.preview : []);
        setConnectorCsvPath(String(payload.csv_path || ""));
        setConnectorMessage({
          tone: "success",
          text: `Preview generada para ${payload.object_name || connectorObjectName}.`,
        });
      }
    } catch (error) {
      setConnectorMessage({
        tone: "danger",
        text: error instanceof Error ? error.message : "Error operando el conector.",
      });
    } finally {
      setConnectorBusyAction(null);
    }
  }

  return (
    <div className="stack-lg">
      <div className="summary-strip">
        <div className="summary-chip">
          <span>Slots principales</span>
          <strong>{groupedAgents.primary.length}</strong>
        </div>
        <div className="summary-chip">
          <span>Sandbox</span>
          <strong>{provider}</strong>
        </div>
        <div className="summary-chip">
          <span>Credenciales listas</span>
          <strong>
            {configuredRequiredKeys}/{totalRequiredKeys}
          </strong>
        </div>
        <div className="summary-chip">
          <span>Conectores CRM</span>
          <strong>{initialConnectors.count}</strong>
        </div>
      </div>

      <section className="workspace-panel">
        <div className="tab-strip" role="tablist" aria-label="Settings">
          {settingsTabs.map((tab) => (
            <button
              key={tab.key}
              className={`tab-button${activeTab === tab.key ? " active" : ""}`}
              onClick={() => setActiveTab(tab.key)}
              type="button"
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === "models" ? (
          <div className="stack-lg">
            <div className="section-head">
              <div>
                <p className="eyebrow">Modelos</p>
                <h2>Configuracion de modelos por agente</h2>
              </div>
              <div className="panel-actions">
                <button className="secondary-button" onClick={resetModels} disabled={savingModels}>
                  Reset
                </button>
                <button className="primary-button" onClick={saveModels} disabled={savingModels}>
                  {savingModels ? "Guardando..." : "Guardar modelos"}
                </button>
              </div>
            </div>

            {modelsMessage ? (
              <p className={`inline-message ${modelsMessage.tone}`}>{modelsMessage.text}</p>
            ) : null}

            <div className="settings-form-grid">
              {groupedAgents.primary.map((agent) => (
                <div className="field" key={agent.key}>
                  <span>{agent.label}</span>
                  <select
                    value={resolveModelSelectValue(agent.key)}
                    onChange={(event) => updateModelPreset(agent.key, event.target.value)}
                  >
                    <option value="">Sin configurar</option>
                    {modelsState.presets.map((preset) => (
                      <option key={preset.id} value={preset.id}>
                        {preset.label}
                      </option>
                    ))}
                    {(() => {
                      const v = String(modelInputs[agent.key] || "").trim();
                      if (v && !presetModels.has(v) && !customModeAgents.has(agent.key)) {
                        return <option value={v}>{v} (personalizado)</option>;
                      }
                      return null;
                    })()}
                    <option value={customModelOption}>Modelo personalizado…</option>
                  </select>
                  {customModeAgents.has(agent.key) ? (
                    <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                      <input
                        type="text"
                        autoFocus
                        value={modelInputs[agent.key] || ""}
                        placeholder="z-ai/glm-5.1"
                        onChange={(event) =>
                          setModelInputs((current) => ({ ...current, [agent.key]: event.target.value }))
                        }
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            commitCustomModel(agent.key, (event.target as HTMLInputElement).value);
                          } else if (event.key === "Escape") {
                            event.preventDefault();
                            commitCustomModel(agent.key, (event.target as HTMLInputElement).value);
                          }
                        }}
                      />
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => commitCustomModel(agent.key)}
                        disabled={!String(modelInputs[agent.key] || "").trim()}
                        aria-label="Confirmar modelo personalizado"
                      >
                        OK
                      </button>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>

            <details className="settings-details">
              <summary>Modelos avanzados</summary>
              <div className="settings-form-grid">
                {groupedAgents.advanced.map((agent) => (
                  <div className="field" key={agent.key}>
                    <span>{agent.label}</span>
                    <select
                      value={resolveModelSelectValue(agent.key)}
                      onChange={(event) => updateModelPreset(agent.key, event.target.value)}
                    >
                      <option value="">Sin configurar</option>
                      {modelsState.presets.map((preset) => (
                        <option key={preset.id} value={preset.id}>
                          {preset.label}
                        </option>
                      ))}
                      <option value={customModelOption}>Modelo personalizado</option>
                    </select>
                    {resolveModelSelectValue(agent.key) === customModelOption ? (
                      <input
                        type="text"
                        value={modelInputs[agent.key] || ""}
                        placeholder="anthropic/claude-sonnet-4.6"
                        onChange={(event) =>
                          setModelInputs((current) => ({ ...current, [agent.key]: event.target.value }))
                        }
                      />
                    ) : null}
                  </div>
                ))}
              </div>
            </details>
          </div>
        ) : null}

        {activeTab === "sandbox" ? (
          <div className="stack-lg">
            <div className="section-head">
              <div>
                <p className="eyebrow">Sandbox</p>
                <h2>Proveedor y backend de ejecucion</h2>
              </div>
              <button className="primary-button" onClick={saveSandbox} disabled={savingSandbox}>
                {savingSandbox ? "Guardando..." : "Guardar sandbox"}
              </button>
            </div>

            {sandboxMessage ? (
              <p className={`inline-message ${sandboxMessage.tone}`}>{sandboxMessage.text}</p>
            ) : null}

            <div className="simple-info-list">
              <div className="metric-row">
                <span>Estado actual</span>
                <StatusPill value={String(sandboxState.provider_status?.severity || "neutral")} />
              </div>
              <div className="metric-row">
                <span>Backend</span>
                <strong>{String(sandboxState.execution_backend_status?.detail || "N/A")}</strong>
              </div>
            </div>

            <div className="settings-form-grid">
              <label className="field field-full">
                <span>Proveedor</span>
                <select
                  value={provider}
                  onChange={(event) => {
                    setProvider(event.target.value);
                    setProviderSettings({});
                  }}
                >
                  {sandboxState.providers.map((item) => (
                    <option key={item.name} value={item.name}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>

              {(selectedProviderSpec?.config_fields || []).map((field) => (
                <label className="field" key={field.key}>
                  <span>{field.label}</span>
                  <input
                    type={field.secret ? "password" : "text"}
                    placeholder={field.placeholder || field.description || field.key}
                    value={providerSettings[field.key] || ""}
                    onChange={(event) =>
                      setProviderSettings((current) => ({
                        ...current,
                        [field.key]: event.target.value,
                      }))
                    }
                  />
                </label>
              ))}
            </div>

            <details className="settings-details" open>
              <summary>Execution backend</summary>
              <div className="settings-form-grid">
                <label className="field">
                  <span>Modo</span>
                  <select
                    value={executionBackend.mode || "cloudrun"}
                    onChange={(event) =>
                      setExecutionBackend((current) => ({ ...current, mode: event.target.value }))
                    }
                  >
                    <option value="cloudrun">cloudrun</option>
                    <option value="local">local</option>
                  </select>
                </label>
                {["job", "region", "bucket", "project", "script_timeout_seconds"].map((key) => (
                  <label className="field" key={key}>
                    <span>{key}</span>
                    <input
                      value={executionBackend[key] || ""}
                      onChange={(event) =>
                        setExecutionBackend((current) => ({
                          ...current,
                          [key]: event.target.value,
                        }))
                      }
                    />
                  </label>
                ))}
              </div>
            </details>
          </div>
        ) : null}

        {activeTab === "keys" ? (
          <div className="stack-lg">
            <div className="section-head">
              <div>
                <p className="eyebrow">Credenciales</p>
                <h2>API keys operativas</h2>
              </div>
            </div>

            {apiKeysMessage ? (
              <p className={`inline-message ${apiKeysMessage.tone}`}>{apiKeysMessage.text}</p>
            ) : null}

            <div className="stack-sm">
              {apiKeysState.items.map((item) => (
                <div key={item.env_var} className="credential-card">
                  <div className="credential-top">
                    <div>
                      <strong>{item.label}</strong>
                      <p>{item.description}</p>
                    </div>
                    <StatusPill value={item.configured ? "ok" : item.required ? "warning" : "neutral"} />
                  </div>
                  <div className="settings-form-grid">
                    <label className="field field-full">
                      <span>{item.env_var}</span>
                      <input
                        type="password"
                        placeholder={item.masked_value || item.placeholder}
                        value={apiKeyInputs[item.env_var] || ""}
                        onChange={(event) =>
                          setApiKeyInputs((current) => ({
                            ...current,
                            [item.env_var]: event.target.value,
                          }))
                        }
                      />
                    </label>
                  </div>
                  <div className="panel-actions">
                    <button
                      className="secondary-button"
                      onClick={() => testApiKey(item.env_var)}
                      disabled={savingApiKey === item.env_var}
                    >
                      Test
                    </button>
                    <button
                      className="primary-button"
                      onClick={() => updateApiKey(item.env_var)}
                      disabled={savingApiKey === item.env_var}
                    >
                      {savingApiKey === item.env_var ? "Guardando..." : "Actualizar"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {activeTab === "connectors" ? (
          <div className="stack-lg">
            <div className="section-head">
              <div>
                <p className="eyebrow">Conectores CRM</p>
                <h2>Test y preview de extraccion</h2>
              </div>
            </div>

            {connectorMessage ? (
              <p className={`inline-message ${connectorMessage.tone}`}>{connectorMessage.text}</p>
            ) : null}

            <div className="settings-form-grid">
              <label className="field">
                <span>Conector</span>
                <select
                  value={selectedConnectorId}
                  onChange={(event) => {
                    const connectorId = event.target.value;
                    setSelectedConnectorId(connectorId);
                    const connector = initialConnectors.items.find((item) => item.id === connectorId);
                    setSelectedAuthModeId(connector?.auth_modes[0]?.id || "");
                    setConnectorCredentials({});
                    setConnectorObjects([]);
                    setConnectorPreview([]);
                    setConnectorObjectName("");
                    setConnectorCsvPath("");
                  }}
                >
                  {initialConnectors.items.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Modo auth</span>
                <select
                  value={selectedAuthModeId}
                  onChange={(event) => setSelectedAuthModeId(event.target.value)}
                >
                  {(selectedConnector?.auth_modes || []).map((mode) => (
                    <option key={mode.id} value={mode.id}>
                      {mode.label}
                    </option>
                  ))}
                </select>
              </label>

              {(selectedAuthMode?.fields || []).map((field) => (
                <label className="field" key={field.key}>
                  <span>{field.label}</span>
                  <input
                    type={field.secret ? "password" : "text"}
                    value={connectorCredentials[field.key] || ""}
                    onChange={(event) =>
                      setConnectorCredentials((current) => ({
                        ...current,
                        [field.key]: event.target.value,
                      }))
                    }
                  />
                </label>
              ))}

              <label className="field field-full">
                <span>Objeto CRM</span>
                <input
                  list="connector-objects"
                  value={connectorObjectName}
                  onChange={(event) => setConnectorObjectName(event.target.value)}
                  placeholder="Selecciona o escribe el objeto a extraer"
                />
                <datalist id="connector-objects">
                  {connectorObjects.map((item) => (
                    <option key={item} value={item} />
                  ))}
                </datalist>
              </label>
            </div>

            <div className="panel-actions">
              <button
                className="secondary-button"
                onClick={() => runConnectorAction("test")}
                disabled={connectorBusyAction !== null}
              >
                {connectorBusyAction === "test" ? "Probando..." : "Probar conexion"}
              </button>
              <button
                className="secondary-button"
                onClick={() => runConnectorAction("objects")}
                disabled={connectorBusyAction !== null}
              >
                {connectorBusyAction === "objects" ? "Cargando..." : "Listar objetos"}
              </button>
              <button
                className="primary-button"
                onClick={() => runConnectorAction("fetch")}
                disabled={connectorBusyAction !== null || !connectorObjectName}
              >
                {connectorBusyAction === "fetch" ? "Extrayendo..." : "Preview y guardar CSV"}
              </button>
            </div>

            {connectorCsvPath ? (
              <p className="helper-copy">
                CSV generado en: <strong>{connectorCsvPath}</strong>
              </p>
            ) : null}

            {connectorPreview.length ? (
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      {Object.keys(connectorPreview[0]).map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {connectorPreview.map((row, index) => (
                      <tr key={index}>
                        {Object.keys(connectorPreview[0]).map((column) => (
                          <td key={column}>{String(row[column] ?? "")}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>
        ) : null}
      </section>

      {isRefreshing ? <p className="helper-copy">Sincronizando estado del frontend...</p> : null}
    </div>
  );
}
