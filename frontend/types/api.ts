export type JsonRecord = Record<string, unknown>;

export interface RunListItem {
  run_id: string;
  status: string;
  started_at: number;
  started_str: string;
  elapsed: string;
  strategy: string;
  metric_name: string;
  metric_value: string;
  iterations: number;
  verdict: string;
  business_objective: string;
}

export interface RunListResponse {
  items: RunListItem[];
  count: number;
}

export interface ActiveRunResponse {
  active_run_id: string | null;
  status: JsonRecord | null;
}

export interface RunDetailResponse {
  run_id: string;
  has_final_state: boolean;
  input: JsonRecord;
  manifest: JsonRecord;
  status: JsonRecord;
  result: JsonRecord;
}

export interface RunLogsResponse {
  run_id: string;
  after_line: number;
  next_after_line: number;
  entries: Array<{
    ts: string;
    agent: string;
    msg: string;
    level: string;
  }>;
}

export interface ReportPlotItem {
  filename: string;
  title: string;
  facts: string[];
  referenced_in_report: boolean;
  order: number;
  image_url: string;
}

export interface RunReportResponse {
  run_id: string;
  status: string | null;
  run_outcome: string | null;
  markdown: string;
  blocks: JsonRecord[];
  pdf_available: boolean;
  pdf_url: string | null;
  plots: ReportPlotItem[];
  artifact_manifest_summary: JsonRecord;
  run_summary: JsonRecord;
  visual_tables: JsonRecord;
}

export interface ModelsConfigResponse {
  runtime_available: boolean;
  bootstrap_error: string | null;
  presets: Array<{ id: string; label: string }>;
  custom_option?: string;
  saved_models?: Record<string, string>;
  agents: Array<{ key: string; label: string; section: string }>;
  base_models: Record<string, string>;
  persisted_models: Record<string, string>;
  effective_models: Record<string, string>;
  primary_models: Record<string, string>;
  advanced_models: Record<string, string>;
}

export interface SandboxProviderField {
  key: string;
  label: string;
  description: string;
  placeholder: string;
  secret: boolean;
  required: boolean;
}

export interface SandboxProviderSpecItem {
  name: string;
  label: string;
  description: string;
  implemented: boolean;
  available: boolean;
  config_fields: SandboxProviderField[];
}

export interface SandboxConfigResponse {
  config: JsonRecord;
  provider: string;
  provider_spec: JsonRecord;
  provider_status: JsonRecord;
  execution_backend: JsonRecord;
  execution_backend_status: JsonRecord;
  provider_connectivity: JsonRecord;
  providers: SandboxProviderSpecItem[];
}

export interface ApiKeysResponse {
  items: Array<{
    env_var: string;
    label: string;
    description: string;
    placeholder: string;
    required: boolean;
    configured: boolean;
    masked_value: string;
  }>;
  summary: JsonRecord;
}

export interface ConnectorSpecsResponse {
  items: Array<{
    id: string;
    label: string;
    auth_modes: Array<{
      id: string;
      label: string;
      fields: Array<{
        key: string;
        label: string;
        secret: boolean;
        required: boolean;
      }>;
    }>;
  }>;
  count: number;
}
