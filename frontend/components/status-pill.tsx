type StatusPillProps = {
  value?: string | null;
};

function resolveStatusTone(rawValue: string): string {
  const value = rawValue.toLowerCase();
  if (["ok", "approved", "complete", "go", "go_with_limitations", "success"].includes(value)) {
    return "success";
  }
  if (["warning", "approved_with_warnings", "needs_improvement", "running", "present_optional"].includes(value)) {
    return "warning";
  }
  if (["error", "rejected", "failed", "aborted", "stop"].includes(value)) {
    return "danger";
  }
  if (["info", "pending", "queued"].includes(value)) {
    return "info";
  }
  return "neutral";
}

export function StatusPill({ value }: StatusPillProps) {
  const text = String(value || "N/A").trim() || "N/A";
  const tone = resolveStatusTone(text);
  return <span className={`pill pill-${tone}`}>{text.replaceAll("_", " ")}</span>;
}
