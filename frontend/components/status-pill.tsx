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
  const isRunning = text.toLowerCase() === "running";
  
  return (
    <span className={`pill pill-${tone}`} style={isRunning ? { animation: "pulseOpacity 2s infinite ease-in-out" } : {}}>
      {isRunning && (
        <span 
          style={{ 
            display: "inline-block", 
            width: "6px", 
            height: "6px", 
            background: "currentcolor", 
            borderRadius: "50%", 
            marginRight: "6px", 
            verticalAlign: "middle",
            animation: "pulseOpacity 1s infinite alternate" 
          }} 
        />
      )}
      {text.replaceAll("_", " ")}
    </span>
  );
}
