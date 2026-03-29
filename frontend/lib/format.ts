export function formatNumber(value: unknown, digits = 2): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    if (typeof value === "string" && value.trim()) {
      const parsed = Number(value);
      if (!Number.isNaN(parsed)) {
        return formatNumber(parsed, digits);
      }
    }
    return "N/A";
  }
  return new Intl.NumberFormat("es-ES", {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatDate(value: unknown): string {
  if (typeof value === "number" && value > 0) {
    return new Date(value * 1000).toLocaleString("es-ES");
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toLocaleString("es-ES");
    }
    return value;
  }
  return "N/A";
}

export function formatShortText(value: unknown, maxLength = 140): string {
  const text = String(value || "").trim();
  if (!text) {
    return "N/A";
  }
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 1)}…`;
}
