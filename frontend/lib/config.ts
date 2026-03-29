const rawBaseUrl =
  process.env.STRATEGYENGINE_API_BASE_URL ||
  process.env.NEXT_PUBLIC_STRATEGYENGINE_API_BASE_URL ||
  "http://127.0.0.1:8000";

export const API_BASE_URL = rawBaseUrl.replace(/\/+$/, "");

export function toApiUrl(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalized}`;
}
