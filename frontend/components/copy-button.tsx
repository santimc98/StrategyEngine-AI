"use client";

import { useCallback, useState } from "react";

type CopyButtonProps = {
  /** The text to copy to clipboard */
  text: string;
  /** Visual variant: "dark" for code-frame backgrounds, "light" for white/panel backgrounds */
  variant?: "dark" | "light";
  /** Optional label shown next to the icon */
  label?: string;
};

/**
 * Compact copy-to-clipboard button with animated feedback.
 * Designed to sit in the top-right corner of code/text blocks.
 */
export function CopyButton({ text, variant = "dark", label }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers / non-HTTPS
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [text]);

  const isDark = variant === "dark";

  return (
    <button
      type="button"
      onClick={handleCopy}
      title={copied ? "Copiado!" : "Copiar al portapapeles"}
      className="copy-btn"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "5px",
        padding: "5px 10px",
        borderRadius: "8px",
        border: `1px solid ${isDark ? "rgba(139, 157, 204, 0.25)" : "var(--border)"}`,
        background: copied
          ? isDark ? "rgba(74, 222, 128, 0.15)" : "rgba(74, 222, 128, 0.1)"
          : isDark ? "rgba(255, 255, 255, 0.06)" : "rgba(0, 0, 0, 0.03)",
        color: copied
          ? "#4ade80"
          : isDark ? "#8b9dcc" : "var(--muted)",
        fontSize: "0.75rem",
        fontWeight: 600,
        cursor: "pointer",
        transition: "all 0.2s ease",
        lineHeight: 1,
        whiteSpace: "nowrap",
      }}
    >
      {copied ? (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
      {label && <span>{copied ? "Copiado!" : label}</span>}
      {!label && copied && <span>Copiado!</span>}
    </button>
  );
}

type CopyableFrameProps = {
  /** The raw text to copy (may differ from displayed content) */
  copyText: string;
  /** Visual variant */
  variant?: "dark" | "light";
  /** Optional label for the button */
  copyLabel?: string;
  /** Children to render inside the frame */
  children: React.ReactNode;
  /** Extra class for the wrapper */
  className?: string;
  /** Extra inline styles */
  style?: React.CSSProperties;
};

/**
 * A wrapper that positions a CopyButton in the top-right corner of its children.
 * Use this to wrap `.code-frame` divs or text panels.
 */
export function CopyableFrame({
  copyText,
  variant = "dark",
  copyLabel,
  children,
  className,
  style,
}: CopyableFrameProps) {
  return (
    <div className={className} style={{ position: "relative", ...style }}>
      <div style={{ position: "absolute", top: "10px", right: "10px", zIndex: 2 }}>
        <CopyButton text={copyText} variant={variant} label={copyLabel} />
      </div>
      {children}
    </div>
  );
}
