"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { StatusPill } from "@/components/status-pill";
import { formatShortText } from "@/lib/format";
import type { RunListItem } from "@/types/api";

type RunsWorkspaceProps = {
  items: RunListItem[];
};

function uniqueValues(items: RunListItem[], field: "status" | "verdict"): string[] {
  return Array.from(
    new Set(
      items
        .map((item) => String(item[field] || "").trim())
        .filter(Boolean),
    ),
  ).sort((a, b) => a.localeCompare(b));
}

export function RunsWorkspace({ items }: RunsWorkspaceProps) {
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [verdictFilter, setVerdictFilter] = useState("all");

  const statusOptions = useMemo(() => uniqueValues(items, "status"), [items]);
  const verdictOptions = useMemo(() => uniqueValues(items, "verdict"), [items]);

  const filteredItems = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return items.filter((item) => {
      if (statusFilter !== "all" && item.status !== statusFilter) {
        return false;
      }
      if (verdictFilter !== "all" && item.verdict !== verdictFilter) {
        return false;
      }
      if (!normalizedQuery) {
        return true;
      }
      const haystack = [
        item.run_id,
        item.business_objective,
        item.strategy,
        item.metric_name,
        item.metric_value,
        item.verdict,
        item.status,
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(normalizedQuery);
    });
  }, [items, query, statusFilter, verdictFilter]);

  const summary = useMemo(() => {
    return {
      total: items.length,
      shown: filteredItems.length,
      completed: items.filter((item) => item.status === "complete").length,
      approved: items.filter((item) => String(item.verdict || "").includes("APPROVE")).length,
    };
  }, [filteredItems.length, items]);

  return (
    <section className="workspace-panel">
      <div className="workspace-panel-head">
        <div>
          <p className="eyebrow">Inventario</p>
          <h2>Runs del entorno</h2>
        </div>
      </div>

      <div className="summary-strip">
        <div className="summary-chip">
          <span>Total</span>
          <strong>{summary.total}</strong>
        </div>
        <div className="summary-chip">
          <span>Visibles</span>
          <strong>{summary.shown}</strong>
        </div>
        <div className="summary-chip">
          <span>Completadas</span>
          <strong>{summary.completed}</strong>
        </div>
        <div className="summary-chip">
          <span>Aprobadas</span>
          <strong>{summary.approved}</strong>
        </div>
      </div>

      <div className="workspace-divider" />

      <div className="filter-stack">
        <label className="field field-full">
          <span>Busqueda</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Buscar por run id, objetivo, estrategia o metrica"
          />
        </label>

        <div className="filter-inline-grid">
          <label className="field">
            <span>Status</span>
            <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
              <option value="all">Todos</option>
              {statusOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Veredicto</span>
            <select value={verdictFilter} onChange={(event) => setVerdictFilter(event.target.value)}>
              <option value="all">Todos</option>
              {verdictOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="workspace-divider" />

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Run</th>
              <th>Inicio</th>
              <th>Estado</th>
              <th>Veredicto</th>
              <th>Metrica</th>
              <th>Duracion</th>
              <th>Objetivo</th>
            </tr>
          </thead>
          <tbody>
            {filteredItems.map((run) => (
              <tr key={run.run_id}>
                <td>
                  <Link className="table-link" href={`/runs/${run.run_id}`}>
                    {run.run_id}
                  </Link>
                </td>
                <td>{run.started_str || "N/A"}</td>
                <td>
                  <StatusPill value={run.status} />
                </td>
                <td>
                  <StatusPill value={run.verdict} />
                </td>
                <td>
                  {run.metric_name ? `${run.metric_name}: ${run.metric_value || "N/A"}` : "N/A"}
                </td>
                <td>{run.elapsed || "N/A"}</td>
                <td>{formatShortText(run.business_objective, 150)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {!filteredItems.length ? (
        <p className="muted-copy">No hay runs que coincidan con los filtros actuales.</p>
      ) : null}
    </section>
  );
}
