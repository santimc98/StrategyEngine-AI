"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

const navItems = [
  { href: "/", label: "Nueva Run" },
  { href: "/runs", label: "Runs" },
  { href: "/settings", label: "Settings" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <p className="brand-kicker">StrategyEngine AI</p>
          <h1>Plataforma multiagente con gobernanza</h1>
          <p className="brand-copy">
            Frontend productivo sobre la API de runs, reportes, configuración e integraciones.
          </p>
        </div>

        <nav className="nav-stack" aria-label="Navegación principal">
          {navItems.map((item) => {
            const active =
              item.href === "/" ? pathname === item.href : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`nav-link${active ? " active" : ""}`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="sidebar-panel">
          <p className="sidebar-title">Objetivo de esta fase</p>
          <p className="sidebar-copy">
            Separar definitivamente la experiencia de producto del backend y dejar Streamlit como
            consola transitoria.
          </p>
        </div>
      </aside>

      <main className="main-content">
        <div className="content-inner">{children}</div>
      </main>
    </div>
  );
}
