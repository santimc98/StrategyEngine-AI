import argparse
import json
import os
import sys
from typing import Any, Dict, List

from src.utils.contract_views import (
    build_contract_views_projection,
    persist_views,
)


def _load_json(path: str) -> Any:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph utilities")
    parser.add_argument("--dry_views", action="store_true", help="Generate contract views without invoking LLMs.")
    parser.add_argument("--contract_full", type=str, default="", help="Path to execution contract JSON.")
    parser.add_argument("--contract_min", type=str, default="", help="Deprecated: ignored.")
    parser.add_argument("--artifact_index", type=str, default="", help="Path to artifact index JSON.")
    parser.add_argument("--output_dir", type=str, default="data", help="Base output directory.")
    parser.add_argument("--run_bundle_dir", type=str, default="", help="Optional run bundle dir for persistence.")
    args = parser.parse_args()

    if not args.dry_views:
        parser.print_help()
        return 1

    contract_full = _load_json(args.contract_full) if args.contract_full else _load_json("data/execution_contract.json") or {}

    if not contract_full:
        print("dry_views error: missing execution_contract. Provide --contract_full or data/execution_contract.json.")
        return 2

    views = build_contract_views_projection(contract_full)
    persisted = persist_views(
        views,
        base_dir=args.output_dir,
        run_bundle_dir=args.run_bundle_dir or None,
    )
    print("dry_views completed:")
    for key, path in persisted.items():
        print(f"- {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
