from __future__ import annotations

import argparse

from pdna.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PDNA paths and paper constants")
    parser.add_argument("--config", default="configs/pdna.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print("Configuration: OK")
    for name in ("llm_data", "negmas_data"):
        path = config.resolve_path(name)
        status = "exists" if path.exists() else "empty/not created (allowed)"
        print(f"{name}: {path} [{status}]")


if __name__ == "__main__":
    main()

