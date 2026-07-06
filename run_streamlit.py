"""Start the Streamlit app with the local SSL compatibility patch enabled."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import sitecustomize  # noqa: F401  # Ensure SSL patch is installed before Streamlit imports Tornado.


def main() -> None:
    """Run ``streamlit run app.py`` from Python."""

    project_dir = Path(__file__).resolve().parent
    app_path = project_dir / "app.py"
    extra_args = sys.argv[1:]
    sys.argv = ["streamlit", "run", str(app_path), *extra_args]
    runpy.run_module("streamlit", run_name="__main__")


if __name__ == "__main__":
    main()
