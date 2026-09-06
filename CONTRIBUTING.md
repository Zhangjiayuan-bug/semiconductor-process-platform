# Contributing

## Development setup

Use Python 3.11 or 3.12. Install the reproducible development environment and prepare both datasets:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python scripts_generate_demo.py
python scripts_download_secom.py
```

## Quality checks

Run the same checks required by GitHub Actions before opening a pull request:

```powershell
ruff check .
ruff format --check .
pytest --cov=src --cov-branch
```

Changes to statistical or model behavior must include a deterministic regression test. Document data provenance, leakage controls, validation assumptions, metric changes, and limitations; do not present screening outputs as causal or production-ready conclusions.
