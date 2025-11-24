#!/usr/bin/env bash
set -euo pipefail

# Create and activate venv, then install requirements
python3 -m venv .venv
echo "Created virtualenv at .venv"
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Installed requirements. Activate with: source .venv/bin/activate"
