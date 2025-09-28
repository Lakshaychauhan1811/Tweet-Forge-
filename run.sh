#!/bin/bash
set -e
echo "========================================"
echo "Marketing Tweet Generator (LangChain)"
echo "========================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: Python 3 is not installed"
  exit 1
fi

if [ ! -f .env ]; then
  echo "Creating .env from env_example.txt"
  cp env_example.txt .env
fi

if [ ! -d env ]; then
  python3.11 -m venv env || python3 -m venv env
fi
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 start.py
