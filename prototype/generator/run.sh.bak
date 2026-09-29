#!/bin/bash
# スローアップを1枚生成して ~/graffiti/output/latest.svg を上書きする
cd "$(dirname "$0")"
PY="$(command -v python3 || echo /usr/bin/python3)"
exec "$PY" make_one.py
