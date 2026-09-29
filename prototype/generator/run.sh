#!/bin/bash
# スローアップを1枚生成して ~/graffiti/output/latest.svg を上書きする
cd "$(dirname "$0")"

# Max から起動されたシェルは .zshrc を読まないので、PATH に python3 がない。
# GRAFFITI_PY があればそれを使う。なければ下の既定値（Tim の miniforge）。
# どちらも使えなければ従来どおり PATH から探す（藁科さんの環境はここに落ちる）。
PY="${GRAFFITI_PY:-/opt/homebrew/Caskroom/miniforge/base/bin/python3}"
[ -x "$PY" ] || PY="$(command -v python3 || echo /usr/bin/python3)"

exec "$PY" make_one.py
