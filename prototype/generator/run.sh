#!/bin/bash
# 落書きを1枚生成して ~/graffiti/output/latest.svg を上書きする。
# Max/MSP からはこれを叩く。
cd "$(dirname "$0")"
python3 make_one.py "$@"
