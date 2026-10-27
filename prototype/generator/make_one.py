#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
落書きを1枚生成して SVG を上書きする。

    python3 make_one.py            # 毎回ちがう絵
    python3 make_one.py 12         # 同じ番号なら毎回同じ絵（確認用）

出力: ~/graffiti/output/latest.svg （820 × 860 mm）
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graffiti_gen as gg

HERE  = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.pkl")
OUT   = os.path.expanduser("~/graffiti/output/latest.svg")


def main():
    gg.load_model(MODEL)

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(10 ** 9)
    g    = gg.generate(seed=seed)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    gg.to_svg(g, OUT)

    print("%s   線 %d 本   seed=%d   %d x %d mm"
          % (OUT, len(g), seed, gg.MM_W, gg.MM_H))


if __name__ == "__main__":
    main()
