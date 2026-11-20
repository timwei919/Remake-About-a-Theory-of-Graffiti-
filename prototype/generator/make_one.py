#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
落書きを1枚生成して SVG を上書きする。

    python3 make_one.py            # 毎回ちがう絵
    python3 make_one.py 12         # 同じ番号なら毎回同じ絵（確認用）

出力: ~/graffiti/output/latest.svg （縦長 1030 × 1330 mm）

絵は横 1330 × 縦 1030 mm で生成し、反時計回りに90度回して書き出す（11/20）。
回さずに横長で出したいときは ROTATE = False にする。
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graffiti_gen as gg

HERE   = os.path.dirname(os.path.abspath(__file__))
MODEL  = os.path.join(HERE, "model.pkl")
OUT    = os.path.expanduser("~/graffiti/output/latest.svg")
ROTATE = True          # True なら反時計回りに90度回して、横 1030 × 縦 1330 mm で書き出す


def to_svg_rotated(g, path, stroke=1.2):
    """gg.to_svg と同じ線を、反時計回りに90度回して書き出す。

    横長の紙（横 MM_W × 縦 MM_H）の上の点 (X, Y) は、
    回したあとの縦長の紙（横 MM_H × 縦 MM_W）では (Y, MM_W − X) に来る。
    （SVG の座標は左上が原点で、下向きが正）"""
    mw, mh = gg.MM_W, gg.MM_H
    kx, ky = mw / gg.W, mh / gg.H
    body = []
    for s in g:
        pts = []
        for x, y in s:
            X, Y = x * kx, (gg.H - y) * ky          # 回す前の紙の上の位置（gg.to_svg と同じ）
            pts.append(f"{Y:.2f},{mw - X:.2f}")      # 反時計回りに90度
        body.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                    f'stroke="black" stroke-width="{stroke}" '
                    f'stroke-linecap="round" stroke-linejoin="round"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{mh}mm" height="{mw}mm" '
           f'viewBox="0 0 {mh} {mw}">\n' + "\n".join(body) + "\n</svg>\n")
    with open(path, "w") as f:
        f.write(svg)
    return path


def main():
    gg.load_model(MODEL)

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(10 ** 9)
    g    = gg.generate(seed=seed)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if ROTATE:
        to_svg_rotated(g, OUT)
        w, h = gg.MM_H, gg.MM_W
    else:
        gg.to_svg(g, OUT)
        w, h = gg.MM_W, gg.MM_H

    print("%s   線 %d 本   seed=%d   %d x %d mm"
          % (OUT, len(g), seed, w, h))


if __name__ == "__main__":
    main()
