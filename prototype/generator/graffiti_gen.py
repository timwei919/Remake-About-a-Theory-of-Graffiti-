# -*- coding: utf-8 -*-
"""
落書き生成（2026年10月28日 版）

写真から取り出した一筆の線を分類し、その線を混ぜて新しい線を作り、
文字・単語・段という構造で並べて、1枚の落書きを生成する。

学習（1回だけ）:
    import graffiti_gen as gg
    gg.fit("pieces.pkl")
    gg.save_model("model.pkl")

生成（毎回）:
    import graffiti_gen as gg
    gg.load_model("model.pkl")
    g = gg.generate(seed=0)
    gg.to_svg(g, "latest.svg")

依存: numpy のみ
"""

import pickle
import numpy as np


# ======================================================================
# 設定
# ======================================================================

K = 7                      # 分類のグループ数

# 描画面の実寸（mm）。黒いフレームの内側。
# 横長で使う前提。縦長で使うならこの2行を入れ替える。
MM_W = 860                 # 横
MM_H = 820                 # 縦

ROW_H = 1.15               # 1段の高さ（落書き全体の高さ=1 のスケール）
NROW  = 2                  # 段数
H = ROW_H * NROW
W = H * MM_W / MM_H

P = dict(
    nrow   = 2,            # 段数（1 or 2）。1なら1段でキャンバス全体を使う
    nword  = (2, 4),       # 全体の単語の数（2〜3）
    nlet   = (2, 4),       # 1単語の文字数（2〜3）
    nstk   = (1, 3),       # 1文字を作る線の数（1〜2）
    lsize  = 0.60,         # 1文字の高さ（段の高さに対する割合）
    gap    = -0.15,        # 文字どうしの重なり（文字の高さに対する割合）
    spread = 0.35,         # 1文字の中で線をずらす幅
    nseg   = (4, 9),       # 1本の線を作るのに繋ぐ断片の数（4〜8）
    turns  = (0.6, 1.0),   # 1本の線が回る量（周）
    maxper = 60.0,         # 繋ぎ目で配る角度の上限（度）。折り返し防止
    tilt   = 0.40,         # 文字を縦に立てるときの傾きの幅（ラジアン）
    frac   = 0.15,         # 繋いだ線をならす強さ
    ndot   = (0, 2),       # ポツンと置く小さい線の数（0〜1）
    dsize  = 0.13,         # その線の大きさ（キャンバスの高さに対する割合）
    mix    = 3,            # 形を混ぜるときに使う線の数（1ならコピー）
    t      = 0.05,         # 形のばらつき（線の大きさに対する割合）
)


# ======================================================================
# 基本の道具
# ======================================================================

def resample(p, n):
    """線を n 点に等間隔で取り直す"""
    p = np.asarray(p, float)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] < 1e-12:
        return np.repeat(p[:1], n, 0)
    u = np.linspace(0.0, d[-1], n)
    return np.stack([np.interp(u, d, p[:, j]) for j in (0, 1)], 1)


def smooth(a, w=9):
    """移動平均。長さを保つ"""
    a = np.asarray(a, float)
    w = max(1, min(int(w), len(a)))
    k = np.ones(w) / w
    return np.stack([np.convolve(a[:, j], k, "same") for j in (0, 1)], 1)


def relax(p, frac=0.15):
    """端を保ったまま、小刻みな波をならす"""
    n   = len(p)
    w   = max(3, int(n * frac) | 1)
    pad = w // 2
    q   = np.vstack([np.repeat(p[:1], pad, 0), p, np.repeat(p[-1:], pad, 0)])
    k   = np.ones(w) / w
    return np.stack([np.convolve(q[:, j], k, "valid") for j in (0, 1)], 1)


def rot(p, a):
    """原点まわりに a ラジアン回す"""
    c, s = np.cos(a), np.sin(a)
    return p @ np.array([[c, s], [-s, c]])


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def in_dir(p, k=5):
    k = min(k, len(p) - 1)
    return _unit(p[k] - p[0])


def out_dir(p, k=5):
    k = min(k, len(p) - 1)
    return _unit(p[-1] - p[-1 - k])


def signed_turn(p):
    """線が全体で何ラジアン回ったか（符号つき）"""
    d = np.diff(p, axis=0)
    a = np.arctan2(d[:, 1], d[:, 0])
    return float(((np.diff(a) + np.pi) % (2 * np.pi) - np.pi).sum())


def principal_angle(p):
    """線のかたまりの長い軸の向き"""
    a = p - p.mean(0)
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    return float(np.arctan2(vt[0, 1], vt[0, 0]))


# ======================================================================
# 学習済みの中身（fit か load_model で埋まる）
# ======================================================================

A32 = LAB = SZ = X = None
BIG = SMALL = BG = WOUT = WSM = None


def _bend(p):
    """曲がりの総量。直線=0 / 半回転=1 / 一周の輪=2"""
    a = resample(p, 32)
    e = np.diff(a, axis=0)
    ang = np.arctan2(e[:, 1], e[:, 0])
    return float(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi).sum() / np.pi)


def _nrm(A):
    """ブロックの大きさ（RMS）を1に揃える"""
    A = np.asarray(A, float)
    if A.ndim == 1:
        A = A[:, None]
    A = A - A.mean(0)
    return A / np.sqrt((A ** 2).sum(1).mean())


def _kmeans(Xm, k, seed=0):
    """Bellona.Analysis と同じ手順の k-means"""
    r = np.random.RandomState(seed)
    C, seen = [], set()
    for i in r.permutation(len(Xm)):
        t = tuple(Xm[i])
        if t in seen:
            continue
        seen.add(t)
        C.append(Xm[i])
        if len(C) == k:
            break
    C, lab, prev = np.array(C), None, None
    for _ in range(300):
        lab  = np.argmin(((Xm[:, None] - C[None]) ** 2).sum(2), 1)
        keep = [i for i in range(len(C)) if (lab == i).any()]
        C    = np.array([Xm[lab == i].mean(0) for i in keep])
        lab  = np.array([keep.index(v) for v in lab])
        if prev is not None and np.array_equal(prev, lab):
            break
        prev = lab
    return lab


def _derive():
    """A32 / SZ / LAB から、生成に使う重みを作る"""
    global BIG, SMALL, BG, WOUT, WSM
    BIG   = np.where(SZ >= 0.5)[0]
    SMALL = np.where(SZ <  0.5)[0]

    # 「輪郭らしさ」：縦横が1に近く、始点と終点が近い（閉じている）ほど重い
    BG  = np.array(BIG)
    arb = np.array([np.ptp(A32[i][:, 0]) / max(np.ptp(A32[i][:, 1]), 1e-9) for i in BG])
    cls = np.array([np.linalg.norm(A32[i][0] - A32[i][-1]) / max(SZ[i], 1e-9) for i in BG])
    w   = np.exp(-(np.log(arb) ** 2) / (2 * 0.35 ** 2))
    w  *= np.exp(-(cls ** 2) / (2 * 0.40 ** 2))
    WOUT = w / w.sum()

    WSM = SZ[SMALL] / SZ[SMALL].sum()


def fit(path="pieces.pkl", k=K, seed=0):
    """学習データを読んで、特徴量と分類を作る（1回だけ実行する）"""
    global A32, LAB, SZ, X

    d      = pickle.load(open(path, "rb"))
    pieces = d["pieces"] if isinstance(d, dict) else (d[0] if isinstance(d, tuple) else d)
    allp   = [np.asarray(s, float) for pc in pieces for s in pc]

    A32 = np.array([resample(p, 32) - resample(p, 32).mean(0) for p in allp])
    SZ  = np.array([max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) for p in A32])

    # 特徴量：32点の座標64個 ＋ 曲がりの総量1個（重さ2.5倍）
    X   = np.c_[_nrm(A32.reshape(len(allp), -1)),
                _nrm([_bend(p) for p in allp]) * 2.5]
    LAB = _kmeans(X, k, seed)

    _derive()
    return dict(photos=len(pieces), strokes=len(allp),
                groups=int(LAB.max()) + 1, counts=np.bincount(LAB).tolist())


def save_model(path="model.pkl"):
    """学習結果を保存する"""
    with open(path, "wb") as f:
        pickle.dump(dict(A32=A32, LAB=LAB, SZ=SZ, X=X, P=P,
                         MM_W=MM_W, MM_H=MM_H, ROW_H=ROW_H, NROW=NROW,
                         version="2026-10-28"), f)
    return path


def load_model(path="model.pkl"):
    """学習結果を読み込む。生成だけならこれでよい"""
    global A32, LAB, SZ, X, P, MM_W, MM_H, ROW_H, NROW, H, W
    d = pickle.load(open(path, "rb"))
    A32, LAB, SZ, X = d["A32"], d["LAB"], d["SZ"], d["X"]
    P = dict(d["P"])
    MM_W, MM_H = d["MM_W"], d["MM_H"]
    ROW_H, NROW = d["ROW_H"], d["NROW"]
    H = ROW_H * NROW
    W = H * MM_W / MM_H
    _derive()
    return dict(strokes=len(A32), groups=int(LAB.max()) + 1,
                version=d.get("version", "?"))


# ======================================================================
# 形を作る
# ======================================================================

def pick_shape(i, rng, mix=3):
    """同じグループの近い線を混ぜて、新しい形を作る。mix=1 ならコピー"""
    if mix <= 1:
        return A32[i].copy()
    cand = np.where(LAB == LAB[i])[0]
    d    = ((X[cand] - X[i]) ** 2).sum(1)
    near = cand[np.argsort(d)[:mix]]
    w    = rng.dirichlet(np.ones(len(near)))
    return (A32[near] * w[:, None, None]).sum(0)


def jitter(shape, t, rng):
    """形をなめらかに少しずらす"""
    e = smooth(rng.normal(0, 1, shape.shape))
    e = e / max(np.abs(e).max(), 1e-9)
    s = max(np.ptp(shape[:, 0]), np.ptp(shape[:, 1]), 1e-6)
    return shape + e * s * t


def build_round(rng, p):
    """小さい線を繋いで、丸みのある長い一本の線にする"""
    n   = int(rng.integers(*p["nseg"]))
    shp = [jitter(pick_shape(int(rng.choice(SMALL, p=WSM)), rng, p["mix"]), p["t"], rng)
           for _ in range(n)]

    goal = rng.choice([-1, 1]) * 2 * np.pi * rng.uniform(*p["turns"])
    per  = (goal - sum(signed_turn(s) for s in shp)) / n
    lim  = np.radians(p["maxper"])
    per  = float(np.clip(per, -lim, lim))          # 折り返して団子になるのを防ぐ

    pts, tip = [], np.zeros(2)
    head = rng.uniform(-np.pi, np.pi)
    for s in shp:
        e = in_dir(s)
        q = rot(s, head - np.arctan2(e[1], e[0]))
        q = q - q[0] + tip
        pts.append(q)
        d    = out_dir(q)
        head = np.arctan2(d[1], d[0]) + per
        tip  = q[-1]
    return relax(np.vstack(pts), p["frac"])


def make_letter(rng, p, size):
    """文字 = 丸い線を1〜2本重ね、縦に立てて、高さを size に合わせたもの"""
    base = build_round(rng, p)
    base = base - base.mean(0)
    s0   = max(np.ptp(base[:, 0]), np.ptp(base[:, 1]), 1e-9)
    L    = [base]
    for _ in range(int(rng.integers(*p["nstk"])) - 1):
        q = build_round(rng, p)
        q = q - q.mean(0)
        q = q * (s0 / max(np.ptp(q[:, 0]), np.ptp(q[:, 1]), 1e-9)) * rng.uniform(0.6, 1.1)
        L.append(q + rng.normal(0, p["spread"] * s0, 2))

    a   = np.vstack(L)
    ctr = a.mean(0)
    tgt = np.pi / 2 + rng.uniform(-p["tilt"], p["tilt"])        # 縦に立てる
    L   = [rot(q - ctr, tgt - principal_angle(a)) for q in L]

    a = np.vstack(L)
    k = size * rng.uniform(0.85, 1.15) / max(np.ptp(a[:, 1]), 1e-9)
    return [q * k for q in L]


def make_word(rng, p, size):
    """単語 = 文字が2〜3個、重なって横に並んだもの"""
    parts, x0 = [], 0.0
    for _ in range(int(rng.integers(*p["nlet"]))):
        L   = make_letter(rng, p, size)
        a   = np.vstack(L)
        off = np.array([x0 - a[:, 0].min(), -a[:, 1].min()])
        L   = [q + off for q in L]
        parts += L
        x0 = max(q[:, 0].max() for q in L) + p["gap"] * size
    return parts


def add_dots(out, rng, p):
    """小さい線を、そのまま変な場所にポツンと置く"""
    for _ in range(int(rng.integers(*p["ndot"]))):
        s = jitter(pick_shape(int(rng.choice(SMALL, p=WSM)), rng, p["mix"]), p["t"], rng)
        s = s - s.mean(0)
        s = s * (p["dsize"] * H / max(np.ptp(s[:, 0]), np.ptp(s[:, 1]), 1e-9))
        out.append(s + np.array([rng.uniform(0.08, 0.92) * W,
                                 rng.uniform(0.08, 0.92) * H]))
    return out


def shrink_to_fit(g, margin=0.04):
    """枠からはみ出したときだけ縮める（拡大はしない）"""
    a      = np.vstack(g)
    lo, hi = a.min(0), a.max(0)
    ext    = np.maximum(hi - lo, 1e-9)
    k      = min(1.0, W * (1 - 2 * margin) / ext[0], H * (1 - 2 * margin) / ext[1])
    off    = np.array([W, H]) / 2 - (lo + hi) / 2 * k
    return [q * k + off for q in g]


def generate(seed=None, fit_frame=True, **kw):
    """落書きを1枚作る。戻り値は線のリスト（各要素は (n,2) の座標）"""
    if A32 is None:
        raise RuntimeError("先に fit() か load_model() を呼んでください")

    p   = {**P, **kw}
    rng = np.random.default_rng(seed)
    nr  = p["nrow"]
    rh  = H / nr
    nw  = int(rng.integers(*p["nword"]))

    share = [0] * nr
    for i in range(nw):
        share[i % nr] += 1

    out = []
    for r in range(nr):
        if share[r] == 0:
            continue
        words = [make_word(rng, p, rh * p["lsize"]) for _ in range(share[r])]
        wid   = [np.ptp(np.vstack(w)[:, 0]) for w in words]
        free  = max(W - sum(wid), 0.0)
        sp    = rng.dirichlet(np.ones(share[r] + 1)) * free
        x = sp[0]
        for i, wrd in enumerate(words):
            a   = np.vstack(wrd)
            off = np.array([x - a[:, 0].min(),
                            r * rh + (rh - np.ptp(a[:, 1])) * rng.random() - a[:, 1].min()])
            out += [q + off for q in wrd]
            x   += wid[i] + sp[i + 1]

    out = add_dots(out, rng, p)
    return shrink_to_fit(out) if fit_frame else out


# ======================================================================
# 出力
# ======================================================================

def to_svg(g, path, mm_w=None, mm_h=None, stroke=1.2):
    """生成した線を SVG で書き出す。y は上下を反転して紙の向きに合わせる"""
    mm_w = MM_W if mm_w is None else mm_w
    mm_h = MM_H if mm_h is None else mm_h
    kx, ky = mm_w / W, mm_h / H
    body = []
    for s in g:
        pts = " ".join(f"{x * kx:.2f},{(H - y) * ky:.2f}" for x, y in s)
        body.append(f'<polyline points="{pts}" fill="none" '
                    f'stroke="black" stroke-width="{stroke}" '
                    f'stroke-linecap="round" stroke-linejoin="round"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{mm_w}mm" height="{mm_h}mm" '
           f'viewBox="0 0 {mm_w} {mm_h}">\n' + "\n".join(body) + "\n</svg>\n")
    with open(path, "w") as f:
        f.write(svg)
    return path
