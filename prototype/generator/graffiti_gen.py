# -*- coding: utf-8 -*-
"""
落書き生成（2026年11月2日 版）

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
MM_W = 1030                # 横
MM_H = 535                 # 縦

ROW_H = 1.15               # 1段の高さ（落書き全体の高さ=1 のスケール）
NROW  = 2                  # 段数
H = ROW_H * NROW
W = H * MM_W / MM_H

P = dict(
    nrow   = None,         # 段数。None なら p1row の確率で 1 か 2 を引く
    p1row  = 0.60,         # 1段になる確率（1段のときは大きく少なく置かれる）
    nword  = (2, 4),       # 2段のときの単語の数（2〜3）
    nword1 = (1, 3),       # 1段のときの単語の数（1〜2）
    nline  = (8, 15),      # 線の本数がこの範囲（8〜14本）に入るまで作り直す。
    nline1 = (5, 11),      # 1段のときの本数（5〜10本）。1段は単語が1〜2個なので
                           # 2段と同じ下限では満たせず作り直しを繰り返すことになる。
                           # None なら制限なし。途中で打ち切らないので形は変わらない
    aspect = (1.75, 2.10), # 絵の縦横比（横/縦）がこの範囲に入るまで作り直す。
                           # 枠は 1.925。外れるほど上下か左右に余白が出る
    fillgap= True,         # 空いているところに線を足して埋める
    fillto = 0.22,         # どこまで埋めるか（空白の正方形がこの大きさ以下になるまで）
    fillmax= 2,            # 足してよい本数の上限（線の本数の上限も超えない）
    fillsize= 0.80,        # 足す文字の大きさ（空白の正方形に対する割合）
    maxhole= 0.50,         # 絵の中に空いてよい正方形の大きさ（枠に対する割合）。
                           # 小さくすると、中に大きな空白のある絵が出なくなる
    ntry   = 400,          # 何回まで作り直すか
    nlet   = (2, 9),       # 1単語の文字数（2〜8）
    gapmax = 0.30,         # 文字どうしを離してよい上限（文字の高さに対する割合）
    fillw  = 0.00,         # 単語の長さを「段の中の持ち分の幅」に合わせる度合い。
                           # 0 なら文字数は nlet から引くだけ（段がガラ空きになりうる）
    loose  = 0.45,         # 横の置き場所の自由さ。0=持ち場にきっちり / 1=全幅のどこでも
    xspread= 1.00,         # 段の中の単語を、位置だけ横に引き伸ばして端に合わせる度合い。
                           # 0 なら何もしない。形も大きさも変わらず、すきまだけ広がる
    xletmax= 0.15,         # 文字どうしを広げてよい上限（文字の幅に対する割合）
    xmargin= 0.04,         # 横に広げるときに残す余白
    ymargin= 0.04,         # 段の帯を作るときに上下に残す余白         # そのときに残す左右の余白（枠の幅に対する割合）
    rowsplit= True,        # 2段のとき、上下の帯を分けて交わらないようにする
    rowsep = 0.03,         # そのときの段と段のすきま（枠の高さに対する割合）
    rowlap = 0.25,         # rowsplit=False のときだけ使う。段の間隔の詰め方
    yjit   = 0.12,         # 段の中での上下の揺れ（段の高さに対する割合）
    cap    = 1.60,         # 枠に合わせて拡大してよい上限（1.0 なら拡大しない）
    pstk   = (0.55, 0.30, 0.15),   # 1文字が何本の線でできるかの確率（1本/2本/3本）
                                   # 文字ごとに引き直す。密な文字と開いた文字が混ざる
    lsize  = 0.95,         # 1文字の高さ（段の高さに対する割合）の基準
    lsize1 = 0.95,         # 1段のときの1文字の高さ。1段だと段の高さ＝枠の高さなので
                           # lsize のままだと文字が2倍になってしまう
    lvar   = (0.70, 1.25), # 文字ごとの高さのばらつき。広げると高さが揃わなくなる
    gap    = -0.15,        # 文字どうしの重なり（文字の高さに対する割合）
    spread = 0.35,         # 1文字の中で線をずらす幅
    nseg   = (4, 9),       # 1本の線を作るのに繋ぐ断片の数（4〜8）
    turns  = (0.8, 1.3),   # 1本の線が回る量（周）。2本以上でできる字に使う
    turns1 = (1.1, 1.6),   # 1本だけでできる字に使う。その1本で形にするので大きく回す   # 1本の線が回る量（周）
    maxper = 60.0,         # 繋ぎ目で配る角度の上限（度）。折り返し防止
    tilt   = 0.40,         # 文字を縦に立てるときの傾きの幅（ラジアン）
    frac   = 0.15,         # 繋いだ線をならす強さ
    ndot   = (0, 2),       # ポツンと置く小さい線の数（0〜1）
    dnear  = 0.18,         # ポツンと置く線を、他の線からどれくらい離してよいか
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
                         version="2026-11-02"), f)
    return path


def load_model(path="model.pkl"):
    """学習結果を読み込む。生成だけならこれでよい。

    読むのは学習で得たもの（線とその分類）だけ。
    設定（P・寸法）はこのファイルの上のほうに書いてあるものを使う。
    model.pkl にも古い設定が入っているが、意図的に無視している
    （入れてしまうと、このファイルの P を書き換えても効かなくなるため）。
    """
    global A32, LAB, SZ, X
    d = pickle.load(open(path, "rb"))
    A32, LAB, SZ, X = d["A32"], d["LAB"], d["SZ"], d["X"]
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


def make_letter(rng, p, size, maxstk=None):
    """文字 = 丸い線を何本か重ね、縦に立てて、高さを size に合わせたもの。

    何本重ねるかは pstk の確率で引く。文字ごとに引き直すので、
    1本だけの開いた文字と、何本も重なった密な文字が混ざる。
    """
    base = build_round(rng, p)
    base = base - base.mean(0)
    s0   = max(np.ptp(base[:, 0]), np.ptp(base[:, 1]), 1e-9)
    L    = [base]
    w    = np.asarray(p["pstk"], float)
    nst  = 1 + int(rng.choice(len(w), p=w / w.sum()))
    if maxstk is not None:
        nst = max(1, min(nst, int(maxstk)))

    # 1本でできる字は、その1本だけで字の形にならないといけないので、よく回す。
    # 2〜3本重なる字は、重なりで形になるのでそこまで回さなくてよい。
    if nst == 1 and p.get("turns1") is not None:
        p = {**p, "turns": p["turns1"]}
    for _ in range(nst - 1):
        q = build_round(rng, p)
        q = q - q.mean(0)
        q = q * (s0 / max(np.ptp(q[:, 0]), np.ptp(q[:, 1]), 1e-9)) * rng.uniform(0.6, 1.1)
        L.append(q + rng.normal(0, p["spread"] * s0, 2))

    a   = np.vstack(L)
    ctr = a.mean(0)
    tgt = np.pi / 2 + rng.uniform(-p["tilt"], p["tilt"])        # 縦に立てる
    L   = [rot(q - ctr, tgt - principal_angle(a)) for q in L]

    a = np.vstack(L)
    k = size * rng.uniform(*p["lvar"]) / max(np.ptp(a[:, 1]), 1e-9)
    return [q * k for q in L]


def make_word(rng, p, size, target=None, budget=None):
    """単語 = 文字が何個か横に並んだもの。

    budget … この単語に割り当てられた線の本数。使い切るまで文字を足す。
    target … 届かせたい横幅。文字数ではなく「文字どうしの間隔」で合わせる。
             間隔は gapmax までしか開かない（開きすぎると単語に見えなくなるため）。
    """
    hi = max(1, p["nlet"][1] - 1)

    Ls, used = [], 0
    if budget is None:
        # 本数の予算を使わないとき：文字数を一度だけ引く
        n = int(rng.integers(*p["nlet"])) if target is None else hi
        for i in range(n):
            Ls.append(make_letter(rng, p, size))
            if target is not None and i + 1 >= p["nlet"][0]:
                w = sum(float(np.ptp(np.vstack(L)[:, 0])) for L in Ls)
                if w + p["gap"] * size * i >= target:
                    break
    else:
        while len(Ls) < hi and used < budget:
            L = make_letter(rng, p, size, maxstk=budget - used)
            used += len(L)
            Ls.append(L)

    # 素の間隔で並べたときの幅を見て、足りなければ間隔を開く
    wds = [float(np.ptp(np.vstack(L)[:, 0])) for L in Ls]
    g   = p["gap"] * size
    if target is not None and len(Ls) > 1:
        need = (target - sum(wds)) / (len(Ls) - 1)
        g    = float(np.clip(need, g, p["gapmax"] * size))

    out, x0 = [], 0.0
    for L in Ls:
        a   = np.vstack(L)
        off = np.array([x0 - a[:, 0].min(), -a[:, 1].min()])
        L   = [q + off for q in L]
        out.append(L)                      # 文字ごとにまとめて返す
        x0 = max(q[:, 0].max() for q in L) + g
    return out


def add_dots(out, rng, p, n=None):
    """小さい線をポツンと置く。ただし他の形の近くにだけ置く。
    離れた場所に1本だけ独立して描かれることは実際の落書きでは起きないため。"""
    n = int(rng.integers(*p["ndot"])) if n is None else int(n)
    if n <= 0 or not out:
        return out

    for _ in range(n):
        s = jitter(pick_shape(int(rng.choice(SMALL, p=WSM)), rng, p["mix"]), p["t"], rng)
        s = s - s.mean(0)
        s = s * (p["dsize"] * H / max(np.ptp(s[:, 0]), np.ptp(s[:, 1]), 1e-9))

        # すでに置いてある線を1本選び、その近くに置く
        host = out[int(rng.integers(len(out)))]
        base = host[int(rng.integers(len(host)))]
        ang  = rng.uniform(-np.pi, np.pi)
        r    = p["dnear"] * H * rng.uniform(0.5, 1.0)
        pos  = base + np.array([np.cos(ang), np.sin(ang)]) * r
        out.append(s + pos)
    return out


def fit_to_frame(g, cap=1.0, margin=0.04):
    """枠に合わせて大きさを揃える。

    cap は拡大してよい上限。cap=1.0 なら縮めるだけ（旧 shrink_to_fit と同じ）。
    cap>1.0 なら、枠より小さく生まれた絵を cap 倍まで広げて余白を減らす。
    """
    a      = np.vstack(g)
    lo, hi = a.min(0), a.max(0)
    ext    = np.maximum(hi - lo, 1e-9)
    k      = min(cap, W * (1 - 2 * margin) / ext[0], H * (1 - 2 * margin) / ext[1])
    off    = np.array([W, H]) / 2 - (lo + hi) / 2 * k
    return [q * k + off for q in g]


def generate(seed=None, fit_frame=True, **kw):
    """落書きを1枚作る。戻り値は線のリスト（各要素は (n,2) の座標）

    nline = (lo, hi) を入れると、線の本数がその範囲に入るまで作り直す。
    途中で打ち切らないので、形は本数を指定しないときと同じ性質のまま。
    ntry 回ためして入らなければ、いちばん近かったものを返す。
    """
    if A32 is None:
        raise RuntimeError("先に fit() か load_model() を呼んでください")

    p    = {**P, **kw}
    rng  = np.random.default_rng(seed)
    want = p["nline"]

    asp  = p["aspect"]
    free = want is None and asp is None and p["maxhole"] is None

    # 段数は最初に1回だけ引いて、作り直しの間は変えない。
    # 作り直すたびに引くと、条件を通りやすい段数（1段）ばかりが残ってしまう。
    p = dict(p)
    if p["nrow"] is None:
        p["nrow"] = 1 if rng.random() < p["p1row"] else 2
    # 1段のときは線の本数の範囲を nline1 に差し替える
    if p["nrow"] == 1 and p.get("nline1"):
        p["nline"] = p["nline1"]
        want       = p["nline"]
    best, bestd = None, None
    for _ in range(1 if free else int(p["ntry"])):
        out = _compose(rng, p)
        d   = 0.0

        if want is not None:                       # 線の本数
            n = len(out)
            if n < want[0]:
                d += (want[0] - n) / max(want[0], 1)
            elif n >= want[1]:
                d += (n - want[1] + 1) / max(want[1], 1)

        if p["maxhole"] is not None:               # 絵の中に空く穴の大きさ
            h = _hole(out)
            if h > p["maxhole"]:
                # 条件を全部は満たせないとき、大きな穴がいちばん困るので重く見る
                d += 3.0 * (h - p["maxhole"]) / p["maxhole"]

        if asp is not None:                        # 絵の縦横比
            q = np.vstack(out)
            e = np.maximum(q.max(0) - q.min(0), 1e-9)
            r = e[0] / e[1]
            if r < asp[0]:
                d += (asp[0] - r) / asp[0]
            elif r > asp[1]:
                d += (r - asp[1]) / asp[1]

        if d == 0.0:
            best = out
            break
        if bestd is None or d < bestd:
            best, bestd = out, d

    return fit_to_frame(best, p["cap"]) if fit_frame else best


def _hole(g, n=8, where=False):
    """絵の中にできる「線が1本も通らない正方形」の大きさ。
    枠を n x n のマスに切って測る。0.5 なら枠の半分の辺の正方形が空いている。"""
    q  = np.vstack(g)
    gx = np.clip((q[:, 0] / W * n).astype(int), 0, n - 1)
    gy = np.clip((q[:, 1] / H * n).astype(int), 0, n - 1)
    c  = np.zeros((n, n), bool)
    c[gy, gx] = True

    best, bi, bj = 0, 0, 0
    for i in range(n):
        for j in range(n):
            k = best + 1
            while i + k <= n and j + k <= n and not c[i:i + k, j:j + k].any():
                best, bi, bj = k, i, j
                k += 1
    if not where:
        return best / n
    # 空いている正方形の中心（絵の座標）
    cy = (bi + best / 2) / n * H
    cx = (bj + best / 2) / n * W
    return best / n, cx, cy


def _fill_holes(out, rng, p):
    """空いているところに線を足して埋める。

    いちばん大きな空白の正方形を探し、その真ん中に文字をひとつ置く。
    maxhole 以下になるか、線の本数が上限に達したら止める。
    """
    if not p["fillgap"] or not out:
        return out

    nmax = (p["nline"][1] - 1) if p["nline"] else 99
    for _ in range(int(p["fillmax"])):
        if len(out) >= nmax:
            break
        h, cx, cy = _hole(out, where=True)
        if h <= p["fillto"]:
            break

        size = h * H * p["fillsize"]
        L    = make_letter(rng, p, size)
        a    = np.vstack(L)
        off  = np.array([cx, cy]) - (a.min(0) + a.max(0)) / 2
        out += [q + off for q in L]
    return out


def _compose(rng, p):
    """1枚ぶんの線を組む（枠への収めはしない）"""
    nr = p["nrow"]
    rh = H / nr

    nw    = int(rng.integers(*(p["nword1"] if nr == 1 else p["nword"])))
    share = [0] * nr
    for i in range(nw):
        share[i % nr] += 1

    lz  = p["loose"]
    out = []
    for r in range(nr):
        n   = share[r]
        row = []
        for i in range(n):
            # lsize は段の高さに対する割合なので、1段だと文字が2倍になってしまう。
            # 1段のときは lsize1 を使う。
            ls     = p["lsize1"] if (nr == 1 and p.get("lsize1")) else p["lsize"]
            wrd    = make_word(rng, p, rh * ls)        # 文字のリスト
            a      = np.vstack([q for L in wrd for q in L])
            wd, ht = np.ptp(a[:, 0]), np.ptp(a[:, 1])

            # 横：持ち場の中心はそのまま、幅だけ loose で広げる
            hw = 0.5 / n * (1 + lz * (n - 1))
            cx = (i + 0.5) / n + (2 * rng.random() - 1) * hw
            x  = float(np.clip(cx, 0.0, 1.0)) * max(W - wd, 0.0)

            # 縦の置き方
            if nr > 1 and p["rowsplit"]:
                # 段ごとに帯を分け、上下が交わらないようにする。
                # 帯より高い単語は、その単語だけ縮めて帯に収める（形は変わらない）。
                # 枠いっぱいに作ると、あとで fit_to_frame が余白を作るために
                # 全体を縮めてしまう。最初から余白のぶん内側に帯を作る。
                vm  = p["ymargin"] * H
                sep = p["rowsep"] * H
                bh  = (H - 2 * vm - sep * (nr - 1)) / nr
                blo = vm + r * (bh + sep)
                if ht > bh:
                    k      = bh / ht
                    wrd    = [[q * k for q in L] for L in wrd]
                    a      = np.vstack([q for L in wrd for q in L])
                    wd, ht = np.ptp(a[:, 0]), np.ptp(a[:, 1])
                    x      = float(np.clip(cx, 0.0, 1.0)) * max(W - wd, 0.0)
                y = blo + (bh - ht) * rng.random()
            else:
                pitch = rh * (1 - p["rowlap"])
                cy    = (H / 2 + (r - (nr - 1) / 2) * pitch
                         + (2 * rng.random() - 1) * p["yjit"] * rh)
                y     = float(np.clip(cy - ht / 2, 0.0, max(H - ht, 0.0)))

            off = np.array([x - a[:, 0].min(), y - a[:, 1].min()])
            row.append([[q + off for q in L] for L in wrd])   # 単語 → 文字 → 線

        out += _spread_x(row, p, rng)

    out = _fill_holes(out, rng, p)
    return add_dots(out, rng, p)


def _shift(items, d):
    return [[q + np.array([d, 0.0]) for q in L] for L in items]


def _spread_x(row, p, rng):
    """段の中身を横に広げて枠の端に合わせる。

    まず単語どうしのすきまを広げる（文字の間隔は変わらない）。
    それでも届かないぶんだけ、文字どうしを広げる。ただし xletmax まで。
    形も大きさも変えない。動かすのは位置だけ。
    """
    words = [w for w in row if w]
    flat  = [q for w in words for L in w for q in L]
    if not flat or p["xspread"] <= 0:
        return flat

    m  = p["xmargin"] * W
    ta, tb = m, W - m

    def span(ws):
        xs = [q[:, 0] for w in ws for L in w for q in L]
        return min(x.min() for x in xs), max(x.max() for x in xs)

    # ── 1段目：単語どうしを広げる
    if len(words) > 1:
        wl = [min(q[:, 0].min() for L in w for q in L) for w in words]
        wr = [max(q[:, 0].max() for L in w for q in L) for w in words]
        order = sorted(range(len(words)), key=lambda i: wl[i])
        need  = (tb - ta) - sum(wr[i] - wl[i] for i in order)
        if need > 0:
            step = need / (len(order) - 1) if len(order) > 1 else 0.0
            x = ta
            new = [None] * len(words)
            for i in order:
                new[i] = _shift(words[i], x - wl[i])
                x += (wr[i] - wl[i]) + step * p["xspread"]
            words = new

    # ── 2段目：まだ届かなければ、文字どうしを広げる（上限つき）
    a, b = span(words)
    if (tb - ta) - (b - a) > 1e-9:
        lets = [L for w in words for L in w]
        if len(lets) > 1:
            ll = [min(q[:, 0].min() for q in L) for L in lets]
            lr = [max(q[:, 0].max() for q in L) for L in lets]
            order = sorted(range(len(lets)), key=lambda i: ll[i])
            need  = (tb - ta) - sum(lr[i] - ll[i] for i in order)
            step  = max(need, 0.0) / max(len(order) - 1, 1)
            cap   = p["xletmax"] * np.median([lr[i] - ll[i] for i in order])
            step  = min(step * p["xspread"], cap)
            x, out = a, []
            for i in order:
                out += [q + np.array([x - ll[i], 0.0]) for q in lets[i]]
                x += (lr[i] - ll[i]) + step
            return out

    return [q for w in words for L in w for q in L]


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
