# -*- coding: utf-8 -*-
"""
落書き生成（2026年11月17日 版）

写真から取り出した一筆の線を分類し、その線を混ぜて新しい線を作り、
文字・単語・段という構造で並べて、1枚の落書きを生成する。

文字の輪郭は2通りの作り方を混ぜている（pbig で割合を決める）。
  a. 大きい輪郭の線（文字まるごとの輪郭）を、近いもの3本で混ぜる
  b. 小さい線を何本も繋いで、丸みのある長い一本にする

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

import os
import pickle
import numpy as np


# ======================================================================
# 設定
# ======================================================================

K = 7                      # 分類のグループ数

# 描画面の実寸（mm）。最終出力の大きさ（2026-11-17 に 1030 x 535 から変更）。横長。
# 枠の比（横/縦）は 1.29。11/2 の 1030 x 710（比 1.45）に近いので、
# 下の p1row / nline / fillto / fillmax は 710 のときの値に戻してある。
MM_W = 1330                # 横
MM_H = 1030                # 縦

ROW_H = 1.15               # 1段の高さ（落書き全体の高さ=1 のスケール）
NROW  = 2                  # 段数
H = ROW_H * NROW
W = H * MM_W / MM_H

P = dict(
    nrow   = None,         # 段数。None なら p1row の確率で 1 か 2 を引く
    p1row  = 0.20,         # 1段になる確率（1段のときは大きく少なく置かれる）
                           # 原作の出力は1段が2割ほど
    nword  = (2, 4),       # 2段のときの単語の数（2〜3）
    nword1 = (1, 2),       # 1段のときの単語の数（1つ）。この枠の比だと、1段に入るのは
                           # 高さいっぱいの文字が2〜3個。それより多く作っても条件で捨てられるだけ
                           # （作り直しの9割以上が「横に長すぎる」だった）ので、最初から入る数だけ作る
    nlet1  = (2, 4),       # 1段のときの1単語の文字数（2〜3）
    nline  = (13, 22),     # 2段のときの線の本数。この範囲（13〜21本）に入るまで作り直す。11/20 に 11〜19 から上げた（大きいほうの段で1本だけの文字をなくしたため）。
                           # 重なった所を消すと輪郭が切れて本数が増えるので、その分も含む（11/18）
                           # 重ならずに入る文字は1段に4〜5個なので、これ以上増やすと詰まる
    nline1 = (6, 11),      # 1段のときの本数（6〜10本）。実際に出るのは6〜8本ほど。5本だと物足りない
                           # （単語が2つあふれて重なる絵を maxover で作り直すため）。1段は単語が1〜2個なので
                           # 2段と同じ下限では満たせず作り直しを繰り返すことになる。
                           # None なら制限なし。途中で打ち切らないので形は変わらない
    aspect = (1.17, 1.41), # 絵の縦横比（横/縦）がこの範囲に入るまで作り直す。
                           # 枠は 1.291。外れるほど上下か左右に余白が出る。
                           # 枠の比を変えたら、1単語の文字数と横縦比が保たれているかも見る
                           # （比だけ合わせると、単語が「大きな文字2つ」に崩れることがある）
    fillgap= True,         # 空いているところに線を足して埋める（2段のときは fillgap2）
    fillgap2 = False,      # 2段のときに空いた所を埋めるか。埋める文字はぽつんと1文字だけ離れて置かれ、
                           # 文字がばらけて見えるので使わない（11/18）
    minlet = 3,            # 2段のとき、1語の文字数の下限。2文字の単語が離れて並ぶと、
                           # 文字がばらけて見える（11/18）。段に入らなければ1語にする
    fillto = 0.30,         # どこまで埋めるか（空白の正方形がこの大きさ以下になるまで）
    fillmax= 3,            # 足してよい本数の上限（線の本数の上限も超えない）
    fillsize= 0.80,        # 足す文字の大きさ（空白の正方形に対する割合）
    maxcrowd= 5,           # 詰まりすぎの上限。枠を 10x10 のマスに切ったとき、
                           # 1マスを通ってよい線の本数。超えるマスがあれば作り直す。
                           # None なら見ない
    maxshort= 3,           # 短い線（ちょんちょんとした線）は1枚に何本までか。超えたら作り直す。
    shortsz= 0.13,         # 短い線とみなす大きさ（枠の高さに対する割合。これより小さい線を数える）
    maxover= 1.00,         # 段からあふれてよい度合い。1段に置いた単語の幅の合計が、
                           # 段の幅（左右の余白を除く）の何倍まで許すか。1.0 なら収まる絵だけ。
                           # 超えると単語どうしが重なって団子になるので作り直す。None なら見ない
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
    xletmax= 0.00,         # 文字どうしを広げてよい上限（文字の幅に対する割合）。
                           # 0 なら文字は単語の中で接したまま。原作は文字が接して帯になっている
    xmargin= 0.01,         # 横に広げるときに残す余白
    ymargin= 0.01,         # 段の帯を作るときに上下に残す余白         # そのときに残す左右の余白（枠の幅に対する割合）
    rowsplit= True,        # 2段のとき、上下の帯を分けて交わらないようにする
    rowsep = 0.03,         # そのときの段と段のすきま（枠の高さに対する割合）
    rowlap = 0.25,         # rowsplit=False のときだけ使う。段の間隔の詰め方
    yjit   = 0.12,         # 段の中での上下の揺れ（段の高さに対する割合）
    cap    = 1.60,         # 枠に合わせて拡大してよい上限（1.0 なら拡大しない）
    pstk   = (0.30, 0.55, 0.15),   # 1文字が何本の線でできるかの確率（1本/2本/3本）
                                   # 文字ごとに引き直す。密な文字と開いた文字が混ざる
    pstkr  = (0.75, 0.25), # 輪郭が「小さい線を繋いだ丸い線」の文字が、何本の線でできるかの確率（1本/2本）。
                           # 丸い線を3本重ねると、1文字の中で線が団子になる
    pbig   = 0.80,         # 文字の輪郭に「大きい輪郭の線を混ぜたもの」を使う確率。
                           # 残りは「小さい線を繋いだ丸い線」。0 なら 11/3 までと同じ作り方
    pbig1  = 0.75,         # 1段のときの pbig。1段は文字が1m近くになるので、丸い線ばかりだと
                           # 文字ではなく「さまよう線」に見える。大きい輪郭の文字を主にする
    minbig1= 0.50,         # 1段のとき、文字のうち「大きい輪郭の文字」が占める割合の下限。
                           # 下回る絵は作り直す。None なら見ない
    bigopen1 = 0.70,       # 1段のとき、大きい輪郭の線として使ってよい開き具合の上限
                           # （始点と終点の離れ具合）。これより開いた弧は引き直す。None なら見ない
    pstk1  = (0.30, 0.45, 0.25),   # 1段のときの pstk。文字が2〜3個しか並ばないので、
                                   # 輪郭の中の線を増やして 6 本以上にする
    pjoin1 = 0.50,         # 1段のときの pjoin。2文字ぶん続けた輪郭は、繋ぎ目が角になって形が複雑になる
    minwig1 = 2.5,         # 1段のとき、輪郭の「こぶ・くぼみ」の量の下限（全部の線の合計）。
                           # つるんとした楕円が並ぶだけの絵は、落書きとしては単純すぎる。
                           # 1本の線について（曲がりの絶対量 − 正味の曲がり）÷180度。楕円は0に近い
    mincross1 = None,       # 1段のとき、線どうしが交わる回数の下限。文字が離れて並ぶだけの絵を避ける
    minclosed1 = 1,        # 1段のとき、「閉じかけた輪郭」の文字がいくつ以上要るか。
                           # 開いた弧ばかりだと、落書きの文字に見えない。None なら見ない
    closedgap = 0.55,      # 閉じかけているとみなす基準。輪郭の始点と終点の離れ具合
                           # （輪郭の大きさに対する割合）がこれ以下。2文字ぶん続けた輪郭は閉じかけた扱い
    bigasp = (0.55, 1.25), # 大きい輪郭の線の横/縦がこの範囲のものだけ使う（平たい線は文字にならない）
    dmin   = 0.15,         # 混ぜた輪郭が、学習データのどの線ともこれ以上離れているものだけ使う。
                           # 近すぎる（ほぼ写しの）形は引き直す。距離は32点のずれの二乗和の平方根。
                           # 学習データの大きい線どうしでいちばん近い組は 0.057
    binset = 0.30,         # 大きい輪郭の中に小さい線を置くときの、中心からずらす幅
    # ── 原作の画面から言葉にした組み合わせ方のルール（11/18）──
    occlude = True,        # 重なった所は描かない。前にある文字の輪郭の内側に入った、後ろの文字の線を消す。
                           # 線どうしが交差せず、突き当たりで止まる（バブル文字が前後に重なって見える）
    front  = None,         # どちらの文字が前か（"left" / "right"）。None なら1枚ごとにくじで決める
    occgap = 0.60,         # 輪郭の内側を決められる閉じ具合（始点と終点の離れ具合の上限）
    maxhide = 0.30,        # 重なりで隠れてよいのは1文字の線のこの割合まで。超えるときは後ろの文字をずらす（11/20）
    occmin = 0.06,         # 消したあとに残す線の長さの下限（枠の高さに対する割合）
    rowfrac = (0.40, 0.60),# 2段のとき、下の段が使う高さの割合。段の高さはそろえない
                           # （原作は片方の段が小さく密、もう片方が大きく疎）
    pmark  = None,         # 印（交差した2本の小さい線）が入る確率。None なら学習データの割合
                           # （印があった写真の数 ÷ 写真の数）。0 なら入れない
    row1h  = 0.50,         # 1段のとき、帯の高さ（枠の高さに対する割合）。帯は上下の真ん中あたりに置き、
                           # 2段と同じく帯の幅に入る数だけ文字を作る。None なら枠いっぱい（11/17 までの1段）。
                           # 枠いっぱいだと文字が約1mになり、原作のどの文字より2倍ほど大きく不自然だった
    minwidth1 = 0.85,  
    nline1b = (9, 16),     # 1段を帯にするときの線の本数（9〜15本）。8本以下は線が少なすぎて寂しい（11/18）
    minink1 = 5.0,         # 1段を帯にするとき、線の総延長の下限（枠の高さに対する倍数）。
                           # 下回る絵は全体として寂しく見える（11/18。40枚の中央は5.8、寂しいと言われた絵は4.3）    # 1段の帯の長さの下限（枠の幅に対する割合）。文字が2個しかない短い帯は作り直す
    lnarrow = 1.00,        # 文字を横に縮める倍率。重なり（gap）と単語の幅の見積もり（advance）も同じ倍率で縮める。
                           # 0.7 で1段に5〜6個並ぶ（原作の文字数）が、細長く詰め込まれた感じになり採らなかった（11/18）
    wrow   = (1, 3),       # 2段のとき、1段に置く単語の数（1〜2）。段ごとに引く
    advance = (1.00, 0.68),# 単語の幅の見積もり（文字の高さに対する割合）。1文字目の幅と、1文字足すごとに伸びる幅。
                           # 実際に作った単語の幅の中央値に合わせてある（あふれた絵は maxover で作り直す）。
                           # 2文字ぶん続けた輪郭があるので、1文字の平均の幅より大きい
    fmargin = 0.01,        # 枠に合わせるときの余白。原作は枠の端まで描いている
    minstk = 1,            # 1文字の線の本数の下限。2文字ぶん続けた輪郭は minstk + 1 本以上。
                           # 1本だけの文字（輪郭だけの文字）は maxsingle 個まで（11/18）
    singlemax = 0.45,      # 1本だけの文字を作ってよいのは、文字の大きさが枠の高さのこの割合以下のときだけ。
                           # 大きい文字の段（2段の高いほうの段）で1本だけの文字は目立つ（11/20）
    maxsingle = 2,         # 1本だけの文字は1枚に何文字まで許すか。超えたら2本以上で作る。None なら制限なし
    minstk1 = 2,           # 1段のときの minstk。1段は文字が大きいので、1本だけの文字は作らない
    binmin = 0.45,         # 輪郭の中に置く線の大きさの下限（輪郭の幅・高さの大きいほうに対する割合）。
                           # 小さすぎると「ちょんちょんとした短い線」になるので、ここまで大きくする
    binmax = 0.60,         # その小さい線の大きさの上限（輪郭の幅・高さに対する割合）。
                           # 内側の線は輪郭の中にある。上限がないと輪郭より大きい線が出て、
                           # その単語だけ帯に収めるために縮められ、小さすぎる文字の列になる
    mixbig = 3,            # 大きい輪郭の線を混ぜる本数。写しを避けるのは dmin が受け持つので、
    mixsame = True,        # 大きい輪郭は、閉じかけは閉じかけどうし、開いた線は開いた線どうしで、向きをそろえて混ぜる（11/20）
    samethr = 0.35,        # 閉じかけとみなす、始点と終点の離れ具合（輪郭の大きさに対する割合）
    wgap    = (0.25, 0.70),# 1つの段に単語が2つあるときの、単語どうしのすきま（文字の高さに対する割合）。段ごとにくじ（11/20）
    wlet2   = (2, 4),      # 2語に分ける段の、1語の文字数（2〜4）（11/20）
    all_like1 = True,      # 2段の絵のどの段も、輪郭は大きい輪郭の線から・開きすぎない形だけで作る（11/20）
    split_like1 = True,    # 2語に分けた段の文字は、1段の帯と同じ条件で作る（下の pbig2・bigopen1・minstk2）（11/20）
    pbig2   = 1.0,         # 2語に分けた段で、輪郭に大きい輪郭の線を使う確率（小さい線をつないだ輪郭はうねって文字に見えない）
    minstk2 = 2,           # 2語に分けた段の、1文字の線の本数の下限
    wtot2   = (5, 6),      # 2語に分ける段の、段全体の文字数（2文字＋2文字は作らない）
    wdiff   = 0.15,        # 上下とも2語のとき、1語目の文字の割合を上下でこれだけ変える（切れ目の位置をずらす）
    wgdiff  = 0.15,        # 上下とも2語のとき、すきまの広さを上下でこれだけ変える（文字の高さに対する割合）
    wgapmax = 0.90,        # 単語どうしのすきまの上限（文字の高さに対する割合）（11/20）
    minlet2 = 2,           # 段を2語に分けるときの、1語の文字数の下限（11/20。単語の中の文字が密着するようになったので2文字でもまとまって見える）
                           # 3 である必要はない（2 なら元の形が残りやすく、5 ならならされる）
    pjoin  = 0.35,         # 大きい輪郭の文字を「1本の線で2文字ぶん続けて描いた輪郭」にする確率。
                           # 原作の出力では、1本の線が隣の文字の輪郭まで続くことがある
    joinx  = (0.45, 1.00), # そのとき、2文字めを1文字めからどれだけ右に置くか（文字の幅に対する割合）
    joiny  = 0.25,         # そのとき、2文字めの上下のずれの上限（文字の高さに対する割合）
    lsize  = 0.95,         # 1文字の高さ（段の高さに対する割合）の基準
    lsize1 = 0.95,         # 1段のときの1文字の高さ。1段だと段の高さ＝枠の高さなので
                           # lsize のままだと文字が2倍になってしまう
    lvar   = (0.92, 1.08), # 文字ごとの高さのばらつき。原作は段の中で高さがそろっている（11/18 に 0.70〜1.25 から）
    gap    = -0.35,        # 文字どうしの重なり（文字の高さに対する割合）
    contact = (0.10, 0.22), # 単語の中の文字を線どうしで寄せる。線が文字の高さのこの割合だけ食い込む（11/20）。None なら gap の重なりで置く
    spread = 0.35,         # 1文字の中で線をずらす幅
    nseg   = (4, 9),       # 1本の線を作るのに繋ぐ断片の数（4〜8）
    turns  = (0.8, 1.3),   # 1本の線が回る量（周）。2本以上でできる字に使う
    turns1 = (1.1, 1.6),   # 1本だけでできる字に使う。その1本で形にするので大きく回す   # 1本の線が回る量（周）
    maxper = 60.0,         # 繋ぎ目で配る角度の上限（度）。折り返し防止
    tilt   = 0.40,         # 文字を縦に立てるときの傾きの幅（ラジアン）
    frac   = 0.15,         # 繋いだ線をならす強さ
    ndot   = (0, 1),       # ポツンと置く小さい線の数（0〜1）
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
MARKS  = []        # 学習データの中の本物の印。(線i, 線j, 線iの中心, 線jの中心)。中心は印の中心から見た位置
P_MARK = 0.0       # 印があった写真の割合
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

    _find_marks(pieces)
    _derive()
    return dict(photos=len(pieces), strokes=len(allp), marks=len(MARKS),
                groups=int(LAB.max()) + 1, counts=np.bincount(LAB).tolist())


def _cross_frac(a, b):
    """2本の線が1回だけ交わるとき、その交点がそれぞれの線のどこにあるか（始点0〜終点1）。
    交わらない、または2回以上交わるときは None"""
    pa, ra = a[:-1], np.diff(a, axis=0)
    pb, rb = b[:-1], np.diff(b, axis=0)
    den = ra[:, None, 0] * rb[None, :, 1] - ra[:, None, 1] * rb[None, :, 0]
    qp  = pb[None] - pa[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * rb[None, :, 1] - qp[..., 1] * rb[None, :, 0]) / den
        u = (qp[..., 0] * ra[:, None, 1] - qp[..., 1] * ra[:, None, 0]) / den
    ok = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    ii, jj = np.nonzero(ok)
    if len(ii) != 1:
        return None
    la = np.r_[0, np.cumsum(np.linalg.norm(ra, axis=1))]
    lb = np.r_[0, np.cumsum(np.linalg.norm(rb, axis=1))]
    fa = (la[ii[0]] + t[ii[0], jj[0]] * np.linalg.norm(ra[ii[0]])) / la[-1]
    fb = (lb[jj[0]] + u[ii[0], jj[0]] * np.linalg.norm(rb[jj[0]])) / lb[-1]
    return fa, fb


def _find_marks(pieces, small=0.25, near=0.5, mid=(0.2, 0.8)):
    """学習データの中から、本物の印（小さい線2本が途中で1回だけ交わり、中心が近いもの）を探す。

    形は決めない。写真に描かれていたものだけを覚える。
    small … 線の大きさの上限（落書き全体の高さ＝1）
    near  … 2本の中心の離れ具合の上限（2本の大きさの平均に対する割合）
    mid   … 交点が、2本それぞれの線のこの範囲（始点0〜終点1）にあること。端で触れているだけのものは除く"""
    global MARKS, P_MARK
    MARKS, base, photos = [], 0, set()
    for k, pc in enumerate(pieces):
        S  = [np.asarray(q, float) for q in pc]
        sz = [max(np.ptp(q[:, 0]), np.ptp(q[:, 1])) for q in S]
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                if sz[i] >= small or sz[j] >= small:
                    continue
                ci = (S[i].min(0) + S[i].max(0)) / 2
                cj = (S[j].min(0) + S[j].max(0)) / 2
                if np.linalg.norm(ci - cj) > near * (sz[i] + sz[j]) / 2:
                    continue
                f = _cross_frac(S[i], S[j])
                if f is None or not (mid[0] < f[0] < mid[1] and mid[0] < f[1] < mid[1]):
                    continue                            # 2本とも線の途中で、1回だけ交わる
                # 学習データの線（A32）は重心を原点にしているので、重心の位置で覚える
                gi, gj = resample(S[i], 32).mean(0), resample(S[j], 32).mean(0)
                c = (gi + gj) / 2
                MARKS.append((base + i, base + j, gi - c, gj - c))
                photos.add(k)
        base += len(S)
    P_MARK = len(photos) / max(len(pieces), 1)
    return MARKS


def save_model(path="model.pkl"):
    """学習結果を保存する"""
    with open(path, "wb") as f:
        pickle.dump(dict(A32=A32, LAB=LAB, SZ=SZ, X=X, P=P,
                         MM_W=MM_W, MM_H=MM_H, ROW_H=ROW_H, NROW=NROW,
                         version="2026-11-02"), f)
    save_marks(os.path.join(os.path.dirname(os.path.abspath(path)), "marks.pkl"))
    return path


def save_marks(path="marks.pkl"):
    """学習データの中の本物の印を、model.pkl とは別のファイルに保存する（model.pkl の中身は変えない）"""
    with open(path, "wb") as f:
        pickle.dump(dict(MARKS=MARKS, P_MARK=P_MARK, version="2026-11-18"), f)
    return path


def make_marks(pieces_path="pieces.pkl", out="marks.pkl"):
    """pieces.pkl から印だけを探して marks.pkl を作る（学習し直さずに作れる）"""
    d      = pickle.load(open(pieces_path, "rb"))
    pieces = d["pieces"] if isinstance(d, dict) else (d[0] if isinstance(d, tuple) else d)
    _find_marks(pieces)
    return save_marks(out), len(MARKS), P_MARK


def load_model(path="model.pkl"):
    """学習結果を読み込む。生成だけならこれでよい。

    読むのは学習で得たもの（線とその分類）だけ。
    設定（P・寸法）はこのファイルの上のほうに書いてあるものを使う。
    model.pkl にも古い設定が入っているが、意図的に無視している
    （入れてしまうと、このファイルの P を書き換えても効かなくなるため）。
    """
    global A32, LAB, SZ, X, MARKS, P_MARK
    d = pickle.load(open(path, "rb"))
    A32, LAB, SZ, X = d["A32"], d["LAB"], d["SZ"], d["X"]
    # 印は model.pkl の横の marks.pkl から読む。無ければ印は入らない
    mp = os.path.join(os.path.dirname(os.path.abspath(path)), "marks.pkl")
    if os.path.exists(mp):
        m = pickle.load(open(mp, "rb"))
        MARKS, P_MARK = m["MARKS"], m["P_MARK"]
    else:
        MARKS, P_MARK = [], 0.0
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


def chaikin(p, n=2):
    """角を少し落としながら点を増やす。32点の線を大きく描いても折れ線に見えないように"""
    p = np.asarray(p, float)
    for _ in range(n):
        q = np.empty((2 * len(p) - 2, 2))
        q[0::2] = 0.75 * p[:-1] + 0.25 * p[1:]
        q[1::2] = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], q, p[-1:]])
    return p


def nearest(shape):
    """学習データのいちばん近い線までの距離（重心を揃え、大きさは揃えない）"""
    c = shape - shape.mean(0)
    return float(np.sqrt(((A32 - c) ** 2).sum((1, 2))).min())


def _closed(o, thr):
    sz = max(np.ptp(o[:, 0]), np.ptp(o[:, 1]), 1e-9)
    return np.linalg.norm(o[0] - o[-1]) / sz < thr


def _align(b, q, thr):
    """q を b に重ねやすい向きにする：描く向き（逆にするか）を選び、
    両方が閉じかけの輪郭なら、描き始めの位置もずらして b に合わせる"""
    best = None
    cb = _closed(b, thr)
    for r in (q, q[::-1]):
        ks = range(len(r) - 1) if (cb and _closed(r, thr)) else [0]
        for k in ks:
            rr = r if k == 0 else np.vstack([r[k:-1], r[:k], r[k:k + 1]])
            d  = float(((rr - b) ** 2).sum())
            if best is None or d < best[0]:
                best = (d, rr)
    return best[1]


def pick_outline(i, rng, mix=3, thr=0.35):
    """大きい輪郭の線どうしを混ぜて、新しい輪郭を作る（11/20）。

    pick_shape と同じく近い線を混ぜるが、相手は大きい輪郭の線のうち、
    閉じかけなら閉じかけの線、開いているなら開いている線から選ぶ。
    混ぜる前に、描く向きと描き始めを元の線にそろえる。
    閉じた輪郭の近くには開いた線が多く、そのまま混ぜると閉じた輪郭が開いてしまい、
    学習データより「上が開いた形」が多く出ていたため。"""
    if mix <= 1:
        return A32[i].copy()
    key = (id(A32), i, mix, thr)
    if key not in _OUTC:                                   # 混ぜる相手と、そろえた形は毎回同じなので覚えておく
        bg   = np.asarray(BG)
        same = np.array([_closed(A32[j], thr) for j in bg]) == _closed(A32[i], thr)
        cand = bg[same]
        d    = ((X[cand] - X[i]) ** 2).sum(1)
        near = cand[np.argsort(d)[:mix]]
        _OUTC[key] = np.stack([_align(A32[i], A32[j], thr) for j in near])
    S = _OUTC[key]
    w = rng.dirichlet(np.ones(len(S)))
    return (S * w[:, None, None]).sum(0)


_OUTC = {}


def build_outline(rng, p):
    """大きい輪郭の線を混ぜて、1文字ぶんの輪郭を作る。

    学習データの文字まるごとの輪郭（大きさ0.5以上）から、輪郭らしいものを重みつきで選び、
    同じグループの近い線と混ぜる。平たすぎる形と、学習データに近すぎる形は引き直す。
    向きは回さない（文字は立ったまま学習データに入っているため）。
    """
    lo, hi = p["bigasp"]
    s = None
    for _ in range(40):
        i = int(rng.choice(BG, p=WOUT))
        if p.get("mixsame"):
            s = pick_outline(i, rng, p.get("mixbig", p["mix"]), p.get("samethr", 0.35))
        else:
            s = pick_shape(i, rng, p.get("mixbig", p["mix"]))
        r = np.ptp(s[:, 0]) / max(np.ptp(s[:, 1]), 1e-9)
        g = np.linalg.norm(s[0] - s[-1]) / max(np.ptp(s[:, 0]), np.ptp(s[:, 1]), 1e-9)
        if lo <= r <= hi and (p.get("bigopen") is None or g <= p["bigopen"]) and nearest(s) >= p["dmin"]:
            break
    return jitter(s, p["t"] * 0.5, rng)


def build_double(rng, p):
    """1本の線で、2文字ぶんの輪郭を続けて描く。

    輪郭を2つ作り、高さを揃え、1つめの終点に2つめの始点を繋ぐ。
    線の向き（どちらの端から描くか）は4通りあるので、2つめが1つめの右隣に来る向きを選ぶ。
    どの向きでも隣に並ばなければ None を返す（その文字は1文字ぶんの輪郭になる）。
    """
    for _ in range(8):
        a = build_outline(rng, p)
        b = build_outline(rng, p)
        a = a - a.mean(0)
        b = (b - b.mean(0)) * (np.ptp(a[:, 1]) / max(np.ptp(b[:, 1]), 1e-9))
        w = (np.ptp(a[:, 0]) + np.ptp(b[:, 0])) / 2
        h = np.ptp(a[:, 1])
        ok = []
        for a2 in (a, a[::-1]):
            for b2 in (b, b[::-1]):
                q  = b2 - b2[0] + a2[-1]                      # 終点に始点を繋ぐ
                dx = (q[:, 0].min() + q[:, 0].max()) / 2 - (a2[:, 0].min() + a2[:, 0].max()) / 2
                dy = (q[:, 1].min() + q[:, 1].max()) / 2 - (a2[:, 1].min() + a2[:, 1].max()) / 2
                if p["joinx"][0] * w <= dx <= p["joinx"][1] * w and abs(dy) <= p["joiny"] * h:
                    ok.append(np.vstack([a2, q[1:]]))
        if ok:
            return ok[int(rng.integers(len(ok)))]
    return None


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
    """文字を作り、横に lnarrow 倍に縮める（中心は動かさない）"""
    L = _make_letter(rng, p, size, maxstk)
    if len(L) == 1:
        _LAST["nsingle"] = _LAST.get("nsingle", 0) + 1
    k = p.get("lnarrow", 1.0)
    if k != 1.0:
        c = (np.vstack(L).min(0) + np.vstack(L).max(0)) / 2
        L = [np.c_[(q[:, 0] - c[0]) * k + c[0], q[:, 1]] for q in L]
    return L


def _make_letter(rng, p, size, maxstk=None):
    """文字 = 丸い線を何本か重ね、縦に立てて、高さを size に合わせたもの。

    何本重ねるかは pstk の確率で引く。文字ごとに引き直すので、
    1本だけの開いた文字と、何本も重なった密な文字が混ざる。
    """
    w    = np.asarray(p["pstk"], float)
    nst  = 1 + int(rng.choice(len(w), p=w / w.sum()))
    mn   = int(p.get("minstk", 1))
    if p.get("nrow") == 1 and p.get("minstk1") is not None:
        mn = max(mn, int(p["minstk1"]))
    if p.get("singlemax") is not None and size > p["singlemax"] * H:
        mn = max(mn, 2)                                 # 大きい文字は1本だけで作らない
    if p.get("maxsingle") is not None and _LAST.get("nsingle", 0) >= p["maxsingle"]:
        mn = max(mn, 2)                                 # 1本だけの文字はもう使い切った
    if maxstk is not None:
        nst = min(nst, int(maxstk))
    nst  = max(nst, mn)                                 # 1本だけの文字は作らない

    # a. 輪郭が「大きい輪郭の線を混ぜたもの」の文字。
    #    2本目からは小さい線を、輪郭と同じ倍率のまま（元の落書きの中での大きさの比で）中に置く。
    _LAST["nlet"] = _LAST.get("nlet", 0) + 1
    if len(BG) and rng.random() < p.get("pbig", 0.0):
        _LAST["nbig"] = _LAST.get("nbig", 0) + 1
        base = build_double(rng, p) if rng.random() < p.get("pjoin", 0.0) else None
        if base is not None:
            nst = max(nst, mn + 1)                      # 2文字ぶんなので、中の線も1本多く
        if base is None:
            base = build_outline(rng, p)
            gap  = np.linalg.norm(base[0] - base[-1]) / max(np.ptp(base[:, 0]), np.ptp(base[:, 1]), 1e-9)
            if gap <= p.get("closedgap", 0.55):
                _LAST["nclosed"] = _LAST.get("nclosed", 0) + 1
        else:
            _LAST["nclosed"] = _LAST.get("nclosed", 0) + 1
        base = base - (base.min(0) + base.max(0)) / 2
        k    = size * rng.uniform(*p["lvar"]) / max(np.ptp(base[:, 1]), 1e-9)
        L    = [chaikin(base) * k]
        bw, bh = np.ptp(L[0][:, 0]), np.ptp(L[0][:, 1])
        for _ in range(nst - 1):
            q = pick_shape(int(rng.choice(SMALL, p=WSM)), rng, p["mix"])
            q = relax(resample(q, 64), p["frac"])      # 大きくするので、小さい線の細かい揺れをならす
            q = chaikin(q - q.mean(0)) * k
            q = q - (q.min(0) + q.max(0)) / 2
            lo = p.get("binmin", 0.0) * max(bw, bh)     # 小さすぎる線は大きくする
            q  = q * max(1.0, lo / max(np.ptp(q[:, 0]), np.ptp(q[:, 1]), 1e-9))
            m = p.get("binmax", 0.6)                    # 輪郭からはみ出す大きさなら縮める
            q = q * min(1.0, m * bw / max(np.ptp(q[:, 0]), 1e-9), m * bh / max(np.ptp(q[:, 1]), 1e-9))
            c = np.array([rng.uniform(-1, 1) * p["binset"] * bw,
                          rng.uniform(-1, 1) * p["binset"] * bh])
            lim = np.maximum((np.array([bw, bh]) - np.ptp(q, axis=0)) / 2, 0.0)
            L.append(q + np.clip(c, -lim, lim))         # 輪郭の箱の中に収める
        return L

    # b. 輪郭が「小さい線を繋いだ丸い線」の文字（11/3 までの作り方）
    #    重ねる本数は pstkr で引き直す（3本重ねると団子になるので、1〜2本）
    if p.get("pstkr") is not None:
        w   = np.asarray(p["pstkr"], float)
        nst = 1 + int(rng.choice(len(w), p=w / w.sum()))
        if maxstk is not None:
            nst = min(nst, int(maxstk))
        nst = max(nst, mn)
    base = build_round(rng, p)
    base = base - base.mean(0)
    s0   = max(np.ptp(base[:, 0]), np.ptp(base[:, 1]), 1e-9)
    L    = [base]

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


def make_word(rng, p, size, target=None, budget=None, nfix=None):
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
        if nfix is not None:
            n = int(nfix)
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
    g   = p["gap"] * size * p.get("lnarrow", 1.0)    # 重なりは文字の幅に合わせて縮める
    if target is not None and len(Ls) > 1:
        need = (target - sum(wds)) / (len(Ls) - 1)
        g    = float(np.clip(need, g, p["gapmax"] * size))

    out, x0 = [], 0.0
    ct = p.get("contact")
    for L in Ls:
        a   = np.vstack(L)
        off = np.array([x0 - a[:, 0].min(), -a[:, 1].min()])
        L   = [q + off for q in L]
        if ct is not None and out:
            # 線どうしで寄せる：高さごとに「前の文字の右端」と「この文字の左端」を比べ、
            # いちばん近いところで、線が文字の高さの ct の割合だけ食い込むように置く（11/20）
            d = _touch(out[-1], L, size)
            if d is not None:
                dx = -d - rng.uniform(*ct) * size
                L  = [q + np.array([dx, 0.0]) for q in L]
        out.append(L)                      # 文字ごとにまとめて返す
        x0 = max(q[:, 0].max() for q in L) + g
    if p.get("occlude"):
        n0  = sum(len(L) for L in out)
        out = occlude(out, p)
        # 重なりで切れて増えた線の数（線の本数の条件では数えない）
        _LAST["ncut"] = _LAST.get("ncut", 0) + sum(len(L) for L in out) - n0
    return out


def _touch(A, B, size, nb=24):
    """文字A（左）と文字B（右）の、同じ高さでの「Bの左端 − Aの右端」のいちばん小さい値。
    同じ高さに両方の線がなければ None（そのときは gap の重なりで置く）。"""
    A = np.vstack(A); B = np.vstack(B)
    lo = max(A[:, 1].min(), B[:, 1].min()); hi = min(A[:, 1].max(), B[:, 1].max())
    if hi - lo < 0.1 * size:
        return None
    e  = np.linspace(lo, hi, nb + 1)
    ia = np.digitize(A[:, 1], e) - 1; ib = np.digitize(B[:, 1], e) - 1
    best = None
    for k in range(nb):
        ma, mb = ia == k, ib == k
        if ma.any() and mb.any():
            v = B[mb, 0].min() - A[ma, 0].max()
            best = v if best is None else min(best, v)
    return best


class _Region:
    """閉じた輪郭の内側かどうかを調べる（numpy だけで書いた点の内外判定）"""
    def __init__(self, o):
        self.a = np.asarray(o, float)
        self.b = np.roll(self.a, -1, axis=0)          # 終点と始点もつないで閉じる
        self.lo, self.hi = self.a.min(0), self.a.max(0)

    def contains_points(self, pts):
        pts = np.asarray(pts, float)
        res = np.zeros(len(pts), bool)
        m   = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)   # 外枠の外は外
        if not m.any():
            return res
        x, y = pts[m, 0][:, None], pts[m, 1][:, None]
        ax, ay = self.a[:, 0][None], self.a[:, 1][None]
        bx, by = self.b[:, 0][None], self.b[:, 1][None]
        cross = (ay > y) != (by > y)                   # 横向きの半直線が辺をまたぐか
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = ax + (y - ay) * (bx - ax) / (by - ay)
        res[m] = (np.sum(cross & (x < xi), axis=1) % 2) == 1
        return res


def _cut(L, front, p):
    """文字 L の線から、front（前にある文字の輪郭の内側）に入った部分を消す。
    残った線と、消えた割合（線の長さで）を返す"""
    keep, tot, kept = [], 0.0, 0.0
    for q in L:
        Lq  = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
        tot += Lq
        d   = resample(q, max(int(Lq / (0.004 * H)) + 2, len(q)))
        ins = np.zeros(len(d), bool)
        for r in front:
            ins |= r.contains_points(d)
        idx = np.flatnonzero(np.diff(np.r_[0, (~ins).astype(int), 0]))
        for s0, s1 in zip(idx[0::2], idx[1::2]):
            seg = d[s0:s1]
            ls  = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) if len(seg) > 1 else 0.0
            if ls >= p.get("occmin", 0.03) * H:
                keep.append(seg)
                kept += ls
    return keep, 1.0 - kept / max(tot, 1e-9)


def occlude(letters, p):
    """重なった所は描かない。

    前にある文字の輪郭（1本目の線）の内側に入った、後ろの文字の線を消す。
    どちらが前かは p["front"]（"left" なら左の文字が前）。
    輪郭が閉じていない文字（始点と終点が occgap より離れている）は内側を決められないので、
    その文字は他を隠さない。

    隠れるのは1文字の線の maxhide（割合）まで。それより多く隠れるときは、後ろの文字を
    前の文字から離れる向きに少しずつずらしてから重ねる（後ろに続く文字も一緒にずらす）。
    深く入り込みすぎると、後ろの文字が切れ端だけになり形が崩れるため（11/20）。"""
    letters = [list(L) for L in letters]
    order = list(range(len(letters)))
    sign  = 1.0
    if p.get("front", "left") != "left":
        order, sign = order[::-1], -1.0

    def region(L):
        o  = L[0]
        sz = max(np.ptp(o[:, 0]), np.ptp(o[:, 1]), 1e-9)
        return _Region(o) if np.linalg.norm(o[0] - o[-1]) / sz <= p.get("occgap", 0.6) else None

    out, regs = [None] * len(letters), {}
    mh = p.get("maxhide")
    for k, j in enumerate(order):
        front = [regs[i] for i in order[:k] if regs.get(i) is not None]
        if not front:
            out[j] = letters[j]
        else:
            keep, frac = _cut(letters[j], front, p)
            step = 0.04 * float(np.ptp(np.vstack(letters[j])[:, 1]))
            n = 0
            while mh is not None and frac > mh and n < 15:
                for jj in order[k:]:                      # この文字と、その後ろに続く文字をずらす
                    letters[jj] = [q + np.array([sign * step, 0.0]) for q in letters[jj]]
                keep, frac = _cut(letters[j], front, p)
                n += 1
            out[j] = keep
        regs[j] = region(letters[j])
    return [L for L in out if L]


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
    free = (want is None and asp is None and p["maxhole"] is None and p.get("maxshort") is None
            and p.get("maxcrowd") is None and p.get("maxover") is None)

    # 段数は最初に1回だけ引いて、作り直しの間は変えない。
    # 作り直すたびに引くと、条件を通りやすい段数（1段）ばかりが残ってしまう。
    p = dict(p)
    if p["nrow"] is None:
        p["nrow"] = 1 if rng.random() < p["p1row"] else 2
    # 1段のときは線の本数の範囲を nline1 に差し替える
    if p["nrow"] == 1 and p.get("nline1"):
        p["nline"] = p["nline1"]
        want       = p["nline"]
    if p.get("front") is None:                     # どちらの文字が前かは1枚に1回だけ決める
        p["front"] = "left" if rng.random() < 0.5 else "right"
    # 1段のときは、大きい輪郭の文字を主にする
    if p["nrow"] == 1 and p.get("pbig1") is not None:
        p["pbig"] = p["pbig1"]
    if p["nrow"] == 1:
        p["bigopen"] = p.get("bigopen1")
        if p.get("pjoin1") is not None:
            p["pjoin"] = p["pjoin1"]
        if p.get("nlet1") is not None:
            p["nlet"] = p["nlet1"]
        if p.get("pstk1") is not None:
            p["pstk"] = p["pstk1"]
    if p["nrow"] == 2 and p.get("fillgap2") is not None:
        p["fillgap"] = p["fillgap2"]
    if p["nrow"] == 1 and p.get("row1h"):
        # 1段を帯にするときは、絵の縦横比・中の空白・穴埋め・拡大を使わない（帯なので上下は空く）
        p["aspect"], p["maxhole"], p["fillgap"], p["cap"] = None, None, False, 1.0
        asp = None
        if p.get("nline1b"):
            p["nline"] = p["nline1b"]
            want       = p["nline"]
    best, bestd = None, None
    for _ in range(1 if free else int(p["ntry"])):
        out = _compose(rng, p)
        d   = 0.0

        if want is not None:                       # 線の本数
            n = len(out) - _LAST.get("ncut", 0)    # 重なりで切れて増えた分は数えない（切れ端の多い絵が選ばれないように）
            if n < want[0]:
                d += (want[0] - n) / max(want[0], 1)
            elif n >= want[1]:
                d += (n - want[1] + 1) / max(want[1], 1)

        if p["maxhole"] is not None:               # 絵の中に空く穴の大きさ
            h = _hole(out)
            if h > p["maxhole"]:
                # 条件を全部は満たせないとき、大きな穴がいちばん困るので重く見る
                d += 3.0 * (h - p["maxhole"]) / p["maxhole"]

        if p["nrow"] == 1 and p.get("minbig1") is not None:   # 1段：全体として文字に見えるか
            r = _LAST.get("nbig", 0) / max(_LAST.get("nlet", 0), 1)
            if r < p["minbig1"]:
                d += 1.0 * (p["minbig1"] - r)

        if p["nrow"] == 1 and p.get("minwig1") is not None:      # 1段：輪郭にこぶ・くぼみがあるか
            w = _wiggle(out, p["shortsz"] * H)
            if w < p["minwig1"]:
                d += 0.3 * (p["minwig1"] - w)

        if p["nrow"] == 1 and p.get("mincross1") is not None:    # 1段：線どうしが交わっているか
            c = _crossings(out)
            if c < p["mincross1"]:
                d += 0.3 * (p["mincross1"] - c)

        if p["nrow"] == 1 and p.get("minclosed1") is not None:   # 1段：閉じかけた輪郭があるか
            if _LAST.get("nclosed", 0) < p["minclosed1"]:
                d += 0.5 * (p["minclosed1"] - _LAST.get("nclosed", 0))

        if p["nrow"] == 1 and p.get("row1h") and p.get("minink1"):     # 1段：線が少なく寂しくないか
            ink = sum(float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()) for q in out) / H
            if ink < p["minink1"]:
                d += 0.5 * (p["minink1"] - ink)

        if p["nrow"] == 1 and p.get("row1h") and p.get("minwidth1"):   # 1段：帯が短すぎないか
            q = np.vstack(out)
            r = np.ptp(q[:, 0]) / W
            if r < p["minwidth1"]:
                d += 1.0 * (p["minwidth1"] - r)

        if p.get("maxshort") is not None:          # 短い線が多すぎないか
            ns = sum(1 for q in out
                     if max(np.ptp(q[:, 0]), np.ptp(q[:, 1])) < p["shortsz"] * H)
            if ns > p["maxshort"]:
                d += 0.3 * (ns - p["maxshort"])

        if p.get("maxover") is not None:           # 段から単語があふれていないか
            o = _LAST.get("over", 0.0)
            if o > p["maxover"]:
                d += 2.0 * (o - p["maxover"])

        if p.get("maxcrowd") is not None:          # 詰まりすぎ
            c = _crowd(out)
            if c > p["maxcrowd"]:
                d += 0.5 * (c - p["maxcrowd"])

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

    # 印は、絵を選び終わってから足す（線の本数などの条件に影響させない）
    best = add_marks(list(best), rng, p)
    return fit_to_frame(best, p["cap"], p.get("fmargin", 0.04)) if fit_frame else best


def _wiggle(g, small=0.0):
    """線の「こぶ・くぼみ」の量の合計。行って戻る曲がりを数える。

    1本の線について（曲がりの絶対量 − 正味の曲がり）÷ 180度。
    一方向に回るだけの楕円や弧は 0、途中でくぼむ・こぶがある線ほど大きい。
    small より小さい線（短い線）は数えない。"""
    tot = 0.0
    for s in g:
        if max(np.ptp(s[:, 0]), np.ptp(s[:, 1])) < small:
            continue
        a  = resample(s, 48)
        e  = np.diff(a, axis=0)
        an = np.arctan2(e[:, 1], e[:, 0])
        d  = (np.diff(an) + np.pi) % (2 * np.pi) - np.pi
        tot += (np.abs(d).sum() - abs(d.sum())) / np.pi
    return float(tot)


def _crossings(g, n=80):
    """線どうし（自分自身も含む）が交わる回数"""
    segs, ids = [], []
    for k, s in enumerate(g):
        s = np.asarray(s)
        if len(s) > n:
            s = s[np.linspace(0, len(s) - 1, n).astype(int)]
        segs.append(np.c_[s[:-1], s[1:]])
        ids.append(np.c_[np.full(len(s) - 1, k), np.arange(len(s) - 1)])
    S, I = np.vstack(segs), np.vstack(ids)
    p, r = S[:, :2], S[:, 2:] - S[:, :2]
    qp  = p[None] - p[:, None]
    den = r[:, None, 0] * r[None, :, 1] - r[:, None, 1] * r[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * r[None, :, 1] - qp[..., 1] * r[None, :, 0]) / den
        u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / den
    ok  = (np.abs(den) > 1e-12) & (t > 0) & (t < 1) & (u > 0) & (u < 1)
    ok &= ~((I[:, None, 0] == I[None, :, 0]) & (np.abs(I[:, None, 1] - I[None, :, 1]) <= 1))
    return int(np.triu(ok, 1).sum())


def _crowd(g, n=10):
    """いちばん詰まっているところで、線が何本重なっているか。
    絵の入っている範囲を n x n のマスに切り、1マスを通る線の本数の最大を返す。"""
    q  = np.vstack(g)
    lo = q.min(0)
    ex = np.maximum(q.max(0) - lo, 1e-9)
    c  = np.zeros((n, n), int)
    for s in g:
        t  = resample(s, max(len(s), 64))
        ix = np.clip(((t[:, 0] - lo[0]) / ex[0] * n).astype(int), 0, n - 1)
        iy = np.clip(((t[:, 1] - lo[1]) / ex[1] * n).astype(int), 0, n - 1)
        m  = np.zeros((n, n), bool)
        m[iy, ix] = True
        c += m
    return int(c.max())


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
        L    = make_letter(rng, p, size, maxstk=int(p.get("minstk", 1)))   # 穴埋めの文字は線の本数を最小に
        a    = np.vstack(L)
        off  = np.array([cx, cy]) - (a.min(0) + a.max(0)) / 2
        out += [q + off for q in L]
    return out


_LAST = {}      # 直前に組んだ1枚の記録（generate が条件を見るのに使う）


def _compose(rng, p):
    """1枚ぶんの線を組む（枠への収めはしない）"""
    _LAST["over"] = 0.0
    _LAST["nlet"] = 0          # 文字の数（穴埋めで足した文字も含む）
    _LAST["nbig"] = 0          # そのうち、輪郭が「大きい輪郭の線」の文字の数
    _LAST["nclosed"] = 0       # そのうち、輪郭が閉じかけている文字の数
    _LAST["nsingle"] = 0       # 1本だけでできた文字の数
    _LAST["ncut"] = 0          # 重なりで切れて増えた線の数
    nr = p["nrow"]
    rh = H / nr

    nw    = int(rng.integers(*(p["nword1"] if nr == 1 else p["nword"])))
    share = [0] * nr
    for i in range(nw):
        share[i % nr] += 1

    lz  = p["loose"]
    out = []
    # 段の帯（下から）。2段のときは高さをそろえない
    vm, sep = p["ymargin"] * H, p["rowsep"] * H
    usable  = H - 2 * vm - sep * (nr - 1)
    if nr == 2 and p.get("rowfrac"):
        f = rng.uniform(*p["rowfrac"])
        bands = [(vm, usable * f), (vm + usable * f + sep, usable * (1 - f))]
    elif nr == 1 and p.get("row1h"):
        bh0   = usable * p["row1h"]                # 1段：帯の高さを枠より低くし、上下の真ん中あたりに置く
        bands = [(vm + (usable - bh0) * rng.uniform(0.3, 0.7), bh0)]
    else:
        bands = [(vm + r * (usable / nr + sep), usable / nr) for r in range(nr)]
    prevsp = None                      # 2語に分けた前の段の（1語目の文字の割合、すきま）
    for r in range(nr):
        n   = share[r]
        row = []
        blo, bh = bands[r]
        nfix = [None] * n
        gw   = 0.0
        rsize = None
        if (nr > 1 or p.get("row1h")) and p.get("wrow"):
            # 段に入る数だけ文字を作る。段が低いほど文字が小さく、数が多くなる
            ls   = p["lsize"]
            n    = int(rng.integers(*p["wrow"])) if nr > 1 else 1   # 1段の帯は1語
            size = bh * ls
            a1, ad = (v * p.get("lnarrow", 1.0) for v in p["advance"])
            wg   = p.get("wgap", 0.0)
            gw   = float(rng.uniform(*wg)) if np.ndim(wg) else float(wg)   # この段の単語どうしのすきま
            room = (W * (1 - 2 * p["xmargin"])) / size - a1 * n - gw * (n - 1)
            tot  = int(np.floor(room / ad)) + n - int(rng.integers(0, 2))
            mw = int(p.get("minlet", 2))
            if n > 1:
                mw = int(p.get("minlet2", mw))
            if tot < mw * n and not (n > 1 and p.get("wlet2")):   # 2語に分けると1語が下限の文字数に満たないなら、1語にする
                n    = 1
                mw   = int(p.get("minlet", 2))
                room = (W * (1 - 2 * p["xmargin"])) / size - a1
                tot  = int(np.floor(room / ad)) + 1 - int(rng.integers(0, 2))
            tot  = max(tot, mw * n, 3 if nr == 1 else 0)   # 1段の帯は3文字以上
            nfix = [tot // n + (1 if k < tot % n else 0) for k in range(n)]
            if n > 1 and p.get("wlet2"):
                # 2語に分ける段：1語の文字数を wlet2 からくじで決め、段の幅に収まるまで文字を小さくする
                # （縦横の比は変えない。分けない段は大きいまま）
                # 段の文字数は wtot2 の範囲（2文字＋2文字の小さな玉が4つ並ぶのを避ける）。
                # 上下とも2語のときは、単語の切れ目の位置とすきまを前の段と変える（田の字に並ばないように）（11/20）
                for _try in range(50):
                    nfix = [int(rng.integers(p["wlet2"][0], p["wlet2"][1] + 1)) for _ in range(n)]
                    if not (p["wtot2"][0] <= sum(nfix) <= p["wtot2"][1]):
                        continue
                    if prevsp is not None and abs(nfix[0] / sum(nfix) - prevsp[0]) < p.get("wdiff", 0.0):
                        continue
                    break
                if prevsp is not None and np.ndim(wg):
                    for _try in range(50):
                        if abs(gw - prevsp[1]) >= p.get("wgdiff", 0.0):
                            break
                        gw = float(rng.uniform(*wg))
                prevsp = (nfix[0] / sum(nfix), gw)
                tot  = sum(nfix)
                need = a1 * n + ad * (tot - n) + gw * (n - 1)
                size = min(size, (W * (1 - 2 * p["xmargin"])) / need)
            rsize = size
        for i in range(n):
            # lsize は段の高さに対する割合なので、1段だと文字が2倍になってしまう。
            # 1段のときは lsize1 を使う。
            band   = nr > 1 or bool(p.get("row1h"))
            ls     = p["lsize1"] if (nr == 1 and p.get("lsize1") and not band) else p["lsize"]
            pw     = p
            if nr > 1 and p.get("all_like1"):
                # 2段の絵のどの段も、文字に見える作り方に寄せる（11/20）：
                # 小さい線をつないだ輪郭は使わない・輪郭は開きすぎない（1本だけの文字は今までどおり2文字まで）
                pw = {**p, "pbig": p.get("pbig2", 1.0), "bigopen": p.get("bigopen1")}
            if rsize is not None and n > 1 and p.get("split_like1"):
                # 2語に分けた段は文字が小さく数が多いので、1段の帯と同じく「文字に見える」作り方に寄せる（11/20）：
                # 小さい線をつないだ輪郭は使わない・輪郭は開きすぎない・1本だけの文字は作らない
                pw = {**p, "pbig": p.get("pbig2", 1.0), "bigopen": p.get("bigopen1"),
                      "minstk": max(int(p.get("minstk", 1)), int(p.get("minstk2", 2)))}
            wrd    = make_word(rng, pw, rsize if rsize is not None else (bh if band else rh) * ls, nfix=nfix[i])   # 文字のリスト。段の帯の高さに合わせる
            a      = np.vstack([q for L in wrd for q in L])
            wd, ht = np.ptp(a[:, 0]), np.ptp(a[:, 1])

            # 横：持ち場の中心はそのまま、幅だけ loose で広げる
            hw = 0.5 / n * (1 + lz * (n - 1))
            cx = (i + 0.5) / n + (2 * rng.random() - 1) * hw
            x  = float(np.clip(cx, 0.0, 1.0)) * max(W - wd, 0.0)

            # 縦の置き方
            if (nr > 1 and p["rowsplit"]) or (nr == 1 and p.get("row1h")):
                # 段ごとに帯を分け、上下が交わらないようにする。
                # 帯より高い単語は、その単語だけ縮めて帯に収める（形は変わらない）。
                # 枠いっぱいに作ると、あとで fit_to_frame が余白を作るために
                # 全体を縮めてしまう。最初から余白のぶん内側に帯を作る。
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

        # この段の単語の幅の合計が、段の幅に対してどれだけあるか
        # （単語が1つだけの段は重なる相手がなく、あふれても全体が縮むだけなので見ない）
        if len(row) > 1 and gw > 0:
            # 2語の段：決めたすきまのまま段の幅に届くよう、単語をまとめて拡大する（帯の高さまで）
            bx  = [np.vstack([q for L in w for q in L]) for w in row]
            hh  = float(np.median([np.ptp(b[:, 1]) for b in bx]))
            tot = sum(float(np.ptp(b[:, 0])) for b in bx) + gw * hh * (len(row) - 1)
            k   = min((W * (1 - 2 * p["xmargin"])) / max(tot, 1e-9),
                      bh / max(max(float(np.ptp(b[:, 1])) for b in bx), 1e-9))
            if k > 1.0:
                new = []
                for w, b in zip(row, bx):
                    o  = b.min(0)
                    w2 = [[(q - o) * k + o for q in L] for L in w]
                    ht = float(np.ptp(b[:, 1])) * k
                    y  = blo + (bh - ht) * rng.random()
                    new.append([[q + np.array([0.0, y - o[1]]) for q in L] for L in w2])
                row = new
        if len(row) > 1:
            wsum = sum(float(np.ptp(np.vstack([q for L in w for q in L])[:, 0])) for w in row)
            wsum += gw * (rsize if rsize is not None else bh * p["lsize"]) * (len(row) - 1)
            _LAST["over"] = max(_LAST["over"], wsum / (W * (1 - 2 * p["xmargin"])))

        out += _spread_x(row, {**p, "wgapr": gw}, rng)

    out = _fill_holes(out, rng, p)
    return add_dots(out, rng, p)


def add_marks(out, rng, p):
    """印を置く。学習データの本物の印（写真に描かれていた、交差した2本の小さい線）から作る。

    印を1つ選び、2本の線それぞれを同じグループの近い線と混ぜて新しい形にする（他の線と同じ作り方）。
    2本の位置関係と向きは本物の印のまま。大きさは、元の落書きの中での比のまま
    （落書き全体の高さ＝1 の単位を、今の絵の高さに合わせる）。
    他の線の近くにだけ置く（add_dots と同じ考え）。"""
    _LAST["nmark"] = 0
    pm = P_MARK if p.get("pmark") is None else p["pmark"]
    if not MARKS or not out or rng.random() >= pm:
        return out
    i, j, ci, cj = MARKS[int(rng.integers(len(MARKS)))]
    a   = np.vstack(out)
    k   = float(np.ptp(a[:, 1]))                       # 今の絵の高さ（学習データの「落書き全体の高さ＝1」に当たる）
    host = out[int(rng.integers(len(out)))]
    base = host[int(rng.integers(len(host)))]
    ang  = rng.uniform(-np.pi, np.pi)
    ctr  = base + np.array([np.cos(ang), np.sin(ang)]) * p["dnear"] * H * rng.uniform(0.3, 0.8)
    for idx, c in ((i, ci), (j, cj)):
        q = jitter(pick_shape(int(idx), rng, p["mix"]), p["t"], rng)
        out.append((q + c) * k + ctr)
    _LAST["nmark"] = 1
    return out


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
        hmin  = p.get("wgapr", 0.0) * float(np.median([np.ptp(np.vstack([q for L in w for q in L])[:, 1]) for w in words]))
        if need > 0 or hmin > 0:
            step = max(need / (len(order) - 1), hmin / max(p["xspread"], 1e-9)) if len(order) > 1 else 0.0
            x = ta
            if hmin > 0 and p.get("wgapmax") is not None:
                # すきまは文字の高さの wgapmax 倍まで。余る幅は、単語のかたまりの位置をずらして吸収する
                cap  = p["wgapmax"] * hmin / max(p.get("wgapr", 1e-9), 1e-9)
                if step * p["xspread"] > cap:
                    step = cap / max(p["xspread"], 1e-9)
                    rest = need - step * p["xspread"] * (len(order) - 1)
                    x    = ta + rest / 2                     # 余る幅は左右に半分ずつ（段の真ん中に置く）
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
            return _center_x(out, ta, tb)

    return _center_x([q for w in words for L in w for q in L], ta, tb)


def _center_x(flat, ta, tb):
    """段の中身が段の幅に届かないときは、段の真ん中に置く（片側だけが空かないように）（11/20）"""
    if not flat:
        return flat
    a = min(q[:, 0].min() for q in flat); b = max(q[:, 0].max() for q in flat)
    if b - a >= tb - ta:
        return flat
    d = np.array([(ta + tb) / 2 - (a + b) / 2, 0.0])
    return [q + d for q in flat]


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
