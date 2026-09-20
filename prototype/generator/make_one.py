"""
学習済みモデル(model.pkl)から、スローアップを1枚だけ生成して SVG を上書きする
  実行: ./run.sh  または  python3 make_one.py
  出力: ~/graffiti/output/latest.svg (300mm × 230mm)
  必要: numpy, scikit-learn
"""
import os, sys, pickle
import numpy as np

HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL     = os.path.join(HERE, "model.pkl")
OUT_DIR   = os.path.expanduser("~/graffiti/output")
OUT_NAME  = "latest.svg"
W_MM, H_MM = 300.0, 230.0     # 描画サイズ(mm)
MARGIN    = 10.0              # 余白(mm)
STROKE_MM = 1.2               # 線の太さ(mm)
TEMPERATURE = 0.5             # ばらつき(小さいほど学習データに近い)

# ---------- 生成 ----------
def sample_shape(model, k, rng):
    """グループ k の分布から形をサンプリング(カーネル密度推定)"""
    Zk = model["Z"][model["lab"] == k]
    base = Zk[rng.integers(len(Zk))]
    std = np.sqrt(model["gmm"].covariances_[k])
    z = base + rng.normal(size=len(base)) * std * TEMPERATURE
    return model["pca"].inverse_transform(z[None])[0]

def generate(model, rng):
    """マルコフ連鎖で並びを決め、各グループの分布から形を引いて配置する"""
    K = model["K"]
    n = int(rng.choice(model["counts"]))
    n = max(2, n + int(rng.integers(-2, 3)))
    state, x, paths = K, 0.0, []
    for j in range(n):
        k = int(rng.choice(K, p=model["trans"][state]))
        if j > 0:
            m, s = model["gap_stat"][state]
            x += m + rng.normal() * s * TEMPERATURE
        m, s = model["y_stat"][k]
        y = m + rng.normal() * s * TEMPERATURE
        paths.append(sample_shape(model, k, rng).reshape(-1, 2) + np.array([x, y]))
        state = k
    return paths

# ---------- 出力 ----------
def fit_to_canvas(paths):
    """生成された線を 300×230mm の中央に収める(縦横比は保つ)"""
    allp = np.vstack(paths)
    lo, hi = allp.min(0), allp.max(0)
    size = np.maximum(hi - lo, 1e-6)
    s = min((W_MM - 2 * MARGIN) / size[0], (H_MM - 2 * MARGIN) / size[1])
    off = (np.array([W_MM, H_MM]) - size * s) / 2
    return [(p - lo) * s + off for p in paths]

def to_svg(paths):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_MM}mm" height="{H_MM}mm" '
           f'viewBox="0 0 {W_MM} {H_MM}">']
    for p in paths:
        pts = " ".join(f"{x:.2f},{H_MM - y:.2f}" for x, y in p)   # SVG は y が下向き
        out.append(f'<polyline points="{pts}" fill="none" stroke="#000000" '
                   f'stroke-width="{STROKE_MM}" stroke-linecap="round" stroke-linejoin="round"/>')
    out.append("</svg>")
    return "\n".join(out)

def main():
    if not os.path.exists(MODEL):
        print(f"エラー: {MODEL} がありません", file=sys.stderr); return 1
    with open(MODEL, "rb") as f:
        model = pickle.load(f)["model"]
    rng = np.random.default_rng()
    paths = fit_to_canvas(generate(model, rng))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, OUT_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:                 # 書き込み途中のファイルを読まれないように
        f.write(to_svg(paths))
    os.replace(tmp, path)
    print(path)                               # 成功したらパスを1行出力
    return 0

if __name__ == "__main__":
    sys.exit(main())
