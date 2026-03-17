import argparse
import csv
import os
from datetime import datetime

import cv2
import numpy as np


def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def to_hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def to_lch(lab):
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    c = np.sqrt(a * a + b * b)
    h = (np.arctan2(b, a) + 2 * np.pi) % (2 * np.pi)
    h = h / (2 * np.pi)
    return l, c, h


def to_log_chroma(rgb, eps=1e-4):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.log((r + eps) / (g + eps)), np.log((b + eps) / (g + eps))


def gaussian_low(x, sigma=3.0):
    return cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)


def mae(a, b, mask=None):
    d = np.abs(a - b)
    if mask is not None:
        denom = np.maximum(mask.sum(), 1.0)
        return float((d * mask).sum() / denom)
    return float(d.mean())


def safe_var(x, mask):
    vals = x[mask > 0.5]
    if vals.size == 0:
        return 0.0
    return float(np.var(vals))


def robust_norm01(x):
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def vis_gray(x):
    y = (robust_norm01(x) * 255.0).astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)


def vis_signed(x):
    y = robust_norm01(x)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap((y * 255).astype(np.uint8), cmap)


def vis_hue(h01):
    h = (np.clip(h01, 0.0, 1.0) * 179).astype(np.uint8)
    s = np.full_like(h, 255, dtype=np.uint8)
    v = np.full_like(h, 255, dtype=np.uint8)
    hsv = np.stack([h, s, v], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def tile_grid(rows, headers, out_path):
    h, w = rows[0][0].shape[:2]
    pad = 4
    header_h = 28
    canvas_h = header_h + len(rows) * (h + pad) + pad
    canvas_w = len(headers) * (w + pad) + pad
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    for i, t in enumerate(headers):
        x = pad + i * (w + pad)
        cv2.putText(canvas, t, (x + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    for r, imgs in enumerate(rows):
        y = header_h + pad + r * (h + pad)
        for c, im in enumerate(imgs):
            x = pad + c * (w + pad)
            canvas[y:y + h, x:x + w] = im

    cv2.imwrite(out_path, canvas)


def pick_samples(input_dir, focus_prefixes=("2", "3"), n_default=1, n_focus=4):
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
    by_prefix = {}
    for f in files:
        p = f.split("_")[0]
        by_prefix.setdefault(p, []).append(f)

    selected = []
    for p in sorted(by_prefix, key=lambda x: int(x) if x.isdigit() else 999):
        k = n_focus if p in focus_prefixes else n_default
        selected.extend(by_prefix[p][: min(k, len(by_prefix[p]))])
    return selected


def resolve_gt_path(gt_dir, input_name):
    direct = os.path.join(gt_dir, input_name)
    if os.path.exists(direct):
        return direct
    prefix = input_name.split("_")[0]
    fallback = os.path.join(gt_dir, f"{prefix}_GT.png")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Cannot resolve GT for {input_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--tc", type=float, default=12.0)
    ap.add_argument("--tl", type=float, default=60.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"signal_screening_input_vs_gt_test0_proxy_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    samples = pick_samples(args.input_dir)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sample", "signal",
            "neutral_gap_input", "uniform_var_input", "uniform_var_gt", "low_mae_input_gt"
        ])

        for name in samples:
            inp = read_rgb(os.path.join(args.input_dir, name))
            gt = read_rgb(resolve_gt_path(args.gt_dir, name))

            lab_i, lab_g = to_lab(inp), to_lab(gt)
            l_g, c_g, h_g = to_lch(lab_g)
            mask = ((c_g < args.tc) & (l_g > args.tl)).astype(np.float32)

            hsv_i, hsv_g = to_hsv(inp), to_hsv(gt)
            s_i, v_i = hsv_i[..., 1], hsv_i[..., 2]
            s_g, v_g = hsv_g[..., 1], hsv_g[..., 2]

            l_i, c_i, h_i = to_lch(lab_i)
            a_i, b_i = lab_i[..., 1], lab_i[..., 2]
            a_g, b_g = lab_g[..., 1], lab_g[..., 2]

            rg_i, bg_i = to_log_chroma(inp)
            rg_g, bg_g = to_log_chroma(gt)

            abmag_i = np.sqrt(a_i * a_i + b_i * b_i)
            abmag_g = np.sqrt(a_g * a_g + b_g * b_g)
            logmag_i = np.sqrt(rg_i * rg_i + bg_i * bg_i)
            logmag_g = np.sqrt(rg_g * rg_g + bg_g * bg_g)

            mag_signals = {
                "S": (s_i, s_g),
                "C": (c_i, c_g),
                "logmag": (logmag_i, logmag_g),
                "abmag": (abmag_i, abmag_g),
            }

            for sname, (x_i, x_g) in mag_signals.items():
                mi = float(x_i[mask > 0.5].mean()) if (mask > 0.5).any() else float(x_i.mean())
                mg = float(x_g[mask > 0.5].mean()) if (mask > 0.5).any() else float(x_g.mean())
                gap_in = abs(mi - mg)
                var_in = safe_var(x_i, mask)
                var_gt = safe_var(x_g, mask)
                low_mae = mae(gaussian_low(x_i, args.sigma), gaussian_low(x_g, args.sigma))
                w.writerow([name, sname, gap_in, var_in, var_gt, low_mae])

            headers = ["input", "gt", "|in-gt|", "low_in", "low_gt", "low|in-gt|"]

            rows = []
            for xi, xg in [(s_i, s_g), (v_i, v_g)]:
                li, lg = gaussian_low(xi, args.sigma), gaussian_low(xg, args.sigma)
                rows.append([vis_gray(xi), vis_gray(xg), vis_gray(np.abs(xi - xg)), vis_gray(li), vis_gray(lg), vis_gray(np.abs(li - lg))])
            tile_grid(rows, headers, os.path.join(vis_dir, name.replace(".png", "_HSV.png")))

            rows = []
            li, lg = gaussian_low(c_i, args.sigma), gaussian_low(c_g, args.sigma)
            rows.append([vis_gray(c_i), vis_gray(c_g), vis_gray(np.abs(c_i - c_g)), vis_gray(li), vis_gray(lg), vis_gray(np.abs(li - lg))])
            dh = np.abs(h_i - h_g)
            dh = np.minimum(dh, 1.0 - dh)
            lhi, lhg = gaussian_low(h_i, args.sigma), gaussian_low(h_g, args.sigma)
            ldh = np.minimum(np.abs(lhi - lhg), 1.0 - np.abs(lhi - lhg))
            rows.append([vis_hue(h_i), vis_hue(h_g), vis_gray(dh), vis_hue(np.mod(lhi, 1.0)), vis_hue(np.mod(lhg, 1.0)), vis_gray(ldh)])
            tile_grid(rows, headers, os.path.join(vis_dir, name.replace(".png", "_LCh.png")))

            rows = []
            for xi, xg in [(rg_i, rg_g), (bg_i, bg_g)]:
                li, lg = gaussian_low(xi, args.sigma), gaussian_low(xg, args.sigma)
                rows.append([vis_signed(xi), vis_signed(xg), vis_gray(np.abs(xi - xg)), vis_signed(li), vis_signed(lg), vis_gray(np.abs(li - lg))])
            tile_grid(rows, headers, os.path.join(vis_dir, name.replace(".png", "_LOG.png")))

            rows = []
            for xi, xg in [(a_i, a_g), (b_i, b_g)]:
                li, lg = gaussian_low(xi, args.sigma), gaussian_low(xg, args.sigma)
                rows.append([vis_signed(xi), vis_signed(xg), vis_gray(np.abs(xi - xg)), vis_signed(li), vis_signed(lg), vis_gray(np.abs(li - lg))])
            tile_grid(rows, headers, os.path.join(vis_dir, name.replace(".png", "_LAB.png")))

    rows = []
    with open(metrics_path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    agg = {}
    for r in rows:
        s = r["signal"]
        agg.setdefault(s, {"n": 0, "gap": 0.0, "var_in": 0.0, "var_gt": 0.0, "low": 0.0})
        agg[s]["n"] += 1
        agg[s]["gap"] += float(r["neutral_gap_input"])
        agg[s]["var_in"] += float(r["uniform_var_input"])
        agg[s]["var_gt"] += float(r["uniform_var_gt"])
        agg[s]["low"] += float(r["low_mae_input_gt"])

    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Input-vs-GT Signal Screening Summary\n\n")
        f.write(f"samples: {len(samples)}\n")
        f.write("(test0 path not found, used CL3AN_id20/test instead.)\n\n")
        f.write("| signal | mean neutral gap (input vs gt) | mean var(input) on neutral | mean var(gt) on neutral | mean low-pass MAE |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for s in sorted(agg):
            n = max(agg[s]["n"], 1)
            f.write(f"| {s} | {agg[s]['gap']/n:.6f} | {agg[s]['var_in']/n:.6f} | {agg[s]['var_gt']/n:.6f} | {agg[s]['low']/n:.6f} |\n")

    print(f"[done] out_dir={out_dir}")
    print(f"[done] samples={len(samples)}")
    print(f"[done] metrics={metrics_path}")
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
