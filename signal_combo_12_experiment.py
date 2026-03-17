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


def robust_norm01(x):
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def blur(x, sigma):
    return cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)


def log_chroma_mag(rgb, eps=1e-4):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    log_rg = np.log((r + eps) / (g + eps))
    log_bg = np.log((b + eps) / (g + eps))
    return np.sqrt(log_rg * log_rg + log_bg * log_bg)


def hue_low_from_lab(lab, sigma):
    a = lab[..., 1]
    b = lab[..., 2]
    h = np.arctan2(b, a)
    c = np.cos(h)
    s = np.sin(h)
    c_low = blur(c, sigma)
    s_low = blur(s, sigma)
    h_low = np.arctan2(s_low, c_low)  # [-pi, pi]
    h_low = (h_low + np.pi) / (2 * np.pi)  # [0,1]
    return h_low


def texture_map(rgb):
    gray = rgb.mean(axis=2).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    return robust_norm01(g)


def candidate_maps(rgb, sigma):
    lab = to_lab(rgb)
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    c = np.sqrt(a * a + b * b)
    c_low = blur(c, sigma)
    c_low_n = robust_norm01(c_low)

    logmag_low_n = robust_norm01(blur(log_chroma_mag(rgb), sigma))
    h_low_n = robust_norm01(hue_low_from_lab(lab, sigma))
    tex_n = texture_map(rgb)

    cand_a = c_low_n
    cand_b = robust_norm01(c_low_n + logmag_low_n)
    cand_c = robust_norm01(c_low_n + h_low_n)
    cand_d = robust_norm01(c_low_n * (1.0 - tex_n))

    return {
        "C_low": cand_a,
        "C_low+logmag_low": cand_b,
        "C_low+h_low": cand_c,
        "C_low*(1-texture)": cand_d,
        "_L_gt_like": l,
        "_C_gt_like": c,
    }


def vis_gray(x):
    y = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)


def vis_heat(x):
    y = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(y, cmap)


def resolve_gt(gt_dir, input_name):
    direct = os.path.join(gt_dir, input_name)
    if os.path.exists(direct):
        return direct
    prefix = input_name.split("_")[0]
    fallback = os.path.join(gt_dir, f"{prefix}_GT.png")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"GT not found for {input_name}")


def tile(rows, headers, out_path):
    h, w = rows[0][0].shape[:2]
    pad = 4
    hh = 30
    canvas = np.full((hh + pad + len(rows) * (h + pad), pad + len(headers) * (w + pad), 3), 245, np.uint8)
    for i, text in enumerate(headers):
        x = pad + i * (w + pad)
        cv2.putText(canvas, text, (x + 3, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    for r, row in enumerate(rows):
        y = hh + pad + r * (h + pad)
        for c, img in enumerate(row):
            x = pad + c * (w + pad)
            canvas[y : y + h, x : x + w] = img
    cv2.imwrite(out_path, canvas)


def pick_by_groups(input_dir, groups):
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    out = []
    for f in files:
        p = f.split("_")[0]
        if p in groups:
            out.append(f)
    return out


def mean_on_mask(x, m):
    vals = x[m > 0.5]
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--tc", type=float, default=12.0)
    ap.add_argument("--tl", type=float, default=60.0)
    ap.add_argument("--tcolor", type=float, default=22.0)
    ap.add_argument("--groups", type=str, default="1,2")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"signal_combo_12_{ts}")
    vis_dir = os.path.join(out_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    groups = tuple([g.strip() for g in args.groups.split(",") if g.strip()])
    samples = pick_by_groups(args.input_dir, groups)

    metrics_csv = os.path.join(out_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "group", "candidate", "neutral_hit", "color_leakage", "separation"])

        for name in samples:
            inp = read_rgb(os.path.join(args.input_dir, name))
            gt = read_rgb(resolve_gt(args.gt_dir, name))
            prefix = name.split("_")[0]

            in_maps = candidate_maps(inp, args.sigma)
            gt_maps = candidate_maps(gt, args.sigma)

            lab_gt = to_lab(gt)
            l_gt = lab_gt[..., 0]
            a_gt = lab_gt[..., 1]
            b_gt = lab_gt[..., 2]
            c_gt = np.sqrt(a_gt * a_gt + b_gt * b_gt)
            neutral = ((c_gt < args.tc) & (l_gt > args.tl)).astype(np.float32)
            colorful = (c_gt > args.tcolor).astype(np.float32)

            candidates = ["C_low", "C_low+logmag_low", "C_low+h_low", "C_low*(1-texture)"]
            for ckey in candidates:
                cm = in_maps[ckey]
                neutral_hit = mean_on_mask(cm, neutral)
                color_leak = mean_on_mask(cm, colorful)
                sep = neutral_hit - color_leak
                w.writerow([name, prefix, ckey, neutral_hit, color_leak, sep])

            headers = [
                "input",
                "gt",
                "C_low",
                "C+log",
                "C+h",
                "C*(1-tex)",
                "neutral_mask",
                "colorful_mask",
            ]

            row1 = [
                cv2.cvtColor((inp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                vis_heat(in_maps["C_low"]),
                vis_heat(in_maps["C_low+logmag_low"]),
                vis_heat(in_maps["C_low+h_low"]),
                vis_heat(in_maps["C_low*(1-texture)"]),
                vis_gray(neutral),
                vis_gray(colorful),
            ]
            row2 = [
                np.full_like(row1[0], 255),
                np.full_like(row1[0], 255),
                vis_heat(np.abs(in_maps["C_low"] - gt_maps["C_low"])),
                vis_heat(np.abs(in_maps["C_low+logmag_low"] - gt_maps["C_low+logmag_low"])),
                vis_heat(np.abs(in_maps["C_low+h_low"] - gt_maps["C_low+h_low"])),
                vis_heat(np.abs(in_maps["C_low*(1-texture)"] - gt_maps["C_low*(1-texture)"])),
                np.full_like(row1[0], 255),
                np.full_like(row1[0], 255),
            ]

            tile([row1, row2], headers, os.path.join(vis_dir, name.replace(".png", "_combo.png")))

    rows = []
    with open(metrics_csv, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    def aggregate(filter_group=None):
        agg = {}
        for r in rows:
            if filter_group is not None and r["group"] != filter_group:
                continue
            k = r["candidate"]
            agg.setdefault(k, {"n": 0, "neutral_hit": 0.0, "color_leakage": 0.0, "separation": 0.0})
            agg[k]["n"] += 1
            agg[k]["neutral_hit"] += float(r["neutral_hit"])
            agg[k]["color_leakage"] += float(r["color_leakage"])
            agg[k]["separation"] += float(r["separation"])
        out = {}
        for k, v in agg.items():
            n = max(v["n"], 1)
            out[k] = {m: v[m] / n for m in ("neutral_hit", "color_leakage", "separation")}
        return out

    g1 = groups[0] if len(groups) > 0 else "1"
    g2 = groups[1] if len(groups) > 1 else "2"
    all_agg = aggregate()
    g1_agg = aggregate(g1)
    g2_agg = aggregate(g2)

    summary_md = os.path.join(out_dir, "summary.md")
    with open(summary_md, "w") as f:
        group_title = ", ".join([f"`{g}_*`" for g in groups])
        f.write(f"# Combo Signal Experiment ({group_title})\n\n")
        f.write(f"samples: {len(samples)}\n\n")
        f.write("## Overall\n\n")
        f.write("| candidate | neutral_hit | color_leakage | separation |\n")
        f.write("|---|---:|---:|---:|\n")
        for k in sorted(all_agg):
            v = all_agg[k]
            f.write(f"| {k} | {v['neutral_hit']:.6f} | {v['color_leakage']:.6f} | {v['separation']:.6f} |\n")
        f.write(f"\n## Group `{g1}_*`\n\n")
        f.write("| candidate | neutral_hit | color_leakage | separation |\n")
        f.write("|---|---:|---:|---:|\n")
        for k in sorted(g1_agg):
            v = g1_agg[k]
            f.write(f"| {k} | {v['neutral_hit']:.6f} | {v['color_leakage']:.6f} | {v['separation']:.6f} |\n")
        f.write(f"\n## Group `{g2}_*`\n\n")
        f.write("| candidate | neutral_hit | color_leakage | separation |\n")
        f.write("|---|---:|---:|---:|\n")
        for k in sorted(g2_agg):
            v = g2_agg[k]
            f.write(f"| {k} | {v['neutral_hit']:.6f} | {v['color_leakage']:.6f} | {v['separation']:.6f} |\n")

        def best_sep(d):
            return max(d.items(), key=lambda kv: kv[1]["separation"])[0] if d else "N/A"

        def best_low_leak(d):
            return min(d.items(), key=lambda kv: kv[1]["color_leakage"])[0] if d else "N/A"

        f.write("\n## Auto Readout\n\n")
        f.write(f"1. Best separation overall: **{best_sep(all_agg)}**\n")
        f.write(f"2. Lowest color leakage overall: **{best_low_leak(all_agg)}**\n")
        f.write(f"3. Best separation in `{g1}_*`: **{best_sep(g1_agg)}**\n")
        f.write(f"4. Best separation in `{g2}_*`: **{best_sep(g2_agg)}**\n")

    print(f"[done] out_dir={out_dir}")
    print(f"[done] samples={len(samples)}")
    print(f"[done] visuals={vis_dir}")
    print(f"[done] metrics={metrics_csv}")
    print(f"[done] summary={summary_md}")


if __name__ == "__main__":
    main()
