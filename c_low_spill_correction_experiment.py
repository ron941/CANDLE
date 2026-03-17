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


def write_rgb(path, rgb):
    bgr = cv2.cvtColor((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def resolve_gt_path(gt_dir, input_name):
    direct = os.path.join(gt_dir, input_name)
    if os.path.exists(direct):
        return direct
    prefix = input_name.split("_")[0]
    fallback = os.path.join(gt_dir, f"{prefix}_GT.png")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Cannot resolve GT for {input_name}")


def pick_samples(input_dir, groups=("1", "2")):
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
    return [f for f in files if f.split("_")[0] in groups]


def blur(x, sigma=3.0):
    return cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)


def robust_norm01(x):
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def to_lab(rgb):
    return cv2.cvtColor(np.clip(rgb, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2LAB)


def to_rgb(lab):
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb, 0.0, 1.0)


def delta_e76(lab1, lab2):
    d = lab1 - lab2
    return np.sqrt(np.sum(d * d, axis=2))


def sobel_texture(rgb):
    gray = rgb.mean(axis=2).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return robust_norm01(np.sqrt(gx * gx + gy * gy))


def neutral_and_color_masks(gt_lab, tc=12.0, tl=60.0, tcolor=22.0):
    l = gt_lab[..., 0]
    a = gt_lab[..., 1]
    b = gt_lab[..., 2]
    c = np.sqrt(a * a + b * b)
    neutral = ((c < tc) & (l > tl)).astype(np.float32)
    colorful = (c > tcolor).astype(np.float32)
    return neutral, colorful


def safe_mask_mean(x, mask):
    vals = x[mask > 0.5]
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def exp1_lowpass_shrink(inp_lab, alpha, sigma):
    l = inp_lab[..., 0]
    a = inp_lab[..., 1]
    b = inp_lab[..., 2]
    a_low = blur(a, sigma)
    b_low = blur(b, sigma)
    a_high = a - a_low
    b_high = b - b_low
    a_new = (1.0 - alpha) * a_low + a_high
    b_new = (1.0 - alpha) * b_low + b_high
    out = np.stack([l, a_new, b_new], axis=2)
    return to_rgb(out)


def exp2_gated_shrink(inp_lab, inp_rgb, alpha, sigma):
    l = inp_lab[..., 0]
    a = inp_lab[..., 1]
    b = inp_lab[..., 2]
    a_low = blur(a, sigma)
    b_low = blur(b, sigma)
    c_low = np.sqrt(a_low * a_low + b_low * b_low)
    c_n = robust_norm01(c_low)
    l_n = robust_norm01(l)
    tex = sobel_texture(inp_rgb)
    gate = np.clip(c_n * l_n * (1.0 - tex), 0.0, 1.0)
    a_new = a - alpha * gate * a_low
    b_new = b - alpha * gate * b_low
    out = np.stack([l, a_new, b_new], axis=2)
    return to_rgb(out), gate, c_n


def exp3_rgb_multiplicative(inp_rgb, inp_lab, alpha, sigma):
    a_low = blur(inp_lab[..., 1], sigma)
    b_low = blur(inp_lab[..., 2], sigma)
    c_low = np.sqrt(a_low * a_low + b_low * b_low)
    c_n = robust_norm01(c_low)
    out = inp_rgb / (1.0 + alpha * c_n[..., None])
    return np.clip(out, 0.0, 1.0), c_n


def metrics(pred_rgb, gt_rgb, neutral_mask, colorful_mask):
    pred_lab = to_lab(pred_rgb)
    gt_lab = to_lab(gt_rgb)
    de = delta_e76(pred_lab, gt_lab)
    pred_c = np.sqrt(pred_lab[..., 1] ** 2 + pred_lab[..., 2] ** 2)
    gt_c = np.sqrt(gt_lab[..., 1] ** 2 + gt_lab[..., 2] ** 2)
    chroma_res = np.abs(pred_c - gt_c)
    return {
        "neutral_de": safe_mask_mean(de, neutral_mask),
        "neutral_chroma_res": safe_mask_mean(chroma_res, neutral_mask),
        "color_de": safe_mask_mean(de, colorful_mask),
    }


def heatmap(x):
    xn = robust_norm01(x)
    return cv2.applyColorMap((xn * 255).astype(np.uint8), getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))


def vis_gray(x):
    y = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--alphas", type=str, default="0.2,0.4,0.6,0.8")
    ap.add_argument("--tc", type=float, default=12.0)
    ap.add_argument("--tl", type=float, default=60.0)
    ap.add_argument("--tcolor", type=float, default=22.0)
    args = ap.parse_args()

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"c_low_correction_12_{ts}")
    vis_dir = os.path.join(out_dir, "visuals")
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    samples = pick_samples(args.input_dir, groups=("1", "2"))

    raw_metrics_path = os.path.join(out_dir, "metrics_raw.csv")
    best_metrics_path = os.path.join(out_dir, "metrics_best.csv")

    with open(raw_metrics_path, "w", newline="") as f_raw, open(best_metrics_path, "w", newline="") as f_best:
        w_raw = csv.writer(f_raw)
        w_best = csv.writer(f_best)
        w_raw.writerow(["sample", "group", "exp", "alpha", "neutral_de", "neutral_chroma_res", "color_de", "sep_after"])
        w_best.writerow(["sample", "group", "exp", "best_alpha", "neutral_de", "neutral_chroma_res", "color_de", "sep_after"])

        for name in samples:
            group = name.split("_")[0]
            inp_rgb = read_rgb(os.path.join(args.input_dir, name))
            gt_rgb = read_rgb(resolve_gt_path(args.gt_dir, name))
            inp_lab = to_lab(inp_rgb)
            gt_lab = to_lab(gt_rgb)
            neutral_mask, colorful_mask = neutral_and_color_masks(gt_lab, args.tc, args.tl, args.tcolor)

            base = metrics(inp_rgb, gt_rgb, neutral_mask, colorful_mask)
            base_sep = (base["neutral_de"] - base["color_de"])

            c_low = np.sqrt(blur(inp_lab[..., 1], args.sigma) ** 2 + blur(inp_lab[..., 2], args.sigma) ** 2)
            c_low_n = robust_norm01(c_low)

            exp_candidates = {"exp1": [], "exp2": [], "exp3": []}
            for a in alphas:
                p1 = exp1_lowpass_shrink(inp_lab, a, args.sigma)
                m1 = metrics(p1, gt_rgb, neutral_mask, colorful_mask)
                sep1 = (base["neutral_de"] - m1["neutral_de"]) - (base["color_de"] - m1["color_de"])
                w_raw.writerow([name, group, "exp1", a, m1["neutral_de"], m1["neutral_chroma_res"], m1["color_de"], sep1])
                exp_candidates["exp1"].append((a, p1, m1, sep1))

                p2, _, _ = exp2_gated_shrink(inp_lab, inp_rgb, a, args.sigma)
                m2 = metrics(p2, gt_rgb, neutral_mask, colorful_mask)
                sep2 = (base["neutral_de"] - m2["neutral_de"]) - (base["color_de"] - m2["color_de"])
                w_raw.writerow([name, group, "exp2", a, m2["neutral_de"], m2["neutral_chroma_res"], m2["color_de"], sep2])
                exp_candidates["exp2"].append((a, p2, m2, sep2))

                p3, _ = exp3_rgb_multiplicative(inp_rgb, inp_lab, a, args.sigma)
                m3 = metrics(p3, gt_rgb, neutral_mask, colorful_mask)
                sep3 = (base["neutral_de"] - m3["neutral_de"]) - (base["color_de"] - m3["color_de"])
                w_raw.writerow([name, group, "exp3", a, m3["neutral_de"], m3["neutral_chroma_res"], m3["color_de"], sep3])
                exp_candidates["exp3"].append((a, p3, m3, sep3))

            best = {}
            for exp_name in ("exp1", "exp2", "exp3"):
                # Primary criterion: lower neutral ΔE
                cand = sorted(exp_candidates[exp_name], key=lambda x: x[2]["neutral_de"])[0]
                best[exp_name] = cand
                w_best.writerow([name, group, exp_name, cand[0], cand[2]["neutral_de"], cand[2]["neutral_chroma_res"], cand[2]["color_de"], cand[3]])

            # Save best preds
            for exp_name in ("exp1", "exp2", "exp3"):
                write_rgb(os.path.join(pred_dir, f"{name[:-4]}_{exp_name}_a{best[exp_name][0]:.1f}.png"), best[exp_name][1])

            # Visual collage
            inp_bgr = cv2.cvtColor((inp_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            gt_bgr = cv2.cvtColor((gt_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            p1_bgr = cv2.cvtColor((best["exp1"][1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            p2_bgr = cv2.cvtColor((best["exp2"][1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            p3_bgr = cv2.cvtColor((best["exp3"][1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            de1 = heatmap(delta_e76(to_lab(best["exp1"][1]), gt_lab))
            de2 = heatmap(delta_e76(to_lab(best["exp2"][1]), gt_lab))
            de3 = heatmap(delta_e76(to_lab(best["exp3"][1]), gt_lab))
            cvis = heatmap(c_low_n)

            cv2.putText(p1_bgr, f"a={best['exp1'][0]:.1f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(p2_bgr, f"a={best['exp2'][0]:.1f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(p3_bgr, f"a={best['exp3'][0]:.1f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

            headers = ["input", "gt", "C_low", "exp1 best", "exp2 best", "exp3 best", "mask neutral"]
            row1 = [inp_bgr, gt_bgr, cvis, p1_bgr, p2_bgr, p3_bgr, vis_gray(neutral_mask)]
            row2 = [
                np.full_like(inp_bgr, 255),
                np.full_like(inp_bgr, 255),
                np.full_like(inp_bgr, 255),
                de1,
                de2,
                de3,
                vis_gray(colorful_mask),
            ]
            tile([row1, row2], headers, os.path.join(vis_dir, name.replace(".png", "_correction_compare.png")))

    # Aggregate summary
    rows = []
    with open(best_metrics_path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    def agg(filter_group=None):
        d = {}
        for r in rows:
            if filter_group is not None and r["group"] != filter_group:
                continue
            k = r["exp"]
            d.setdefault(k, {"n": 0, "neutral_de": 0.0, "neutral_chroma_res": 0.0, "color_de": 0.0, "sep_after": 0.0})
            d[k]["n"] += 1
            d[k]["neutral_de"] += float(r["neutral_de"])
            d[k]["neutral_chroma_res"] += float(r["neutral_chroma_res"])
            d[k]["color_de"] += float(r["color_de"])
            d[k]["sep_after"] += float(r["sep_after"])
        for k in d:
            n = max(d[k]["n"], 1)
            for m in ("neutral_de", "neutral_chroma_res", "color_de", "sep_after"):
                d[k][m] /= n
        return d

    all_agg = agg()
    g1_agg = agg("1")
    g2_agg = agg("2")

    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# C_low Spill-Map Correction Experiment (1_*, 2_*)\n\n")
        f.write(f"samples: {len(samples)}\n\n")
        f.write("## Overall (best alpha per sample, selected by neutral ΔE)\n\n")
        f.write("| exp | neutral ΔE | neutral chroma residual | color-patch ΔE | separation after correction |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for k in ("exp1", "exp2", "exp3"):
            v = all_agg.get(k, {})
            if not v:
                continue
            f.write(f"| {k} | {v['neutral_de']:.6f} | {v['neutral_chroma_res']:.6f} | {v['color_de']:.6f} | {v['sep_after']:.6f} |\n")

        f.write("\n## Group 1_*\n\n")
        f.write("| exp | neutral ΔE | neutral chroma residual | color-patch ΔE | separation after correction |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for k in ("exp1", "exp2", "exp3"):
            v = g1_agg.get(k, {})
            if not v:
                continue
            f.write(f"| {k} | {v['neutral_de']:.6f} | {v['neutral_chroma_res']:.6f} | {v['color_de']:.6f} | {v['sep_after']:.6f} |\n")

        f.write("\n## Group 2_*\n\n")
        f.write("| exp | neutral ΔE | neutral chroma residual | color-patch ΔE | separation after correction |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for k in ("exp1", "exp2", "exp3"):
            v = g2_agg.get(k, {})
            if not v:
                continue
            f.write(f"| {k} | {v['neutral_de']:.6f} | {v['neutral_chroma_res']:.6f} | {v['color_de']:.6f} | {v['sep_after']:.6f} |\n")

        def best_by(metric, mode="min"):
            vals = [(k, all_agg[k][metric]) for k in all_agg]
            if not vals:
                return "N/A"
            return (min(vals, key=lambda x: x[1]) if mode == "min" else max(vals, key=lambda x: x[1]))[0]

        f.write("\n## Auto Readout\n\n")
        f.write(f"1. Best neutral-region ΔE: **{best_by('neutral_de', 'min')}**\n")
        f.write(f"2. Best neutral chroma residual: **{best_by('neutral_chroma_res', 'min')}**\n")
        f.write(f"3. Best color-patch protection (lowest color ΔE): **{best_by('color_de', 'min')}**\n")
        f.write(f"4. Best separation-after-correction: **{best_by('sep_after', 'max')}**\n")

    print(f"[done] out_dir={out_dir}")
    print(f"[done] samples={len(samples)}")
    print(f"[done] metrics_raw={raw_metrics_path}")
    print(f"[done] metrics_best={best_metrics_path}")
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
