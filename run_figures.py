"""Report figures from the saved JSON metrics, emitted as pure-Python SVG.

No matplotlib — the Windows Agg renderer segfaults on this box, and SVG is just
text so it is 100% crash-proof and vector-sharp for the report. Reads the fresh
eval/loyo/loso JSONs. Usage:  python run_figures.py [config_name]
"""
import os, sys, json
os.chdir("C:/Users/mark/Desktop/internship")
name = sys.argv[1] if len(sys.argv) > 1 else "local_cds"
FIG = "figures"; os.makedirs(FIG, exist_ok=True)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def bar_svg(path, labels, values, title, ylabel, colors=None, hline=None, hlabel=""):
    W, H = 680, 380
    ml, mr, mt, mb = 70, 24, 50, 50
    pw, ph = W - ml - mr, H - mt - mb
    vmax = max(values + ([hline] if hline else []) + [1e-6]) * 1.15
    n = len(values); gap = pw / n
    bw = gap * 0.62
    colors = colors or ["#7c4dff"] * n
    px = []
    for i, v in enumerate(values):
        x = ml + gap * i + (gap - bw) / 2
        bh = ph * (v / vmax)
        y = mt + ph - bh
        px.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                  f'fill="{colors[i]}" rx="3"/>')
        px.append(f'<text x="{x + bw/2:.1f}" y="{y - 6:.1f}" font-size="13" '
                  f'text-anchor="middle" fill="#222">{v:.1f}</text>')
        px.append(f'<text x="{x + bw/2:.1f}" y="{mt + ph + 20:.1f}" font-size="13" '
                  f'text-anchor="middle" fill="#444">{labels[i]}</text>')
    # y axis ticks (0 and vmax)
    axis = [f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#999"/>',
            f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#999"/>',
            f'<text x="{ml-8}" y="{mt+ph+4}" font-size="11" text-anchor="end" fill="#666">0</text>',
            f'<text x="{ml-8}" y="{mt+6}" font-size="11" text-anchor="end" fill="#666">{vmax:.0f}</text>']
    hl = ""
    if hline:
        hy = mt + ph - ph * (hline / vmax)
        hl = (f'<line x1="{ml}" y1="{hy:.1f}" x2="{ml+pw}" y2="{hy:.1f}" '
              f'stroke="#e53935" stroke-dasharray="6 4"/>'
              f'<text x="{ml+pw}" y="{hy-5:.1f}" font-size="12" text-anchor="end" '
              f'fill="#e53935">{hlabel}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'font-family="Segoe UI,Arial">'
           f'<rect width="{W}" height="{H}" fill="white"/>'
           f'<text x="{W/2}" y="26" font-size="16" font-weight="bold" '
           f'text-anchor="middle" fill="#111">{title}</text>'
           f'<text x="18" y="{mt+ph/2}" font-size="12" fill="#666" '
           f'transform="rotate(-90 18 {mt+ph/2})" text-anchor="middle">{ylabel}</text>'
           + "".join(axis) + hl + "".join(px) + '</svg>')
    open(path, "w", encoding="utf-8").write(svg)
    print(f"saved {path}", flush=True)


loyo = load(f"models/loyo_{name}.json")
loso = load(f"models/loso_{name}.json")
ev = load(f"models/eval_{name}.json")

if loyo:
    f = loyo["folds"]
    bar_svg(f"{FIG}/{name}_loyo.svg",
            [str(r["year"]) for r in f], [r["MAE"] for r in f],
            f"{name} — LOYO held-out MAE (2023 = the wall)", "MAE (µg/m³)",
            colors=["#e53935" if r["year"] == 2023 else "#7c4dff" for r in f],
            hline=loyo["mean_MAE"], hlabel=f"mean {loyo['mean_MAE']:.1f}")

if loso:
    f = loso["folds"]
    bar_svg(f"{FIG}/{name}_loso.svg",
            [f"fold{r['fold']}" for r in f], [r["MAE"] for r in f],
            f"{name} — LOSO held-out-station MAE (spatial: solved)", "MAE (µg/m³)",
            colors=["#43a047"] * len(f),
            hline=loso["mean_MAE"], hlabel=f"mean {loso['mean_MAE']:.1f}")

if ev and ev.get("per_year"):
    py = ev["per_year"]
    bar_svg(f"{FIG}/{name}_year_bias.svg",
            [str(r["year"]) for r in py], [abs(r["bias"]) for r in py],
            f"{name} — per-year |bias| (W2 gate ≤25)", "|bias| (µg/m³)",
            colors=["#e53935" if abs(r["bias"]) > 25 else "#1e88e5" for r in py],
            hline=25, hlabel="gate 25")

print("FIGURES DONE (SVG)", flush=True)
