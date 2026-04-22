import json
import json.decoder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from pathlib import Path

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent / "Orchestration" / "tmp"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
def load_json_first(path):
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj

def load_questions(name):
    return load_json_first(DATA_DIR / f"{name}.json")["self-orchestrated"]["openai/gpt-4o-mini"]["questions"]

def extract_rows(questions, label):
    rows = []
    for q in questions:
        t_pred = q["agent_time"].get("WorkerAgent", 0)
        t_exec = q.get("worker_agent_execution_time", 0)
        if t_pred > 0 and t_exec > 0:
            steps = q.get("tools", [])
            tool_seq   = tuple(step[0]["name"] for step in steps if step)
            first_tool = tool_seq[0] if tool_seq else None
            rows.append({"t_pred": t_pred, "t_exec": t_exec,
                         "tool_seq": tool_seq, "first_tool": first_tool,
                         "n_tools": len(tool_seq), "label": label})
    return rows

simple_rows  = extract_rows(load_questions("simple"),  "simple")
complex_rows = extract_rows(load_questions("complex"), "complex")
all_rows     = simple_rows + complex_rows

t_pred_s = np.array([r["t_pred"] for r in simple_rows])
t_exec_s = np.array([r["t_exec"] for r in simple_rows])
t_pred_c = np.array([r["t_pred"] for r in complex_rows])
t_exec_c = np.array([r["t_exec"] for r in complex_rows])

def first_tool_hit_rate(rows, k):
    counts = Counter(r["first_tool"] for r in rows if r["first_tool"])
    top_k  = {tool for tool, _ in counts.most_common(k)}
    return sum(1 for r in rows if r["first_tool"] in top_k) / len(rows)

t_exec_s_mean = t_exec_s.mean()
t_exec_c_mean = t_exec_c.mean()
col = {"simple": "#4a90d9", "complex": "#e05c5c",
       "safe": "#4caf50", "warn": "#ff9800", "neutral": "#aaa"}

def save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")

ks = list(range(1, 11))
es_s = [first_tool_hit_rate(simple_rows,  k) * t_exec_s_mean * 1000 for k in ks]
es_c = [first_tool_hit_rate(complex_rows, k) * t_exec_c_mean * 1000 for k in ks]
scenarios = [0.0, 0.25, 0.50, 0.75, 1.00]
labels_sc = ["0%", "25%", "50%", "75%", "100%"]

t_pred_s_mean = t_pred_s.mean() * 1000
t_pred_c_mean = t_pred_c.mean() * 1000
t_exec_s_mean_ms = t_exec_s_mean * 1000
t_exec_c_mean_ms = t_exec_c_mean * 1000

baseline_s = t_pred_s_mean + t_exec_s_mean_ms
baseline_c = t_pred_c_mean + t_exec_c_mean_ms
spec_pct_s = [(t_pred_s_mean + (1-h)*t_exec_s_mean_ms) / baseline_s * 100 for h in scenarios]
spec_pct_c = [(t_pred_c_mean + (1-h)*t_exec_c_mean_ms) / baseline_c * 100 for h in scenarios]
save_pct_s = [100 - p for p in spec_pct_s]
save_pct_c = [100 - p for p in spec_pct_c]

x, w = np.arange(len(scenarios)), 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
fig.suptitle(
    "How much of the critical path (t_pred + t_exec = 100%) does speculation recover per query on average?",
    fontsize=12, fontweight="bold")

for ax, sp_pct, sv_pct, t_p, t_e, baseline, title, c in [
        (axes[0], spec_pct_s, save_pct_s, t_pred_s_mean, t_exec_s_mean_ms,
         baseline_s, "Simple queries", col["simple"]),
        (axes[1], spec_pct_c, save_pct_c, t_pred_c_mean, t_exec_c_mean_ms,
         baseline_c, "Complex queries", col["complex"])]:
    t_pred_pct  = t_p / baseline * 100
    t_exec_full = t_e / baseline * 100

    for i, (sp, sv, h) in enumerate(zip(sp_pct, sv_pct, scenarios)):
        exec_remaining = t_exec_full * (1 - h)
        exec_saved     = t_exec_full * h
        ax.bar(i, t_pred_pct,      color=c,            alpha=0.85, width=0.6)
        ax.bar(i, exec_remaining,  color="#888888",     alpha=0.70, width=0.6,
               bottom=t_pred_pct)
        ax.bar(i, exec_saved,      color=col["safe"],   alpha=0.90, width=0.6,
               bottom=t_pred_pct + exec_remaining)
        if exec_saved > 0.05:
            ax.text(i, t_pred_pct + exec_remaining + exec_saved / 2 + 0.1,
                    f"{sv:.1f}%\n({sv/100*baseline:.0f}ms)",
                    ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="darkgreen")
        ax.text(i, min(sp - 0.05, 100.4), f"{sp:.1f}%", ha="center", va="bottom",
                fontsize=8, color="black")

    ax.axhline(100, color="red", linestyle="--", linewidth=1.5,
               label=f"Baseline 100% = {baseline:.0f}ms")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([f"{l}\nhit rate" for l in labels_sc], fontsize=10)
    ax.set_ylabel("% of baseline critical path (zoomed)", fontsize=10)
    ax.set_ylim(96.5, 100.6)
    ax.set_yticks(np.arange(96.5, 100.7, 0.25))
    ax.set_title(
        f"{title}\n"
        f"t_pred = {t_p:.0f}ms ({t_p/baseline*100:.0f}%)  ·  "
        f"t_exec = {t_e:.0f}ms ({t_e/baseline*100:.0f}%)\n"
        f"Max saving at 100% hit = {t_e/baseline*100:.1f}%",
        fontsize=10)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=c,           alpha=0.85, label=f"t_pred  ({t_p:.0f}ms, always on critical path)"),
        Patch(facecolor="#888888",   alpha=0.70, label="t_exec remaining  (speculation miss)"),
        Patch(facecolor=col["safe"], alpha=0.90, label="t_exec saved  (speculation hit)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.grid(alpha=0.2, axis="y")

plt.tight_layout()
save(fig, "risk_1_scenario_model.png")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle(
    "5X faster speculative predictor finishes before SAGE's tool predictor comfortably",
    fontsize=12, fontweight="bold")

for ax, rows_mean_p, rows_mean_e, label, c in [
        (axes[0], t_pred_s.mean()*1000, t_exec_s.mean()*1000, "Simple queries", col["simple"]),
        (axes[1], t_pred_c.mean()*1000, t_exec_c.mean()*1000, "Complex queries", col["complex"])]:

    t_p  = rows_mean_p
    t_e  = rows_mean_e
    speed_labels  = ["t_pred_spec = t_pred/5\n(5× faster)", "t_pred/10\n(10× faster)",
                     "t_pred/20\n(20× faster)", "t_pred/50\n(50× faster)"]
    speed_fracs   = [1/5, 1/10, 1/20, 1/50]
    y_positions   = [3, 2, 1, 0]

    bar_height = 0.55
    ax.barh(4, t_p, left=0, height=bar_height, color=c, alpha=0.85,
            label=f"t_pred = {t_p:.0f}ms  (current tool predictor, always runs)")
    ax.barh(4, t_e, left=t_p, height=bar_height, color="#888888", alpha=0.75,
            label=f"t_exec = {t_e:.0f}ms  (current tool execution)")
    ax.text(t_p/2, 4, f"t_pred\n{t_p:.0f}ms", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")
    ax.text(t_p + t_e/2, 4, f"t_exec\n{t_e:.0f}ms", ha="center", va="center",
            fontsize=8, color="black", fontweight="bold")
    for y, frac, slabel in zip(y_positions, speed_fracs, speed_labels):
        t_ps = t_p * frac
        spec_done_at = t_ps + t_e
        ax.barh(y, t_p, left=0, height=bar_height, color=c, alpha=0.30)
        ax.barh(y, t_ps, left=0, height=bar_height, color="#ff9800", alpha=0.85,
                label="t_pred_spec  (speculative predictor)" if y == y_positions[0] else "")
        ax.barh(y, t_e, left=t_ps, height=bar_height, color=col["safe"], alpha=0.85,
                label="t_exec  (speculative execution)" if y == y_positions[0] else "")
        gap = t_p - spec_done_at
        ax.annotate(
            f"done at {spec_done_at:.0f}ms\n← gap: {gap:.0f}ms →",
            xy=(spec_done_at, y), xytext=(spec_done_at + gap*0.1, y + 0.35),
            fontsize=7.5, color="darkgreen",
            arrowprops=dict(arrowstyle="-", color="darkgreen", linewidth=0.8))
    ax.axvline(t_p, color="red", linestyle="--", linewidth=1.5,
               label=f"current tool predictor confirms at {t_p:.0f}ms → return if hit")

    yticks = [4] + list(reversed(y_positions))
    ylabels = ["Current\nsequential"] + list(reversed(speed_labels))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("time (ms)", fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.set_xlim(0, (t_p + t_e) * 1.05)
    ax.grid(alpha=0.2, axis="x")
    ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
save(fig, "risk_2_timeline.png")
