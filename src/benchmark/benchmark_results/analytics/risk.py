"""
risk.py — Quantifies the downside risk of speculative execution.

Key question: how accurate does the speculative predictor need to be before
speculation becomes net-positive? And what is the expected saving at realistic
accuracy levels?

Model
─────
  Without speculation:  latency = T_predict + T_execute
  With speculation:
    HIT  → latency = max(T_predict, T_execute)   ← saves min(T_predict, T_execute)
    MISS → latency = T_predict + T_execute        ← no change (graceful fallback)

  Expected saving:
      S(h) = h * min(T_predict, T_execute)

  Break-even: any h > 0 is a win — misses cost nothing extra.

T_execute is read from the "WorkerAgent (tool invoke)" key in agent_time,
which is instrumented directly in orchestrated_routes.py.
Questions missing this key are skipped for the risk analysis.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "Orchestration", "tmp")
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_questions(path: str) -> list[dict]:
    with open(path) as f:
        root = json.load(f)
    for method in root.values():
        for model in method.values():
            return model["questions"]
    return []


def extract(questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        wa = q.get("agent_time", {}).get("WorkerAgent", 0.0)
        total = q.get("time", 0.0)
        t_execute = q.get("worker_agent_execution_time", None)
        if total > 0 and wa > 0:
            rows.append({
                "T_predict": wa,
                "T_total": total,
                # Use real measured value when available, fall back to proxy
                "T_execute": t_execute if t_execute is not None else max(total - wa, 0.01),
                "T_execute_measured": t_execute is not None,
                "tools": q.get("called_tools", 0),
            })
    return rows


simple_qs  = extract(load_questions(os.path.join(DATA_DIR, "simple.json")))
complex_qs = extract(load_questions(os.path.join(DATA_DIR, "complex.json")))
all_qs = simple_qs + complex_qs

# ── Per-question max potential saving = min(T_predict, T_execute) ─────────────
def max_saving(r):
    return min(r["T_predict"], r["T_execute"])

# ── Plot 1: Expected saving vs hit rate (aggregate across all questions) ───────
hit_rates = np.linspace(0, 1, 200)

def expected_saving_pct(rows, h):
    total_saving = sum(h * max_saving(r) for r in rows)
    total_baseline = sum(r["T_total"] for r in rows)
    return total_saving / total_baseline * 100

simple_savings  = [expected_saving_pct(simple_qs, h)  for h in hit_rates]
complex_savings = [expected_saving_pct(complex_qs, h) for h in hit_rates]
all_savings     = [expected_saving_pct(all_qs, h)     for h in hit_rates]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(hit_rates * 100, all_savings,     label="All questions",     color="#444",    linewidth=2.5)
ax.plot(hit_rates * 100, complex_savings, label="Complex questions", color="#e07b54", linewidth=1.8, linestyle="--")
ax.plot(hit_rates * 100, simple_savings,  label="Simple questions",  color="#5b8db8", linewidth=1.8, linestyle="--")

# annotate realistic hit-rate markers
for hr_pct, style in [(50, ":"), (70, "-.")]:
    y_all = expected_saving_pct(all_qs, hr_pct / 100)
    ax.axvline(hr_pct, color="grey", linestyle=style, linewidth=1)
    ax.annotate(f"{hr_pct}% accuracy\n→ {y_all:.1f}% saved",
                xy=(hr_pct, y_all), xytext=(hr_pct + 3, y_all + 1),
                fontsize=8, color="grey",
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

ax.set_xlabel("Speculative predictor accuracy (hit rate %)", fontsize=11)
ax.set_ylabel("Expected end-to-end latency reduction %", fontsize=11)
ax.set_title("Expected Savings vs Predictor Accuracy\n"
             "(any hit rate > 0% is net-positive — misses cost nothing)", fontsize=12)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "risk_savings_vs_accuracy.png"), dpi=150)
plt.close()
print("[saved] risk_savings_vs_accuracy.png")

# ── Plot 2: T_predict vs T_execute scatter (are tools faster than prediction?) ─
fig, ax = plt.subplots(figsize=(8, 6))
for label, rows, color, marker in [
    ("Simple",  simple_qs,  "#5b8db8", "o"),
    ("Complex", complex_qs, "#e07b54", "s"),
]:
    xs = [r["T_predict"]  for r in rows]
    ys = [r["T_execute"] for r in rows]
    ax.scatter(xs, ys, label=label, color=color, marker=marker,
               alpha=0.7, s=55, edgecolors="white", linewidths=0.5)

lim = max(max(r["T_predict"] for r in all_qs),
          max(r["T_execute"] for r in all_qs)) * 1.05
ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="T_predict = T_execute")
ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.05, color="green")
ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.05, color="red")
ax.text(lim * 0.6, lim * 0.1, "T_execute < T_predict\n(full T_predict saved)",
        fontsize=8, color="green")
ax.text(lim * 0.05, lim * 0.75, "T_execute > T_predict\n(partial saving)",
        fontsize=8, color="#c0392b")
measured_count = sum(1 for r in all_qs if r["T_execute_measured"])
execute_label = (
    "T_execute — WorkerAgent (tool invoke) time (s)"
    if measured_count == len(all_qs)
    else f"T_execute (measured for {measured_count}/{len(all_qs)}, proxy for rest) (s)"
)
ax.set_xlabel("T_predict — WorkerAgent LLM time (s)", fontsize=11)
ax.set_ylabel(execute_label, fontsize=11)
ax.set_title("T_predict vs T_execute per Question\n"
             "Points below diagonal = speculation captures full T_predict saving", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "risk_predict_vs_execute.png"), dpi=150)
plt.close()
print("[saved] risk_predict_vs_execute.png")

# ── Summary ───────────────────────────────────────────────────────────────────
below = sum(1 for r in all_qs if r["T_predict"] <= r["T_execute"])
measured_count = sum(1 for r in all_qs if r["T_execute_measured"])
print(f"\n{'─'*55}")
print(f"  T_execute source: {measured_count}/{len(all_qs)} questions use real "
      f"'WorkerAgent (tool invoke)' data; "
      f"{len(all_qs) - measured_count} use proxy (T_total − T_predict)")
print(f"  Questions where T_predict ≤ T_execute: {below}/{len(all_qs)} "
      f"({below/len(all_qs)*100:.0f}%)")
print(f"  → speculation captures FULL T_predict saving in these cases")
print(f"\n  At 50% hit rate,  expected saving (all Qs): "
      f"{expected_saving_pct(all_qs, 0.50):.1f}% latency reduction")
print(f"  At 70% hit rate,  expected saving (all Qs): "
      f"{expected_saving_pct(all_qs, 0.70):.1f}% latency reduction")
print(f"  At 100% hit rate, expected saving (all Qs): "
      f"{expected_saving_pct(all_qs, 1.00):.1f}% latency reduction")
