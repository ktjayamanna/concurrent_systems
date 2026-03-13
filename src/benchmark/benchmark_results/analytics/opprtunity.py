"""
opportunity.py — Quantifies the speculative execution opportunity from benchmark data.

Loads simple.json and complex.json, extracts per-question WorkerAgent (T_predict) vs
total execution time, and produces plots saved to the plots/ directory.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "Orchestration", "tmp")
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_questions(path: str) -> list[dict]:
    with open(path) as f:
        root = json.load(f)
    # structure: { method: { model: { questions: [...], summary: {...} } } }
    for method in root.values():
        for model in method.values():
            return model["questions"]
    return []


def extract(questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        wa = q.get("agent_time", {}).get("WorkerAgent", 0.0)
        total = q.get("time", 0.0)
        tools = q.get("called_tools", 0)
        if total > 0 and wa > 0:
            rows.append({
                "question": q["question"][:60],
                "worker_agent": wa,
                "total": total,
                "pct": wa / total * 100,
                "tools": tools,
            })
    return rows


simple_qs = extract(load_questions(os.path.join(DATA_DIR, "simple.json")))
complex_qs = extract(load_questions(os.path.join(DATA_DIR, "complex.json")))
all_qs = simple_qs + complex_qs

# ── Summary stats ─────────────────────────────────────────────────────────────
def summary(label, rows):
    pcts = [r["pct"] for r in rows]
    wa_total = sum(r["worker_agent"] for r in rows)
    time_total = sum(r["total"] for r in rows)
    print(f"\n{'─'*55}")
    print(f"  {label}  ({len(rows)} questions)")
    print(f"{'─'*55}")
    print(f"  T_predict / total (aggregate):  {wa_total/time_total*100:.1f}%")
    print(f"  T_predict / total (per-q avg):  {np.mean(pcts):.1f}%")
    print(f"  Min overhead:                   {min(pcts):.1f}%")
    print(f"  Max overhead:                   {max(pcts):.1f}%")
    print(f"  Median overhead:                {np.median(pcts):.1f}%")
    print(f"  Total T_predict saved (s):      {wa_total:.1f}s  "
          f"out of {time_total:.1f}s total")

summary("SIMPLE questions", simple_qs)
summary("COMPLEX questions", complex_qs)
summary("ALL questions", all_qs)

# ── Plot 1: sorted bar — T_predict % per question (complex set) ───────────────
sorted_complex = sorted(complex_qs, key=lambda r: r["pct"])
labels = [f"Q{i+1} ({r['tools']}T)" for i, r in enumerate(sorted_complex)]
pcts   = [r["pct"] for r in sorted_complex]
colors = ["#e07b54" if p >= 30 else "#5b8db8" for p in pcts]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(labels, pcts, color=colors, edgecolor="white", linewidth=0.4)
ax.axvline(np.mean(pcts), color="#333", linestyle="--", linewidth=1.2,
           label=f"Mean {np.mean(pcts):.1f}%")
ax.set_xlabel("T_predict as % of total execution time", fontsize=11)
ax.set_title("Speculative Execution Opportunity — Complex Questions\n"
             "(each bar = one question; label shows tool-call count)", fontsize=12)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "opportunity_per_question_complex.png"), dpi=150)
plt.close()
print("\n[saved] opportunity_per_question_complex.png")

# ── Plot 2: scatter — tool count vs T_predict % ───────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for label, rows, color, marker in [
    ("Simple", simple_qs, "#5b8db8", "o"),
    ("Complex", complex_qs, "#e07b54", "s"),
]:
    xs = [r["tools"] for r in rows]
    ys = [r["pct"] for r in rows]
    ax.scatter(xs, ys, label=label, color=color, marker=marker,
               alpha=0.7, s=55, edgecolors="white", linewidths=0.5)

ax.set_xlabel("Number of tool calls", fontsize=11)
ax.set_ylabel("T_predict % of total time", fontsize=11)
ax.set_title("More Tools → More Prediction Overhead\n"
             "(each point = one question)", fontsize=12)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "opportunity_tools_vs_overhead.png"), dpi=150)
plt.close()
print("[saved] opportunity_tools_vs_overhead.png")

# ── Plot 3: simple vs complex aggregate comparison ─────────────────────────────
datasets = {
    "Simple\n(180 Qs)": simple_qs,
    "Complex\n(41 Qs)": complex_qs,
    "Combined\n(221 Qs)": all_qs,
}
agg_pcts  = [sum(r["worker_agent"] for r in v) / sum(r["total"] for r in v) * 100
             for v in datasets.values()]
avg_pcts  = [np.mean([r["pct"] for r in v]) for v in datasets.values()]

x = np.arange(len(datasets))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
b1 = ax.bar(x - width/2, agg_pcts, width, label="Aggregate T_predict %",
            color="#5b8db8", edgecolor="white")
b2 = ax.bar(x + width/2, avg_pcts, width, label="Per-question avg T_predict %",
            color="#e07b54", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(list(datasets.keys()), fontsize=11)
ax.set_ylabel("T_predict as % of total time", fontsize=11)
ax.set_title("Prediction Overhead by Question Complexity", fontsize=12)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_ylim(0, 45)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "opportunity_simple_vs_complex.png"), dpi=150)
plt.close()
print("[saved] opportunity_simple_vs_complex.png")
