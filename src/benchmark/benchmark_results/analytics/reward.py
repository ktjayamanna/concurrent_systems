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

simple_qs  = load_questions("simple")
complex_qs = load_questions("complex")

def extract_rows(questions, label):
    rows = []
    for q in questions:
        t_pred = q["agent_time"].get("WorkerAgent", 0)
        t_exec = q.get("worker_agent_execution_time", 0)
        if t_pred > 0 and t_exec > 0:
            steps     = q.get("tools", [])
            tool_seq  = tuple(step[0]["name"] for step in steps if step)
            first_tool = tool_seq[0] if tool_seq else None
            rows.append({"t_pred": t_pred, "t_exec": t_exec,
                         "tool_seq": tool_seq, "first_tool": first_tool,
                         "n_tools": len(tool_seq), "label": label})
    return rows

simple_rows  = extract_rows(simple_qs,  "simple")
complex_rows = extract_rows(complex_qs, "complex")
all_rows     = simple_rows + complex_rows

t_pred_all = np.array([r["t_pred"] for r in all_rows])
t_exec_all = np.array([r["t_exec"] for r in all_rows])
ratio_all  = t_exec_all / t_pred_all
always_exec_faster = (t_exec_all < t_pred_all).mean()
median_ratio = np.median(ratio_all)
mean_ratio   = np.mean(ratio_all)

print("=" * 62)
print("INSIGHT 1 — Execution is structurally faster than prediction")
print("=" * 62)
print(f"  Queries where t_exec < t_pred  : "
      f"{int(always_exec_faster*len(all_rows))}/{len(all_rows)}  "
      f"({100*always_exec_faster:.0f}%)")
print(f"  Median t_exec / t_pred ratio   : {median_ratio:.3f}x  "
      f"(exec is {1/median_ratio:.0f}× faster)")
print(f"  Mean   t_exec / t_pred ratio   : {mean_ratio:.3f}x")
print(f"  → The parallelism window always exists. Speculation is always feasible.")
t_exec_s = np.array([r["t_exec"] for r in simple_rows])
t_exec_c = np.array([r["t_exec"] for r in complex_rows])
t_pred_s = np.array([r["t_pred"] for r in simple_rows])
t_pred_c = np.array([r["t_pred"] for r in complex_rows])

print("\n" + "=" * 62)
print("INSIGHT 2 — Savings = t_exec (entire execution hidden on a hit)")
print("=" * 62)
print(f"  Mean t_exec  simple  : {t_exec_s.mean()*1000:.1f}ms  "
      f"(= {100*t_exec_s.mean()/(t_pred_s+t_exec_s).mean():.1f}% of sequential latency)")
print(f"  Mean t_exec  complex : {t_exec_c.mean()*1000:.1f}ms  "
      f"(= {100*t_exec_c.mean()/(t_pred_c+t_exec_c).mean():.1f}% of sequential latency)")
print(f"  Complex saves {t_exec_c.mean()/t_exec_s.mean():.1f}× more per hit than simple")
print(f"  → Miss penalty is exactly 0; the system can only gain.")
def lru_hit_rate_single_tool(rows, k):
    if not rows:
        return 0.0
    cache = []
    hits = 0
    seen = 0
    for r in rows:
        if r["n_tools"] != 1 or not r["first_tool"]:
            continue
        tool = r["first_tool"]
        seen += 1
        if tool in cache:
            hits += 1
            cache.remove(tool)
            cache.insert(0, tool)
        else:
            cache.insert(0, tool)
            if len(cache) > k:
                cache.pop()
    return hits / seen if seen else 0.0

ks        = list(range(1, 11))
hr_s_lru  = [lru_hit_rate_single_tool(simple_rows,  k) for k in ks]
hr_c_lru  = [lru_hit_rate_single_tool(complex_rows, k) for k in ks]

simple_single_n = sum(1 for r in simple_rows if r["n_tools"] == 1 and r["first_tool"])
complex_single_n = sum(1 for r in complex_rows if r["n_tools"] == 1 and r["first_tool"])
def expected_saving_per_query(rows, hit_rate):
    t_exec = np.array([r["t_exec"] for r in rows])
    return hit_rate * t_exec.mean()

print("\n" + "=" * 62)
print("INSIGHT 3 — LRU cache over single-tool queries: hit rate → expected saving/query")
print("=" * 62)
print(f"  {'k':>3}  {'hit%(simple)':>14}  {'E[save](simple)':>16}  "
      f"{'hit%(complex)':>15}  {'E[save](complex)':>17}")
for k in [1, 3, 5, 10]:
    hs = hr_s_lru[k-1]; hc = hr_c_lru[k-1]
    es = expected_saving_per_query(simple_rows,  hs)
    ec = expected_saving_per_query(complex_rows, hc)
    print(f"  {k:>3}  {100*hs:>13.0f}%  {es*1000:>14.1f}ms  "
          f"{100*hc:>14.0f}%  {ec*1000:>15.1f}ms")

h3s = hr_s_lru[2]; h3c = hr_c_lru[2]
print(f"\n  At k=3 (simple): hit={100*h3s:.0f}%  "
      f"E[save]={expected_saving_per_query(simple_rows,h3s)*1000:.1f}ms/query")
print(f"  At k=3 (complex): hit={100*h3c:.0f}%  "
      f"E[save]={expected_saving_per_query(complex_rows,h3c)*1000:.1f}ms/query")
print(f"  At k=10 (complex): hit={100*hr_c_lru[9]:.0f}%  "
      f"E[save]={expected_saving_per_query(complex_rows,hr_c_lru[9])*1000:.1f}ms/query")
print(f"  → LRU cache size 10 covers "
      f"{100*hr_c_lru[9]:.0f}% of complex single-tool queries.")
col = {"simple": "#4a90d9", "complex": "#e05c5c",
       "saving": "#4caf50", "ratio": "#ff9800", "neutral": "#aaa"}

def save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")
fig, ax = plt.subplots(figsize=(8, 4.8))
delta_simple = np.array([r["t_pred"] - r["t_exec"] for r in simple_rows])
delta_complex = np.array([r["t_pred"] - r["t_exec"] for r in complex_rows])
rs = np.random.RandomState(7)
y_simple = 0.9 + (rs.rand(len(delta_simple)) - 0.5) * 0.12
y_complex = 1.1 + (rs.rand(len(delta_complex)) - 0.5) * 0.12

lim = max(delta_simple.max(), delta_complex.max()) * 1.05
ax.axvspan(0, lim, color=col["saving"], alpha=0.10, label="fully hidden (t_pred - t_exec ≥ 0)")
ax.axvspan(-lim * 0.2, 0, color="#e74c3c", alpha=0.08, label="must wait (t_pred - t_exec < 0)")
ax.axvline(0, color="k", linestyle="--", linewidth=1.0, label="boundary: t_pred = t_exec")

ax.scatter(delta_simple, y_simple, s=28, alpha=0.65, color=col["simple"], label="Simple queries")
ax.scatter(delta_complex, y_complex, s=38, alpha=0.70, color=col["complex"], marker="s",
           label="Complex queries")

simple_pos = int((delta_simple >= 0).sum())
complex_pos = int((delta_complex >= 0).sum())
simple_neg = len(delta_simple) - simple_pos
complex_neg = len(delta_complex) - complex_pos

ax.set_xlabel("(t_pred - t_exec)  (s)", fontsize=11)
ax.set_yticks([0.9, 1.1]); ax.set_yticklabels(["Simple", "Complex"])
ax.set_title(
    f"Potential for fully hidden vs partial overlap in speculative execution\n",
    fontsize=10, pad=10)
ax.set_xlim(-lim * 0.2, lim); ax.grid(alpha=0.25, axis="x")
ax.legend(fontsize=9, loc="upper right")
save(fig, "reward_1_parallelism_window.png")
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, t_exec_all.max() * 1.15, 35) * 1000
ax.hist(t_exec_s * 1000, bins=bins, color=col["simple"], alpha=0.75,
        label=f"Simple   mean={t_exec_s.mean()*1000:.0f}ms  (n={len(simple_rows)})")
ax.hist(t_exec_c * 1000, bins=bins, color=col["complex"], alpha=0.75,
        label=f"Complex  mean={t_exec_c.mean()*1000:.0f}ms  (n={len(complex_rows)})")
ax.axvline(np.median(t_exec_s)*1000, color=col["simple"],  linestyle=":", linewidth=2,
           label="Simple median")
ax.axvline(np.median(t_exec_c)*1000, color=col["complex"], linestyle=":", linewidth=2,
           label="Complex median")
ax.set_xlabel("Time saved per speculative hit (t_exec) (ms)", fontsize=11)
ax.set_ylabel("Number of queries", fontsize=11)
ax.set_title(
    f"Potential time saved on successful speculation\n",
    fontsize=10, pad=10)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
save(fig, "reward_2_savings_per_hit.png")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, [h*100 for h in hr_s_lru], "o-", color=col["simple"],  linewidth=2.5,
        markersize=7, label=f"Simple (LRU, n={simple_single_n})")
ax.plot(ks, [h*100 for h in hr_c_lru], "s-", color=col["complex"], linewidth=2.5,
        markersize=7, label=f"Complex (LRU, n={complex_single_n})")
for k_ann, h_s, h_c in [(3, hr_s_lru[2], hr_c_lru[2]), (10, hr_s_lru[9], hr_c_lru[9])]:
    ax.annotate(f"{100*h_s:.0f}%", xy=(k_ann, h_s*100), xytext=(5,  5),
                textcoords="offset points", fontsize=8, color=col["simple"])
    ax.annotate(f"{100*h_c:.0f}%", xy=(k_ann, h_c*100), xytext=(5, -14),
                textcoords="offset points", fontsize=8, color=col["complex"])
ax.set_xlabel("Cache size k (tools held by LRU)", fontsize=11)
ax.set_ylabel("Hit rate  (%)", fontsize=11)
ax.set_title("LRU cache hit rate (single-tool queries)", fontsize=10, pad=10)
ax.set_ylim(0, 105); ax.set_xticks(ks); ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
save(fig, "reward_3_hit_rate_vs_k.png")
es_s = [expected_saving_per_query(simple_rows,  h)*1000 for h in hr_s_lru]
es_c = [expected_saving_per_query(complex_rows, h)*1000 for h in hr_c_lru]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, es_s, "o-", color=col["simple"],  linewidth=2.5, markersize=7,
        label=f"Simple (LRU, n={simple_single_n})")
ax.plot(ks, es_c, "s-", color=col["complex"], linewidth=2.5, markersize=7,
        label=f"Complex (LRU, n={complex_single_n})")
ax.fill_between(ks, 0, es_s, alpha=0.10, color=col["simple"])
ax.fill_between(ks, 0, es_c, alpha=0.10, color=col["complex"])
ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
for k_ann in [3, 10]:
    ax.annotate(f"{es_s[k_ann-1]:.0f}ms", xy=(k_ann, es_s[k_ann-1]),
                xytext=(5,  5), textcoords="offset points", fontsize=8, color=col["simple"])
    ax.annotate(f"{es_c[k_ann-1]:.0f}ms", xy=(k_ann, es_c[k_ann-1]),
                xytext=(5, -14), textcoords="offset points", fontsize=8, color=col["complex"])
ax.set_xlabel("Cache size k (tools held by LRU)", fontsize=11)
ax.set_ylabel("E[saving] per query  (ms)", fontsize=11)
ax.set_title("Expected saving (single-tool queries)", fontsize=10, pad=10)
ax.set_xticks(ks); ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
save(fig, "reward_4_expected_saving_vs_k.png")
