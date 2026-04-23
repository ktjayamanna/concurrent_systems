from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
BENCHMARK_DIR = REPO / "src" / "benchmark"
ANALYTICS_DIR = BENCHMARK_DIR / "benchmark_results" / "analytics"
OUTPUT_DIR = ANALYTICS_DIR / "plots" / "full_implementation"

sys.path.insert(0, str(REPO / "src" / "Backend"))
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(ANALYTICS_DIR))

from benchmark_results.analytics.slide_metrics import (  # noqa: E402
    all_labeled_questions,
    candidate_groups,
    evaluate_predictor,
)
from src.sage_plus_plus.predictor.algorithms import HabitPredictor  # noqa: E402


def parse_cache_sizes(value: str | None, max_size: int) -> list[int]:
    if value:
        sizes: set[int] = set()
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                sizes.update(range(int(start), int(end) + 1))
            else:
                sizes.add(int(part))
        return sorted(size for size in sizes if size > 0)
    return list(range(1, max_size + 1))


def sweep(cache_sizes: list[int]) -> dict:
    items = all_labeled_questions()
    candidates = candidate_groups(items)
    rows = []

    for cache_size in cache_sizes:
        predictor = HabitPredictor(cache_size=cache_size)
        metrics = evaluate_predictor(predictor, items, candidates, online=True)
        rows.append(
            {
                "cache_size": cache_size,
                "hits": metrics["hits"],
                "total": metrics["total"],
                "hit_rate": metrics["hit_rate"],
                "simple_hit_rate": metrics["simple"]["hit_rate"],
                "complex_hit_rate": metrics["complex"]["hit_rate"],
                "mean_runtime_ms": metrics["predictor_runtime_ms"]["mean"],
                "p95_runtime_ms": metrics["predictor_runtime_ms"]["p95"],
            }
        )

    best = max(
        rows,
        key=lambda row: (
            row["hit_rate"],
            row["simple_hit_rate"],
            row["complex_hit_rate"],
            -row["cache_size"],
        ),
    )
    return {
        "protocol": "Habit predictor online over the full benchmark with an empty initial cache; full tool-sequence exact match.",
        "total_prompts": len(items),
        "best": best,
        "rows": rows,
    }


def write_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "habit_working_set_sweep.json"
    csv_path = output_dir / "habit_working_set_sweep.csv"
    plot_path = output_dir / "habit_working_set_sweep.png"

    json_path.write_text(json.dumps(result, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    plot_sweep(result, plot_path)

    print(json.dumps(result["best"], indent=2))
    print(f"saved -> {json_path}")
    print(f"saved -> {csv_path}")
    print(f"saved -> {plot_path}")


def plot_sweep(result: dict, plot_path: Path) -> None:
    rows = result["rows"]
    best = result["best"]
    cache_sizes = [row["cache_size"] for row in rows]
    overall = [row["hit_rate"] * 100 for row in rows]
    simple = [row["simple_hit_rate"] * 100 for row in rows]
    complex_ = [row["complex_hit_rate"] * 100 for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(cache_sizes, overall, color="#2563eb", linewidth=2.2, label="Overall")
    ax.plot(cache_sizes, simple, color="#16a34a", linewidth=1.8, alpha=0.9, label="Simple")
    ax.plot(cache_sizes, complex_, color="#f97316", linewidth=1.8, alpha=0.9, label="Complex")
    ax.axvline(best["cache_size"], color="#111827", linestyle="--", linewidth=1.4, label=f"Best k={best['cache_size']}")
    ax.scatter([best["cache_size"]], [best["hit_rate"] * 100], color="#111827", zorder=5)
    ax.annotate(
        f"best: k={best['cache_size']}\n{best['hit_rate'] * 100:.1f}% overall",
        xy=(best["cache_size"], best["hit_rate"] * 100),
        xytext=(34, 18),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 1.2},
    )
    ax.set_title(
        f"Habit Predictor Working-Set Sweep\nOnline full-benchmark stream, n={result['total_prompts']}, empty initial cache",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Cache size / working set k")
    ax.set_ylabel("Full tool-sequence accuracy (%)")
    ax.set_xlim(-10, max(cache_sizes) + 10)
    ax.set_ylim(0, max(max(simple), max(overall), max(complex_)) + 8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep HabitPredictor cache sizes to find the best working set.")
    parser.add_argument(
        "--sizes",
        help="Comma-separated cache sizes and ranges, e.g. '1-250,300,400'. Defaults to 1..number_of_prompts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for JSON/CSV sweep outputs.",
    )
    args = parser.parse_args()

    max_size = len(all_labeled_questions())
    cache_sizes = parse_cache_sizes(args.sizes, max_size)
    result = sweep(cache_sizes)
    write_outputs(result, args.output_dir)


if __name__ == "__main__":
    main()
