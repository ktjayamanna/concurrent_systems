#!/usr/bin/env python3
"""Concise speed report for benchmark result JSON."""
import argparse
import json
from pathlib import Path
from statistics import mean


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return float(values[f])
    return float(values[f] + (values[c] - values[f]) * (k - f))


def load_results(path: Path):
    data = json.loads(path.read_text())
    # Expect {method: {model: {questions: [...], summary: {...}}}}
    return data


def report_method_model(method, model, payload):
    questions = payload.get("questions", [])
    summary = payload.get("summary", {})
    n = summary.get("questions", len(questions)) or len(questions)

    times = [q.get("time", 0.0) for q in questions]
    server_times = [q.get("server_time", 0.0) for q in questions]

    total_time = float(summary.get("total_time", sum(times)))
    total_server_time = float(summary.get("total_server_time", sum(server_times)))

    avg_time = total_time / n if n else 0.0
    avg_server = total_server_time / n if n else 0.0

    p50_time = percentile(times, 50)
    p95_time = percentile(times, 95)
    p50_server = percentile(server_times, 50)
    p95_server = percentile(server_times, 95)

    print(f"Method: {method}")
    print(f"Model: {model}")
    print(f"Questions: {n}")
    print(f"Total time (reported): {total_time:.2f}s")
    print(f"Total server time (measured): {total_server_time:.2f}s")
    print(f"Avg time per question: {avg_time:.2f}s")
    print(f"Avg server time per question: {avg_server:.2f}s")
    print(f"p50 time: {p50_time:.2f}s | p95 time: {p95_time:.2f}s")
    print(f"p50 server: {p50_server:.2f}s | p95 server: {p95_server:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Speed report for benchmark results JSON")
    parser.add_argument(
        "path",
        nargs="?",
        default="src/benchmark/benchmark_results/Orchestration/tmp/20260312_001815.json",
        help="Path to results JSON",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    data = load_results(path)
    if not isinstance(data, dict) or not data:
        raise SystemExit("Unexpected JSON format")

    for method, models in data.items():
        if not isinstance(models, dict):
            continue
        for model, payload in models.items():
            report_method_model(method, model, payload)
            print("-")


if __name__ == "__main__":
    main()
