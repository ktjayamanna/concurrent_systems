import collections
import json
import json.decoder
import random
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
REPO = ROOT.parents[3]
DATA_DIR = ROOT.parent / "Orchestration" / "tmp"
PLOTS_DIR = ROOT / "plots"
FULL_IMPLEMENTATION_DIR = PLOTS_DIR / "full_implementation"
FULL_IMPLEMENTATION_DIR.mkdir(parents=True, exist_ok=True)
HABIT_CACHE_SIZE = 1

sys.path.insert(0, str(REPO / "src" / "Backend"))
sys.path.insert(0, str(REPO / "src" / "benchmark"))

from question_sets.simple import simple_questions
from question_sets.complex import complex_questions
from src.sage_plus_plus.predictor.algorithms import (
    HabitPredictor,
    NaiveBayesPredictor,
    PredictedToolCall,
    SmallLLMPredictor,
)


SMART_TOOLS = {
    "GetRoomName", "GetRoomId", "GetRooms", "GetRoomIds", "GetRoomNames",
    "CheckAvailability", "BookRoom", "GetDesks", "IsFree", "BookDesk",
    "RunFullSystemCheck", "CheckDeviceHealth", "GetSystemUptime",
    "ListActiveDevices", "GetDeviceId", "GetLastMaintenanceDate",
    "ScheduleMaintenance", "CheckNetworkStatus", "RestartDevice",
    "GenerateReport", "GetFridgeContents", "ReserveFridgeSpace",
    "ScheduleCleaning", "AddToGroceryList", "SetLightIntensity",
    "TurnOnLights", "TurnOffLights", "CheckSensorBattery", "GetCompleteInfo",
}
WAREHOUSE_TOOLS = {
    "GetItemLocation", "GetWarehouseEmail", "MakeOrder", "AddOrder",
    "MoveToZone", "PickupItem", "DropItem", "GetWarehouseZoneSizes",
    "GetWarehouseZoneSize", "GetInventory", "GetZones", "AddItemToZone",
    "RemoveItemFromZone",
}
MUSIC_TOOLS = {
    "PlayTrack", "PauseTrack", "SkipToNextTrack", "SkipToPreviousTrack",
    "GetCurrentVolume", "IncreaseVolume", "DecreaseVolume", "AdjustVolume",
    "Mute", "GetTrackIds", "GetTracks", "GetIdByTrack", "GetTrackById",
    "CreatePlaylist", "CreateMultiplePlaylists", "AddSongToPlaylist",
    "RenamePlaylist", "GetPlaylistNames", "GetPlaylistSongs", "GetPlaylists",
    "GetPlaylistId", "RemoveSongFromPlaylist", "DeletePlaylist",
    "FollowArtist", "UnfollowArtist", "LikeTrack",
}


def load_json_first(path: Path):
    text = path.read_text()
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj


def load_questions(name: str):
    return load_json_first(DATA_DIR / f"{name}.json")["self-orchestrated"]["openai/gpt-4o-mini"]["questions"]


def extract_rows(questions, label):
    rows = []
    for q in questions:
        t_pred = q["agent_time"].get("WorkerAgent", 0)
        t_exec = q.get("worker_agent_execution_time", 0)
        if t_pred <= 0 or t_exec <= 0:
            continue
        steps = q.get("tools", [])
        tool_seq = tuple(step[0]["name"] for step in steps if step)
        rows.append(
            {
                "label": label,
                "t_pred": t_pred,
                "t_exec": t_exec,
                "first_tool": tool_seq[0] if tool_seq else "",
                "n_tools": len(tool_seq),
            }
        )
    return rows


def expected_calls(question):
    return tuple(
        PredictedToolCall(tool.name, {param.key: param.value for param in getattr(tool, "args", [])})
        for tool in question.get("tools", [])
        if not getattr(tool, "optional", False)
    )


def expected_tool_sequence(question):
    return tuple(call.name for call in expected_calls(question))


def first_expected_args(question):
    calls = expected_calls(question)
    if not calls:
        return {}
    return calls[0].args


def all_labeled_questions():
    return [
        ("simple", question)
        for question in simple_questions
        if expected_tool_sequence(question)
    ] + [
        ("complex", question)
        for question in complex_questions
        if expected_tool_sequence(question)
    ]


def tool_family(tool_name):
    if tool_name in SMART_TOOLS:
        return "smart"
    if tool_name in WAREHOUSE_TOOLS:
        return "warehouse"
    if tool_name in MUSIC_TOOLS:
        return "music"
    return "other"


def stratified_split(items, test_fraction=0.30, seed=7):
    rng = random.Random(seed)
    by_label = collections.defaultdict(lambda: collections.defaultdict(list))
    train = []
    test = []
    for item in items:
        by_label[expected_tool_sequence(item[1])][item[1]["input"]].append(item)
    for _label, by_input in by_label.items():
        groups = list(by_input.values())
        rng.shuffle(groups)
        if len(groups) == 1:
            train.extend(groups[0])
            continue
        test_count = max(1, round(len(groups) * test_fraction))
        for group in groups[:test_count]:
            test.extend(group)
        for group in groups[test_count:]:
            train.extend(group)
    return train, test


def split_identifier(value):
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)


def training_rows(items, add_label_text=False):
    rows = [
        (question["input"], expected_calls(question))
        for _split, question in items
    ]
    if not add_label_text:
        return rows
    for sequence in sorted({expected_tool_sequence(question) for _split, question in items}):
        if not sequence:
            continue
        label = sequence[0]
        words = split_identifier(label)
        calls = tuple(PredictedToolCall(name, {}) for name in sequence)
        rows.extend([(words, calls), (f"use {words}", calls), (f"call {words}", calls)])
    return rows


def candidate_groups(items):
    labels = sorted({tool for _split, question in items for tool in expected_tool_sequence(question)})
    groups = collections.defaultdict(list)
    for label in labels:
        groups[tool_family(label)].append(f"Agent--{label}")
    return groups


def evaluate_predictor(predictor, test_items, candidates_by_family, online=False):
    hits = 0
    total = 0
    by_split = collections.Counter()
    hits_by_split = collections.Counter()
    latencies_ms = []
    for split, question in test_items:
        actual_sequence = expected_tool_sequence(question)
        first_actual = actual_sequence[0]
        predictor.set_candidate_tools(candidates_by_family[tool_family(first_actual)])
        start = time.perf_counter_ns()
        prediction = predictor.predict_call(question["input"])
        predicted_sequence = tuple(call.name.split("--")[-1] for call in prediction.calls)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        correct = predicted_sequence == actual_sequence
        latencies_ms.append(elapsed_ms)
        hits += correct
        total += 1
        by_split[split] += 1
        hits_by_split[split] += correct
        if online:
            predictor.update_calls(question["input"], [(call.name, call.args) for call in expected_calls(question)])
    latency_array = np.array(latencies_ms) if latencies_ms else np.array([0])
    return {
        "hits": hits,
        "total": total,
        "hit_rate": round(hits / total, 4) if total else 0,
        "simple": split_metric(hits_by_split, by_split, "simple"),
        "complex": split_metric(hits_by_split, by_split, "complex"),
        "predictor_runtime_ms": {
            "mean": round(float(latency_array.mean()), 4),
            "median": round(float(np.median(latency_array)), 4),
            "p95": round(float(np.percentile(latency_array, 95)), 4),
            "max": round(float(latency_array.max()), 4),
        },
    }


def split_metric(hits_by_split, by_split, name):
    total = by_split[name]
    hits = hits_by_split[name]
    return {"hits": hits, "total": total, "hit_rate": round(hits / total, 4) if total else 0}


def timing_summary(rows):
    t_pred = np.array([row["t_pred"] for row in rows])
    t_exec = np.array([row["t_exec"] for row in rows])
    baseline = t_pred.mean() + t_exec.mean()
    return {
        "queries": len(rows),
        "mean_t_pred_ms": round(t_pred.mean() * 1000, 1),
        "mean_t_exec_ms": round(t_exec.mean() * 1000, 1),
        "max_saving_percent_at_100_hit": round(t_exec.mean() / baseline * 100, 2),
        "fully_hideable_percent": round(float((t_exec < t_pred).mean() * 100), 1),
    }


def overall_tool_selection_ms(simple_rows, complex_rows):
    values = [row["t_pred"] * 1000 for row in [*simple_rows, *complex_rows]]
    array = np.array(values)
    return {
        "mean": round(float(array.mean()), 1),
        "median": round(float(np.median(array)), 1),
        "p95": round(float(np.percentile(array, 95)), 1),
        "max": round(float(array.max()), 1),
    }


def real_savings(timing, metric):
    simple_ms = timing["simple"]["mean_t_exec_ms"] * metric["simple"]["hit_rate"]
    complex_ms = timing["complex"]["mean_t_exec_ms"] * metric["complex"]["hit_rate"]
    simple_total = metric["simple"]["total"]
    complex_total = metric["complex"]["total"]
    total = simple_total + complex_total
    overall_ms = ((simple_ms * simple_total) + (complex_ms * complex_total)) / total if total else 0
    return {
        "overall_ms": round(overall_ms, 1),
        "simple_ms": round(simple_ms, 1),
        "complex_ms": round(complex_ms, 1),
    }


def main():
    simple_rows = extract_rows(load_questions("simple"), "simple")
    complex_rows = extract_rows(load_questions("complex"), "complex")
    timing = {"simple": timing_summary(simple_rows), "complex": timing_summary(complex_rows)}
    sage_tool_selection_ms = overall_tool_selection_ms(simple_rows, complex_rows)

    labeled = all_labeled_questions()
    train, test = stratified_split(labeled)
    candidates = candidate_groups(labeled)

    habit = HabitPredictor(cache_size=HABIT_CACHE_SIZE)
    naive_bayes = NaiveBayesPredictor(training_data=training_rows(train, add_label_text=True))
    small_llm = SmallLLMPredictor(training_data=[])
    few_shot_small_llm = SmallLLMPredictor(training_data=training_rows(train), use_transformers=False)

    habit_metrics = evaluate_predictor(habit, labeled, candidates, online=True)
    naive_bayes_metrics = evaluate_predictor(naive_bayes, test, candidates)
    small_llm_metrics = evaluate_predictor(small_llm, labeled, candidates)
    few_shot_small_llm_metrics = evaluate_predictor(few_shot_small_llm, test, candidates)

    metrics = {
        "methodology": {
            "split": "Naive Bayes only: stratified 70/30 by expected full tool-call sequence; singleton labels stay in train; duplicate prompt texts stay in one split",
            "seed": 7,
            "total_prompts": len(labeled),
            "total_simple_prompts": sum(1 for split, _question in labeled if split == "simple"),
            "total_complex_prompts": sum(1 for split, _question in labeled if split == "complex"),
            "naive_bayes_train_prompts": len(train),
            "naive_bayes_test_prompts": len(test),
            "naive_bayes_simple_test_prompts": sum(1 for split, _question in test if split == "simple"),
            "naive_bayes_complex_test_prompts": sum(1 for split, _question in test if split == "complex"),
            "habit_eval_prompts": len(labeled),
            "small_llm_eval_prompts": len(labeled),
            "few_shot_small_llm_train_prompts": len(train),
            "few_shot_small_llm_test_prompts": len(test),
            "candidate_scope": "worker-agent/tool-family candidate set, matching SAGE orchestration after agent routing",
            "label": "full expected benchmark tool-call sequence",
            "habit_protocol": "online over the full benchmark with an empty initial cache",
            "small_llm_backend": "hf_zero_shot"
            if getattr(small_llm, "_pipeline", None) is not None
            else "local_semantic_scorer",
            "small_llm_protocol": "zero-shot/local semantic scorer over the full benchmark; no supervised prompt training examples",
            "few_shot_small_llm_protocol": "local semantic scorer with train-split examples as few-shot/retrieval demonstrations; evaluated on held-out split",
        },
        "timing": timing,
        "sage_openai_tool_selection_runtime_ms": sage_tool_selection_ms,
        "predictor_holdout_accuracy": {
            f"habit_k{HABIT_CACHE_SIZE}": habit_metrics,
            "naive_bayes": naive_bayes_metrics,
            "small_llm": small_llm_metrics,
            "few_shot_small_llm": few_shot_small_llm_metrics,
        },
    }
    metrics["real_savings_from_holdout"] = {
        name: real_savings(timing, value)
        for name, value in metrics["predictor_holdout_accuracy"].items()
    }
    metrics["slide_takeaway"] = {
        "latency": "100% correct prediction saves about 77ms/simple and 229ms/complex, or 2.5% and 3.2% of the measured critical path.",
        "ml": (
            "Full-sequence accuracy is reported with predictor-appropriate protocols: Naive Bayes uses a held-out split, "
            "SmallLLM few-shot uses the same held-out split, while Habit and zero-shot SmallLLM are evaluated over the full benchmark."
        ),
        "safety": "Financially costly tools are blocked before OPACA invocation; non-costly state changes can still be speculated under the current policy.",
    }

    output = FULL_IMPLEMENTATION_DIR / "slide_metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    plot_predictors(metrics["predictor_holdout_accuracy"])
    plot_accuracy_by_query_type(metrics)
    plot_predictor_runtime(metrics)

    print(json.dumps(metrics, indent=2))
    print(f"saved -> {output}")
    print(f"saved -> {FULL_IMPLEMENTATION_DIR / 'predictor_holdout_accuracy.png'}")
    print(f"saved -> {FULL_IMPLEMENTATION_DIR / 'predictor_accuracy_by_query_type.png'}")
    print(f"saved -> {FULL_IMPLEMENTATION_DIR / 'predictor_runtime_vs_sage_openai.png'}")


def plot_predictors(metrics):
    habit_key = f"habit_k{HABIT_CACHE_SIZE}"
    names = [f"Habit\nk={HABIT_CACHE_SIZE}", "Naive\nBayes", "SmallLLM\nzero-shot", "SmallLLM\nfew-shot"]
    values = [
        metrics[habit_key]["hit_rate"] * 100,
        metrics["naive_bayes"]["hit_rate"] * 100,
        metrics["small_llm"]["hit_rate"] * 100,
        metrics["few_shot_small_llm"]["hit_rate"] * 100,
    ]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, values, color=["#4a90d9", "#e05c5c", "#4caf50", "#8b5cf6"], alpha=0.9)
    ax.axhline(50, color="#555", linestyle=":", linewidth=1.4, label="50% target")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Tool-sequence accuracy (%)")
    ax.set_title("Predictor accuracy by evaluation protocol")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"{height:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(FULL_IMPLEMENTATION_DIR / "predictor_holdout_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_query_type(metrics):
    accuracy = metrics["predictor_holdout_accuracy"]
    savings = metrics["real_savings_from_holdout"]
    methodology = metrics["methodology"]
    habit_key = f"habit_k{HABIT_CACHE_SIZE}"
    names = [f"Habit k={HABIT_CACHE_SIZE}", "Naive Bayes", "SmallLLM zero-shot", "SmallLLM few-shot"]
    keys = [habit_key, "naive_bayes", "small_llm", "few_shot_small_llm"]
    series = [
        (
            "Overall",
            [accuracy[key]["hit_rate"] * 100 for key in keys],
            [savings[key]["overall_ms"] for key in keys],
            "#2563eb",
        ),
        (
            "Simple",
            [accuracy[key]["simple"]["hit_rate"] * 100 for key in keys],
            [savings[key]["simple_ms"] for key in keys],
            "#16a34a",
        ),
        (
            "Complex",
            [accuracy[key]["complex"]["hit_rate"] * 100 for key in keys],
            [savings[key]["complex_ms"] for key in keys],
            "#f97316",
        ),
    ]
    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.8))
    offsets = [-width, 0, width]
    for offset, (label, values, ms_values, color) in zip(offsets, series):
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.9)
        for bar, value, ms_value in zip(bars, values, ms_values):
            if value <= 0:
                continue
            y = value + 1.4 if value < 92 else value - 3.0
            va = "bottom" if value < 92 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.1f}%",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
            )
    ax.axhline(50, color="#374151", linestyle=":", linewidth=1.4, label="50% target")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Tool-sequence accuracy (%)")
    ax.set_title(
        "Predictor Accuracy by Query Type\n"
        f"NB + few-shot: {methodology['naive_bayes_train_prompts']} train / {methodology['naive_bayes_test_prompts']} held-out "
        f"(simple {methodology['naive_bayes_simple_test_prompts']}, complex {methodology['naive_bayes_complex_test_prompts']}); "
        f"Habit + zero-shot: full benchmark n={methodology['total_prompts']}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(FULL_IMPLEMENTATION_DIR / "predictor_accuracy_by_query_type.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_predictor_runtime(metrics):
    accuracy = metrics["predictor_holdout_accuracy"]
    sage_runtime = metrics["sage_openai_tool_selection_runtime_ms"]
    habit_key = f"habit_k{HABIT_CACHE_SIZE}"
    names = [
        f"SAGE++:\nHabit k={HABIT_CACHE_SIZE}",
        "SAGE++:\nNaive Bayes",
        "SAGE++:\nSmallLLM zero-shot",
        "SAGE++:\nSmallLLM few-shot",
        "SAGE OpenAI",
    ]
    means = [
        accuracy[habit_key]["predictor_runtime_ms"]["mean"],
        accuracy["naive_bayes"]["predictor_runtime_ms"]["mean"],
        accuracy["small_llm"]["predictor_runtime_ms"]["mean"],
        accuracy["few_shot_small_llm"]["predictor_runtime_ms"]["mean"],
        sage_runtime["mean"],
    ]
    p95s = [
        accuracy[habit_key]["predictor_runtime_ms"]["p95"],
        accuracy["naive_bayes"]["predictor_runtime_ms"]["p95"],
        accuracy["small_llm"]["predictor_runtime_ms"]["p95"],
        accuracy["few_shot_small_llm"]["predictor_runtime_ms"]["p95"],
        sage_runtime["p95"],
    ]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#8b5cf6", "#111827"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(x, means, color=colors, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.03, max(means) * 14)
    ax.set_ylabel("Runtime per tool-selection decision (milliseconds)")
    ax.set_title("Predictors Run Thousands of Times Faster Than SAGE OpenAI Tool Selection")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.grid(axis="y", alpha=0.22, which="major")
    ax.grid(False, which="minor")
    ax.tick_params(axis="y", which="minor", length=0)
    for bar, mean, p95 in zip(bars, means, p95s):
        label = f"mean {mean:.3f}ms\np95 {p95:.3f}ms" if mean < 10 else f"mean {mean:,.0f}ms\np95 {p95:,.0f}ms"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean * 1.35,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
        )
    fig.tight_layout()
    fig.savefig(FULL_IMPLEMENTATION_DIR / "predictor_runtime_vs_sage_openai.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
