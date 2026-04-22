
from __future__ import annotations

import importlib
import json
import logging
import math
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolPrediction:

    name: str
    args: dict[str, Any] = field(default_factory=dict)


def _first_line(subtask: str) -> str:
    return subtask.strip().split("\n")[0].strip()


def _action_name(tool_name: str) -> str:

    return tool_name.split("--", 1)[1] if "--" in tool_name else tool_name


def _canonicalize_tool(tool_name: str, candidate_tools: list[str] | None = None) -> str:

    if not tool_name:
        return ""
    if not candidate_tools:
        return tool_name
    if tool_name in candidate_tools:
        return tool_name

    wanted = _action_name(tool_name).lower()
    matches = [tool for tool in candidate_tools if _action_name(tool).lower() == wanted]
    return matches[0] if matches else tool_name


def _split_identifier(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    return text


def _tokens(text: str) -> list[str]:
    text = _split_identifier(text).lower()
    return re.findall(r"[a-z0-9]+", text)


def _features(text: str) -> list[str]:
    toks = _tokens(text)
    feats = toks[:]
    feats.extend(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
    return feats


def _make_signature(subtask: str) -> str:

    toks = [t for t in _tokens(_first_line(subtask)) if not t.isdigit()]
    return " ".join(toks[:6])


def _freeze_args(args: dict[str, Any] | None) -> str:
    return json.dumps(args or {}, sort_keys=True, default=str)


def _unfreeze_args(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    except json.JSONDecodeError:
        return {}


def _tool_param_dict(tool: Any) -> dict[str, Any]:
    params = getattr(tool, "args", []) or []
    return {getattr(param, "key", ""): getattr(param, "value", None) for param in params if getattr(param, "key", "")}


def _find_benchmark_dir() -> Optional[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "benchmark"
        if (candidate / "question_sets").exists():
            return candidate
    return None


def load_benchmark_training_data() -> list[tuple[str, str, dict[str, Any]]]:

    benchmark_dir = _find_benchmark_dir()
    if benchmark_dir is None:
        return []

    inserted = False
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))
        inserted = True

    try:
        simple = importlib.import_module("question_sets.simple").simple_questions
        complex_ = importlib.import_module("question_sets.complex").complex_questions
    except Exception as exc:
        logger.warning("[Predictor] Could not load benchmark training data: %s", exc)
        return []
    finally:
        if inserted:
            try:
                sys.path.remove(str(benchmark_dir))
            except ValueError:
                pass

    examples: list[tuple[str, str, dict[str, Any]]] = []
    for question in [*simple, *complex_]:
        text = question.get("input", "")
        tools = [tool for tool in question.get("tools", []) if not getattr(tool, "optional", False)]
        if not text or not tools:
            continue
        first = tools[0]
        examples.append((text, first.name, _tool_param_dict(first)))
        for tool in tools:
            args = _tool_param_dict(tool)
            arg_text = " ".join(f"{key} {value}" for key, value in args.items())
            examples.append((f"use {tool.name} {arg_text} for: {text}", tool.name, args))

    logger.info("[Predictor] Loaded %d benchmark training examples.", len(examples))
    return examples


class BasePredictor(ABC):

    def set_candidate_tools(self, tools: list[str]) -> None:
        self._candidate_tools = tools

    def update(self, subtask: str, tool_name: str, args: Optional[dict[str, Any]] = None) -> None:
        pass

    def predict_call(self, subtask: str) -> ToolPrediction:
        return ToolPrediction(self.predict(subtask), {})

    @abstractmethod
    def predict(self, subtask: str) -> str:
        pass


class DummyPredictor(BasePredictor):

    def __init__(self, mock_output: str = "dummy"):
        self._mock_output = mock_output
        self._candidate_tools: list[str] = []

    def predict(self, subtask: str) -> str:
        return _canonicalize_tool(self._mock_output, self._candidate_tools)


class NaiveBayesPredictor(BasePredictor):

    def __init__(
        self,
        training_data: Optional[list[tuple[str, str] | tuple[str, str, dict[str, Any]]]] = None,
    ) -> None:
        self._candidate_tools: list[str] = []
        self._class_doc_counts: Counter[str] = Counter()
        self._feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._feature_totals: Counter[str] = Counter()
        self._vocabulary: set[str] = set()
        self._arg_memory: dict[str, Counter[str]] = defaultdict(Counter)
        self._pending_updates: list[tuple[str, str, dict[str, Any]]] = []
        self._is_trained = False

        data = training_data
        if data is None:
            data = load_benchmark_training_data()
        if data:
            self.fit(data)

    def set_candidate_tools(self, tools: list[str]) -> None:
        self._candidate_tools = tools

    def fit(
        self,
        training_data_or_subtasks: list[tuple[str, str] | tuple[str, str, dict[str, Any]]] | list[str],
        tool_names: Optional[list[str]] = None,
    ) -> None:
        if tool_names is not None:
            training_data = [(subtask, tool, {}) for subtask, tool in zip(training_data_or_subtasks, tool_names)]
        else:
            training_data = [
                (item[0], item[1], item[2] if len(item) > 2 else {})
                for item in training_data_or_subtasks
            ]

        if not training_data:
            raise ValueError("Training data cannot be empty.")

        self._class_doc_counts.clear()
        self._feature_counts.clear()
        self._feature_totals.clear()
        self._vocabulary.clear()
        self._arg_memory.clear()

        for subtask, tool_name, args in training_data:
            label = _action_name(tool_name)
            feats = _features(_first_line(subtask))
            self._class_doc_counts[label] += 1
            self._feature_counts[label].update(feats)
            self._feature_totals[label] += len(feats)
            self._vocabulary.update(feats)
            if args:
                self._arg_memory[label].update([_freeze_args(args)])

        self._is_trained = True
        logger.info(
            "[NaiveBayesPredictor] Trained on %d examples, %d unique tools.",
            len(training_data),
            len(self._class_doc_counts),
        )

    def partial_fit_pending(self) -> None:

        if not self._pending_updates:
            return
        existing = []
        for label, count in self._class_doc_counts.items():
            for _ in range(count):
                existing.append((label, label, {}))
        self.fit([*existing, *self._pending_updates])
        self._pending_updates.clear()

    def predict(self, subtask: str) -> str:
        return self.predict_call(subtask).name

    def predict_call(self, subtask: str) -> ToolPrediction:
        if not self._is_trained:
            logger.warning("[NaiveBayesPredictor] predict() called before fit() - returning ''.")
            return ToolPrediction("", {})

        feats = _features(_first_line(subtask))
        total_docs = sum(self._class_doc_counts.values())
        vocab_size = max(len(self._vocabulary), 1)

        best_label = ""
        best_score = -math.inf
        candidate_labels = {_action_name(tool) for tool in self._candidate_tools} if self._candidate_tools else None
        labels = [
            label for label in self._class_doc_counts
            if candidate_labels is None or label in candidate_labels
        ] or list(self._class_doc_counts)

        for label in labels:
            doc_count = self._class_doc_counts[label]
            score = math.log(doc_count / total_docs)
            denom = self._feature_totals[label] + vocab_size
            counts = self._feature_counts[label]
            for feat in feats:
                score += math.log((counts[feat] + 1) / denom)
            if score > best_score:
                best_label = label
                best_score = score

        tool = _canonicalize_tool(best_label, self._candidate_tools)
        return ToolPrediction(tool, self._infer_args(best_label, subtask))

    def update(self, subtask: str, tool_name: str, args: Optional[dict[str, Any]] = None) -> None:
        if not tool_name:
            return
        self._pending_updates.append((subtask, _action_name(tool_name), args or {}))
        label = _action_name(tool_name)
        feats = _features(_first_line(subtask))
        self._class_doc_counts[label] += 1
        self._feature_counts[label].update(feats)
        self._feature_totals[label] += len(feats)
        self._vocabulary.update(feats)
        if args:
            self._arg_memory[label].update([_freeze_args(args)])
        self._is_trained = True

    def predict_proba(self, subtask: str) -> dict[str, float]:
        if not self._is_trained:
            return {}
        raw_scores = {}
        feats = _features(_first_line(subtask))
        total_docs = sum(self._class_doc_counts.values())
        vocab_size = max(len(self._vocabulary), 1)
        candidate_labels = {_action_name(tool) for tool in self._candidate_tools} if self._candidate_tools else None
        labels = [
            label for label in self._class_doc_counts
            if candidate_labels is None or label in candidate_labels
        ] or list(self._class_doc_counts)

        for label in labels:
            doc_count = self._class_doc_counts[label]
            score = math.log(doc_count / total_docs)
            denom = self._feature_totals[label] + vocab_size
            counts = self._feature_counts[label]
            for feat in feats:
                score += math.log((counts[feat] + 1) / denom)
            raw_scores[label] = score

        max_score = max(raw_scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in raw_scores.items()}
        total = sum(exp_scores.values()) or 1.0
        return dict(sorted(((label, score / total) for label, score in exp_scores.items()), key=lambda x: x[1], reverse=True))

    def _infer_args(self, label: str, subtask: str) -> dict[str, Any]:
        text = _first_line(subtask)
        memory = self._arg_memory.get(label)
        if memory and len(memory) == 1:
            return _unfreeze_args(next(iter(memory)))

        return _heuristic_args(label, text)


class HabitPredictor(BasePredictor):
    def __init__(
        self,
        cache_size: int = 10,
        training_data: Optional[list[tuple[str, str] | tuple[str, str, dict[str, Any]]]] = None,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")

        self._cache_size = cache_size
        self._candidate_tools: list[str] = []
        self._cache: OrderedDict[str, Counter[str]] = OrderedDict()
        self._args: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self._recent_tools: list[str] = []

        if training_data:
            for item in training_data:
                subtask, tool_name = item[0], item[1]
                args = item[2] if len(item) > 2 else {}
                self.update(subtask, tool_name, args)

    def set_candidate_tools(self, tools: list[str]) -> None:
        self._candidate_tools = tools

    def predict(self, subtask: str) -> str:
        return self.predict_call(subtask).name

    def predict_call(self, subtask: str) -> ToolPrediction:
        sig = _make_signature(subtask)
        if sig not in self._cache:
            candidate_actions = {_action_name(tool) for tool in self._candidate_tools} if self._candidate_tools else None
            for action in self._recent_tools:
                if candidate_actions is None or action in candidate_actions:
                    tool = _canonicalize_tool(action, self._candidate_tools)
                    return ToolPrediction(tool, _heuristic_args(action, subtask))
            return ToolPrediction("", {})

        self._cache.move_to_end(sig)
        tool_action = self._cache[sig].most_common(1)[0][0]
        tool = _canonicalize_tool(tool_action, self._candidate_tools)
        arg_counts = self._args.get((sig, tool_action), Counter())
        args = _unfreeze_args(arg_counts.most_common(1)[0][0]) if arg_counts else _heuristic_args(tool_action, subtask)
        return ToolPrediction(tool, args)

    def update(self, subtask: str, tool_name: str, args: Optional[dict[str, Any]] = None) -> None:
        if not tool_name:
            return

        sig = _make_signature(subtask)
        action = _action_name(tool_name)

        if sig in self._cache:
            self._cache.move_to_end(sig)
        else:
            self._cache[sig] = Counter()
            if len(self._cache) > self._cache_size:
                evicted, _ = self._cache.popitem(last=False)
                for key in list(self._args):
                    if key[0] == evicted:
                        del self._args[key]
        self._cache[sig][action] += 1
        if action in self._recent_tools:
            self._recent_tools.remove(action)
        self._recent_tools.insert(0, action)
        if len(self._recent_tools) > self._cache_size:
            self._recent_tools.pop()
        if args:
            self._args[(sig, action)].update([_freeze_args(args)])


class SmallLLMPredictor(BasePredictor):
    _DEFAULT_MODEL = "typeform/distilbert-base-uncased-mnli"

    def __init__(
        self,
        candidate_tools: Optional[list[str]] = None,
        model_name: str = _DEFAULT_MODEL,
        training_data: Optional[list[tuple[str, str] | tuple[str, str, dict[str, Any]]]] = None,
    ) -> None:
        self._model_name = model_name
        self._candidate_tools: list[str] = candidate_tools or []
        self._pipeline = None
        self._transformers_unavailable = False
        self._examples = training_data if training_data is not None else load_benchmark_training_data()

    def set_candidate_tools(self, tools: list[str]) -> None:
        self._candidate_tools = tools

    def predict(self, subtask: str) -> str:
        return self.predict_call(subtask).name

    def predict_call(self, subtask: str) -> ToolPrediction:
        if not self._candidate_tools:
            logger.warning("[SmallLLMPredictor] candidate_tools not set - returning ''.")
            return ToolPrediction("", {})

        if not self._transformers_unavailable:
            try:
                if self._pipeline is None:
                    self._pipeline = self._load_pipeline()
                labels = [_action_name(tool) for tool in self._candidate_tools]
                result = self._pipeline(
                    sequences=_first_line(subtask),
                    candidate_labels=labels,
                    multi_label=False,
                )
                best_label = result["labels"][0]
                tool = _canonicalize_tool(best_label, self._candidate_tools)
                return ToolPrediction(tool, _heuristic_args(best_label, subtask))
            except ImportError:
                self._transformers_unavailable = True
                logger.info("[SmallLLMPredictor] transformers/torch unavailable; using semantic scorer.")
            except Exception as exc:
                self._transformers_unavailable = True
                logger.warning("[SmallLLMPredictor] zero-shot model unavailable (%s); using semantic scorer.", exc)

        return self._semantic_predict(subtask)

    def predict_top_k(self, subtask: str, k: int = 3) -> list[tuple[str, float]]:
        scores = self._semantic_scores(subtask)
        return scores[:k]

    def _semantic_predict(self, subtask: str) -> ToolPrediction:
        scores = self._semantic_scores(subtask)
        if not scores:
            return ToolPrediction("", {})
        tool, _score = scores[0]
        return ToolPrediction(tool, _heuristic_args(_action_name(tool), subtask))

    def _semantic_scores(self, subtask: str) -> list[tuple[str, float]]:
        text_feats = Counter(_features(_first_line(subtask)))
        if not text_feats:
            return []

        scores: dict[str, float] = {}
        for tool in self._candidate_tools:
            action = _action_name(tool)
            label_feats = Counter(_features(action))
            score = _cosine(text_feats, label_feats) * 2.0
            for example_text, example_tool, _args in self._examples:
                if _action_name(example_tool).lower() != action.lower():
                    continue
                score = max(score, _cosine(text_feats, Counter(_features(example_text))))
            aliases = _TOOL_ALIASES.get(action, ())
            for alias in aliases:
                score = max(score, _cosine(text_feats, Counter(_features(alias))) * 1.5)
            scores[tool] = score

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def _load_pipeline(self):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError("SmallLLMPredictor requires 'transformers' and 'torch'.") from exc

        logger.info("[SmallLLMPredictor] Loading '%s' ...", self._model_name)
        return hf_pipeline(task="zero-shot-classification", model=self._model_name, device=-1)


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[key] * right.get(key, 0) for key in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _heuristic_args(tool_name: str, subtask: str) -> dict[str, Any]:

    action = _action_name(tool_name)
    text = _first_line(subtask)
    lower = text.lower()
    ints = [int(value) for value in re.findall(r"\b\d+\b", text)]

    def first_int(default: Any = None) -> Any:
        return ints[0] if ints else default

    args: dict[str, Any] = {}
    numeric_keys = {
        "GetRoomName": "room_id",
        "CheckAvailability": "room_id",
        "BookRoom": "room_id",
        "TurnOnLights": "room_id",
        "TurnOffLights": "room_id",
        "CheckSensorBattery": "room_id",
        "GetCompleteInfo": "room_id",
        "IsFree": "desk",
        "BookDesk": "desk",
        "CheckDeviceHealth": "device_id",
        "GetLastMaintenanceDate": "device_id",
        "RestartDevice": "device_id",
        "ScheduleMaintenance": "device_id",
        "PlayTrack": "track_id",
        "GetTrackById": "track_id",
        "GetPlaylistSongs": "playlist_id",
        "RenamePlaylist": "playlist_id",
        "DeletePlaylist": "playlist_id",
        "GetWarehouseZoneSize": "zone",
        "GetInventory": "zone",
    }

    if action in {"GetWarehouseZoneSize", "GetInventory"}:
        zone = _extract_zone(text)
        if zone is not None:
            args[numeric_keys[action]] = zone
    elif action in numeric_keys and ints:
        key = numeric_keys[action]
        args[key] = first_int()

    if action == "SetLightIntensity":
        if ints:
            args["room_id"] = ints[0]
        pct = re.search(r"\b(\d{1,3})\s*%", text)
        if pct:
            args["intensity"] = int(pct.group(1)) / 100
        elif len(ints) > 1:
            args["intensity"] = ints[1]

    if action in {"AdjustVolume"} and ints:
        args["volume"] = ints[-1]

    if action in {"GetRoomId"}:
        match = re.search(r"(?:id of|id for|room id of)\s+(?:the\s+)?['\"]?([a-z][a-z ]+?)(?:\s+room)?['\"]?(?:$|[.,])", lower)
        if not match:
            match = re.search(r"(?:room|space)\s+(?:named|called)?\s*['\"]?([a-z][a-z ]+?)['\"]?(?:$|[.,])", lower)
        if match:
            args["room_name"] = match.group(1).strip()

    if action in {"GetDeviceId"}:
        for name in ("router", "camera", "thermostat", "filter", "network", "security"):
            if name in lower:
                args["device_name"] = name
                break

    if action in {"GetItemLocation", "PickupItem", "DropItem", "MakeOrder", "AddToGroceryList"}:
        quoted = re.search(r"['\"]([^'\"]+)['\"]", text)
        if quoted:
            args["item"] = quoted.group(1)

    if action in {"GetIdByTrack"}:
        quoted = re.search(r"['\"]([^'\"]+)['\"]", text)
        if quoted:
            args["track"] = quoted.group(1)

    return args


def _extract_zone(text: str) -> Optional[str]:
    match = re.search(r"\bzone[- ]?([a-z])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


_TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "GetInventory": ("stock level inventory count warehouse items in zone",),
    "GetWarehouseZoneSize": ("zone size capacity warehouse area",),
    "GetRoomName": ("name of room room id",),
    "GetRoomId": ("id of room room name",),
    "CheckAvailability": ("is room free available booking",),
    "IsFree": ("is desk free available bookable",),
    "RunFullSystemCheck": ("system check device status health overview",),
    "CheckDeviceHealth": ("device health damaged functioning status",),
    "GetCurrentVolume": ("music volume current loudness",),
    "AdjustVolume": ("set volume change loudness",),
    "GetTracks": ("songs tracks music list",),
    "GetIdByTrack": ("track id song id by name",),
}
