"""
Abstract base class for tool predictors.
To be implemented by Rabia.
"""

from abc import ABC, abstractmethod
#from __future__ import annotations
 
import logging
import re
from collections import OrderedDict
from typing import Optional
 
logger = logging.getLogger(__name__)

def _make_signature(subtask: str) -> str:
    """
    Reduce the full subtask string to a compact cache key.
 
    The subtask from speculative_agent_trio.py is multi-line:
        "{task}\n\n{orchestrator_context}\n{round_context}"
 
    Only the first line captures the user intent. Numeric tokens
    (room numbers, item IDs) are stripped so "room 101" and "room 303"
    collapse to the same signature and share frequency counts.
    """
    first_line = subtask.strip().split("\n")[0].strip()
    tokens = first_line.lower().split()
    tokens = [t for t in tokens if not re.fullmatch(r"\d+", t)]
    return " ".join(tokens[:5])

class BasePredictor(ABC):
    """Abstract interface for tool prediction"""
    
    @abstractmethod
    def predict(self, subtask: str) -> str:
        """
        Predict which tool will be called for a given subtask.
        
        Args:
            subtask: The task description from the orchestrator
            
        Returns:
            Predicted tool name
        """
        pass

class DummyPredictor(BasePredictor):
    """
        Mock predictor to unblock Kaveen.Delete this later.
    """
    def __init__(self, mock_output = 'dummy'):
        self._mock_output = mock_output

    def predict(self, subtask: str) -> str:
        return self._mock_output

class NaiveBayesPredictor(BasePredictor):
    def __init__(
        self,
        training_data: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
 
        self._pipeline = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),   # unigrams and bigrams
                max_features=5_000,
                sublinear_tf=True,    # log(1+tf) damping
            )),
            ("clf", MultinomialNB(alpha=1.0)),  # alpha = Laplace smoothing
        ])
 
        self._is_trained: bool = False
 
        if training_data:
            subtasks = [s for s, _ in training_data]
            tool_names = [t for _, t in training_data]
            self.fit(subtasks, tool_names)
 
    def predict(self, subtask: str) -> str:
        if not self._is_trained:
            logger.warning("[NaiveBayesPredictor] predict() called before fit() — returning ''.")
            return ""
 
        # Use only the first line — ignore orchestrator context below it
        first_line = subtask.strip().split("\n")[0].strip()
        prediction = self._pipeline.predict([first_line])[0]
 
        logger.debug(f"[NaiveBayesPredictor] '{prediction}' for '{first_line[:80]}'")
        return str(prediction)
 
    def fit(self, subtasks: list[str], tool_names: list[str]) -> None:
        """
        Train the classifier on (subtask, tool_name) pairs.
 
        Call once with the full benchmark log before the run starts.
        To retrain with new data, call fit() again with the extended dataset.
 
        Args:
            subtasks:   List of raw subtask strings.
            tool_names: Parallel list of ground-truth tool names.
        """
        if len(subtasks) != len(tool_names):
            raise ValueError(
                f"subtasks and tool_names must be the same length "
                f"(got {len(subtasks)} and {len(tool_names)})"
            )
        if not subtasks:
            raise ValueError("Training data cannot be empty.")
 
        # Strip orchestrator context before training, matching what predict() does
        first_lines = [s.strip().split("\n")[0].strip() for s in subtasks]
        self._pipeline.fit(first_lines, tool_names)
        self._is_trained = True
 
        logger.info(
            f"[NaiveBayesPredictor] Trained on {len(subtasks)} examples, "
            f"{len(set(tool_names))} unique tools."
        )
 
    def update(self, subtask: str, tool_name: str) -> None:
        """
        Store a confirmed pair for use in the next fit() call.
 
        MultinomialNB does not support incremental updates — data is
        accumulated here and consumed on the next call to fit().
        """
        if not tool_name:
            return
        if not hasattr(self, "_pending_updates"):
            self._pending_updates: list[tuple[str, str]] = []
        self._pending_updates.append((subtask, tool_name))
 
    def predict_proba(self, subtask: str) -> dict[str, float]:
        """
        Return a probability distribution over all known tool names.
        Useful for ablation studies. Returns empty dict if untrained.
        """
        if not self._is_trained:
            return {}
        first_line = subtask.strip().split("\n")[0].strip()
        clf = self._pipeline.named_steps["clf"]
        proba_array = self._pipeline.predict_proba([first_line])[0]
        proba_dict = dict(zip(clf.classes_, proba_array))
        return dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))


class HabitPredictor(BasePredictor):
    def __init__(
        self,
        cache_size: int = 10,
        training_data: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
 
        self._cache_size = cache_size
        # OrderedDict: signature -> {tool_name: count}
        # Preserves insertion order for O(1) LRU eviction via popitem(last=False)
        # and O(1) promotion via move_to_end().
        self._cache: OrderedDict[str, dict[str, int]] = OrderedDict()
 
        if training_data:
            for subtask, tool_name in training_data:
                self.update(subtask, tool_name)
            logger.info(
                f"[HabitPredictor] Pre-loaded {len(training_data)} pairs "
                f"(cache_size={cache_size})."
            )
 
    def predict(self, subtask: str) -> str:
        sig = _make_signature(subtask)
 
        if sig not in self._cache:
            logger.debug(f"[HabitPredictor] Cold start for: '{sig}'")
            return ""
 
        self._cache.move_to_end(sig)
        counts = self._cache[sig]
        best_tool = max(counts, key=counts.get)
 
        logger.debug(f"[HabitPredictor] '{best_tool}' (count={counts[best_tool]}) for '{sig}'")
        return best_tool
 
    def update(self, subtask: str, tool_name: str) -> None:
        """
        Record a confirmed (subtask, tool_name) pair into the cache.
 
        Call after every confirmed Worker Tool Predictor decision so the
        cache improves during the current benchmark run.
        """
        if not tool_name:
            return
 
        sig = _make_signature(subtask)
 
        if sig in self._cache:
            self._cache.move_to_end(sig)
            self._cache[sig][tool_name] = self._cache[sig].get(tool_name, 0) + 1
        else:
            self._cache[sig] = {tool_name: 1}
            if len(self._cache) > self._cache_size:
                evicted, _ = self._cache.popitem(last=False)
                logger.debug(f"[HabitPredictor] Evicted LRU entry: '{evicted}'")


class SmallLLMPredictor(BasePredictor):
    _DEFAULT_MODEL = "typeform/distilbert-base-uncased-mnli"
 
    def __init__(
        self,
        candidate_tools: Optional[list[str]] = None,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._model_name = model_name
        self._candidate_tools: list[str] = candidate_tools or []
        self._pipeline = None  # loaded lazily on first predict()
 
        if self._candidate_tools:
            logger.info(
                f"[SmallLLMPredictor] {len(self._candidate_tools)} candidate tools registered. "
                f"Model loads on first predict() call."
            )
 
    def predict(self, subtask: str) -> str:
        if not self._candidate_tools:
            logger.warning("[SmallLLMPredictor] candidate_tools not set — returning ''.")
            return ""
 
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
 
        first_line = subtask.strip().split("\n")[0].strip()
        result = self._pipeline(
            sequences=first_line,
            candidate_labels=self._candidate_tools,
            multi_label=False,
        )
 
        best_tool = result["labels"][0]
        best_score = result["scores"][0]
 
        logger.debug(f"[SmallLLMPredictor] '{best_tool}' (score={best_score:.3f}) for '{first_line[:80]}'")
        return best_tool
 
    def update(self, subtask: str, tool_name: str) -> None:
        """No-op — zero-shot models do not retrain from examples."""
        pass
 
    def set_candidate_tools(self, tools: list[str]) -> None:
        """
        Register or update the OPACA tool name list after initialization.
 
        Args:
            tools: All tool name strings from the OPACA platform.
        """
        if not tools:
            raise ValueError("candidate_tools cannot be empty.")
        self._candidate_tools = tools
        logger.info(f"[SmallLLMPredictor] Updated to {len(tools)} candidate tools.")
 
    def predict_top_k(self, subtask: str, k: int = 3) -> list[tuple[str, float]]:
        """
        Return top-k tool predictions with confidence scores.
        Useful for ablation studies. Not called by the engine.
        """
        if not self._candidate_tools:
            return []
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        first_line = subtask.strip().split("\n")[0].strip()
        result = self._pipeline(
            sequences=first_line,
            candidate_labels=self._candidate_tools,
            multi_label=False,
        )
        return list(zip(result["labels"], result["scores"]))[:k]
 
    def _load_pipeline(self):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "SmallLLMPredictor requires 'transformers'.\n"
                "Run: pip install transformers torch"
            ) from exc
 
        logger.info(f"[SmallLLMPredictor] Loading '{self._model_name}' ...")
        return hf_pipeline(
            task="zero-shot-classification",
            model=self._model_name,
            device=-1,  # CPU. Set device=0 for GPU.
        )