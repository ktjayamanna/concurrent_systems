Class Design Outline (SAGE++)

SpeculativeExecutionEngine (orchestrator)
State
- predictor (pluggable predictor implementation)
- hazard_detection_unit
- reorder_buffer
- timeout_ms (speculative call timeout)
- opaca_interface (to dispatch tool calls)
Behavior
- on_subtask_arrival(subtask): launch speculative shadow thread (FR-01)
- predict_tool_call(context): ask predictor for likely tool call (FR-02/FR-03)
- classify_tool(tool): consult hazard detection unit (FR-04/FR-05)
- dispatch_speculative(tool_call): fire safe tool call in background (FR-06)
- await_or_cancel(timeout): enforce timeout (FR-07)
- commit_or_flush(actual_tool_call): if hit, forward from reorder buffer; if miss, flush and signal executor (FR-08/FR-09/FR-10)

SpeculativeAgentTrio (agent wrapper/entry point)
State
- engine (speculative execution engine instance)
- mode_flag (SAGE vs SAGE++)
Behavior
- handle_subtask(subtask): route into SAGE or SAGE++ flow (FR-11)

Predictor (interface/abstract)
State
- model_or_cache (variant-specific state)
- history (optional for LRU)
Behavior
- predict(context): return likely tool call (FR-02/FR-03)
- update(outcome): optional learning from hits/misses

PredictorVariant: HabitBasedLRU
State
- lru_cache (context -> tool mapping)
Behavior
- predict(context)
- update(outcome)

PredictorVariant: NaiveBayes
State
- classifier_params
- feature_extractor
Behavior
- predict(context)
- update(outcome) (if online)

PredictorVariant: SmallLLM
State
- model_id
- prompt_template
Behavior
- predict(context)

HazardDetectionUnit
State
- tool_safety_table (static lookup loaded at startup)
Behavior
- classify(tool): return Safe/Unsafe (FR-04)
- is_safe(tool): convenience check used by engine (FR-05)

ReorderBuffer
State
- pending_result (speculative tool output)
- pending_tool_call
- lock (asyncio.Lock for thread safety)
Behavior
- commit(predicted_tool, actual_tool) -> result | None: returns result on hit, None on miss (FR-08/FR-09)
- flush() (FR-10, O(1) per NFR-03)
