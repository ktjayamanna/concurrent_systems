"""
Speculative self-orchestrated method - SAGE++ implementation.

Uses Python 3.14 free-threaded mode (no GIL) to achieve true parallelism:
  - Shadow OS thread: predictor.predict() + synchronous OPACA invocation
  - Main asyncio coroutine: call_llm(WorkerAgent)

Both run on separate CPU cores simultaneously. The asyncio event loop is
bridged to the shadow thread via loop.call_soon_threadsafe() for signalling
and threading.Event for cancellation.
"""

import datetime
import json
import logging
import os
import time
import threading
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..orchestrated import SelfOrchestratedMethod
from ..orchestrated.orchestrated_routes import OrchestrationConfig
from ..orchestrated.agents import AgentEvaluator, AgentPlanner, get_current_time
from ..orchestrated.models import AgentResult, AgentTask
from ..orchestrated.prompts import BACKGROUND_INFO, GENERAL_CAPABILITIES_RESPONSE
from ..models import QueryResponse, Chat, AgentMessage, ToolCall, MethodConfig

from .speculative_engine import SpeculativeExecutionEngine
from .reorder_buffer import ReorderBuffer
from .predictor.algorithms import DummyPredictor, HabitPredictor, NaiveBayesPredictor, SmallLLMPredictor
from .hazard_detection import HazardDetectionUnit

logger = logging.getLogger(__name__)


class SagePlusPlusConfig(OrchestrationConfig):
    """Extends the base orchestration config with SAGE++-specific settings."""
    predictor_type: str = MethodConfig.string(
        default="habit",
        options=["habit", "naive-bayes", "small-llm"],
        allow_free_input=False,
        title="Predictor Type",
        description="Tool prediction algorithm used by SAGE++ shadow thread",
    )


class SpeculativeSelfOrchestratedMethod(SelfOrchestratedMethod):
    """
    SAGE++ — true parallel speculative execution via GIL-free OS threads.

    Shadow OS thread (Core N):
      predict() → invoke_opaca_action() synchronously

    Main asyncio coroutine (Core 1):
      call_llm(WorkerAgent) — starts at t=0, no blocking from shadow

    After call_llm() returns:
      MATCH → join shadow thread, commit from ROB, skip invoke_tools()
      MISS  → cancel_event.set(), main runs invoke_tools() normally
    """

    NAME = "sage++"
    CONFIG = SagePlusPlusConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speculative_engine = SpeculativeExecutionEngine()
        self.predictor = HabitPredictor()
        self.hazard_unit = HazardDetectionUnit()
        self._metrics: dict = {"attempts": 0, "hits": 0, "misses": 0, "predictions": []}
        self._current_predictor_type: str = "habit"

    @staticmethod
    def _build_predictor(predictor_type: str, candidate_tools: List[str] = None):
        """Factory: return the right BasePredictor for the given predictor_type string."""
        if predictor_type == "naive-bayes":
            return NaiveBayesPredictor()
        elif predictor_type == "small-llm":
            return SmallLLMPredictor(candidate_tools=candidate_tools or [])
        else:  # default: "habit"
            return HabitPredictor()

    @staticmethod
    def _is_error_result(result: Any) -> bool:
        return isinstance(result, str) and result.startswith("Failed")

    def _launch_shadow(
        self,
        subtask: str,
        rob: ReorderBuffer,
        loop: asyncio.AbstractEventLoop,
        prediction_future: asyncio.Future,
        cancel_event: threading.Event,
    ) -> threading.Thread:
        """Launch shadow OS thread. Returns immediately — thread runs in parallel."""
        shadow = threading.Thread(
            target=self.speculative_engine.run_speculative,
            args=(
                subtask,
                self.session.opaca_client,
                self.hazard_unit,
                self.predictor,
                rob,
                loop,
                prediction_future,
                cancel_event,
            ),
            daemon=True,
        )
        shadow.start()
        return shadow

    async def _run_shadow_and_commit(
        self,
        worker_message,
        rob: ReorderBuffer,
        shadow: threading.Thread,
        shadow_prediction: Optional[str],
        cancel_event: threading.Event,
        agent,
        current_task: str,
        agent_messages: List[AgentMessage],
    ) -> AgentResult:
        """
        Given the main thread's LLM result and shadow's predicted name,
        either commit the speculative result (MATCH) or cancel and invoke normally (MISS).
        """
        actual_tool = worker_message.tools[0].name if worker_message.tools else None  # SAGE (ground truth)

        # ── Prediction logging ────────────────────────────────────────────────
        # Determine outcome label for this prediction event
        if shadow_prediction and actual_tool:
            _outcome = "hit" if shadow_prediction == actual_tool else "miss"
        elif shadow_prediction:
            _outcome = "miss"        # predicted but no actual tool called
        else:
            _outcome = "no_prediction"   # shadow returned "" (cold start / unsafe)

        self._metrics["predictions"].append({
            "subtask": current_task.strip().split("\n")[0][:150],
            "predicted": shadow_prediction or "",
            "actual": actual_tool or "",
            "outcome": _outcome,
        })

        # Online learning: tell the predictor what the correct tool was
        if actual_tool and hasattr(self.predictor, "update"):
            self.predictor.update(current_task, actual_tool)
        # ─────────────────────────────────────────────────────────────────────

        # shadow_prediction is the predicted result from sage++
        if shadow_prediction and actual_tool and shadow_prediction == actual_tool:
            # If the shadow prediction matched the LLM’s tool choice, 
            # wait (without blocking the event loop) for the shadow thread
            # to finish its speculative OPACA invocation, but only up to the timeout.
            await asyncio.get_running_loop().run_in_executor(
                None, shadow.join, self.speculative_engine.timeout
            )

            speculative_result = rob.commit(shadow_prediction, actual_tool)

            if speculative_result is not None and not self._is_error_result(speculative_result):
                # HIT — inject result, skip invoke_tools entirely
                self._metrics["hits"] += 1
                # call_llm(...) was supposed to return the LLM’s decision as an AgentMessage. 
                # That message already contains a ToolCall object in worker_message.tools[0], 
                # but without a result yet. self.invoke_tools(...) then executes that tool call 
                # and fills in the result (it usually returns an AgentResult)
                # below we artificially make that tool call with the speculative result.
                worker_message.tools[0] = ToolCall(
                    id=worker_message.tools[0].id,
                    type="opaca",
                    name=actual_tool,
                    args=worker_message.tools[0].args,
                    result=speculative_result,
                )
                agent_messages.append(AgentMessage(agent="WorkerAgent (shadow)", execution_time=0))
                # Once we updated worker_message.tools[0], we wrap if with AgentResult
                # to fool the system to think it's the return call from invoke_tools()
                return AgentResult(
                    agent_name=agent.agent_name,
                    task=current_task,
                    output=f"- Worker Agent Executed: {actual_tool}.",
                    tool_calls=worker_message.tools,
                )

        # MISS — signal shadow to discard its result, run invoke_tools normally
        self._metrics["misses"] += 1
        cancel_event.set()
        rob.flush()
        _tool_start = time.time()
        result = await self.invoke_tools(agent, current_task, worker_message)
        agent_messages.append(AgentMessage(agent="WorkerAgent (tool invoke)", execution_time=time.time() - _tool_start))
        return result

    async def _execute_round(
            self,
            round_tasks: List[AgentTask],
            worker_agents: Dict,
            config: OrchestrationConfig,
            all_results: List[AgentResult],
            agent_messages: List[AgentMessage],
    ) -> List[AgentResult]:
        """
            Speculative version of _execute_round that polymorphically overrides self._execute_round
            at orchestrated_method.py to run true parallel speculative execution using GIL-free OS threads.
        """

        agent_evaluator = AgentEvaluator() if config.use_agent_evaluator else None

        async def speculative_execute_round_task(
                worker_agent,
                subtask: AgentTask,
                orchestrator_context: str,
                round_context: str,
        ) -> AgentResult:
            current_task = f"{subtask.task}\n\n{orchestrator_context}\n{round_context}"

            rob = ReorderBuffer()
            loop = asyncio.get_running_loop() # get the event loop
            prediction_future = loop.create_future() # gather() wait for this future to be resolved instead of monitoring the shadow thread constantly.
            cancel_event = threading.Event()

            # Launch shadow OS thread — runs predict() on a separate core immediately
            shadow = self._launch_shadow(current_task, rob, loop, prediction_future, cancel_event)
            self._metrics["attempts"] += 1

            # Start call_llm immediately — predict() and call_llm I/O run on separate cores simultaneously
            worker_message, shadow_prediction = await asyncio.gather(
                self.call_llm(
                    model=config.worker_model,
                    agent="WorkerAgent",
                    system_prompt=worker_agent.system_prompt(),
                    messages=worker_agent.messages(subtask),
                    temperature=config.temperature,
                    tool_choice="required",
                    tools=worker_agent.tools,
                ),
                prediction_future,
            )

            agent_result = await self._run_shadow_and_commit(
                worker_message=worker_message,
                rob=rob,
                shadow=shadow,
                shadow_prediction=shadow_prediction,
                cancel_event=cancel_event,
                agent=worker_agent,
                current_task=current_task,
                agent_messages=agent_messages,
            )

            agent_messages.append(worker_message)
            return agent_result

        async def execute_single_task(task: AgentTask) -> AgentResult:
            agent = worker_agents[task.agent_name]
            task_str = task.task if isinstance(task, AgentTask) else task

            logger.info(f"Executing task for {task.agent_name}: {task_str}")
            
            # GeneralAgent is a dumb agent that does not do any llm inferencing.
            # It just returns a pre-defined response about the system capabilities.
            if agent.agent_name == "GeneralAgent":
                predefined_response = get_current_time() + BACKGROUND_INFO + GENERAL_CAPABILITIES_RESPONSE.format(
                    agent_capabilities=json.dumps(await self.get_agent_details(), indent=2))
                return AgentResult(
                    agent_name="GeneralAgent",
                    task=task_str,
                    output="Retrieved system capabilities",
                    tool_calls=[ToolCall(id="-1", type="opaca", name="GetCapabilities", args={}, result=predefined_response)],
                )

            if config.use_agent_planner:
                planner = AgentPlanner(
                    agent_name=task.agent_name,
                    tools=agent.tools,
                    worker_agent=agent,
                    config=config,
                )

                planner_message = await self.call_llm(
                    model=config.orchestrator_model,
                    agent="AgentPlanner",
                    system_prompt=planner.system_prompt(),
                    messages=planner.messages(task, previous_results=all_results),
                    temperature=config.temperature,
                    tools=planner.tools,
                    tool_choice="none",
                    response_format=planner.schema,
                    status_message="Planning function calls for {task.agent_name}'s task: {task_str}",
                )
                agent_messages.append(planner_message)
                plan = planner_message.formatted_output

                if not plan:
                    return AgentResult(
                        agent_name=task.agent_name,
                        task=task_str,
                        output="There was an error during the generation of an agent plan!",
                        tool_calls=[],
                    )

                await self.send_status_to_websocket("WorkerAgent", "Executing function calls.\n\n")

                ex_results: List[AgentResult] = []
                tasks_by_round = {}
                for subtask in plan.tasks:
                    tasks_by_round.setdefault(subtask.round, []).append(subtask)

                for round_num in sorted(tasks_by_round.keys()):
                    logger.info(f"AgentPlanner executing round {round_num}")
                    current_tasks = tasks_by_round[round_num]

                    round_context = ""
                    if round_num > 1 and ex_results:
                        round_context = "\n\nPrevious planner round results:\n"
                        for prev_result in ex_results:
                            round_context += f"\nTask: {prev_result.task}\n"
                            round_context += f"Output: {prev_result.output}\n"
                            if any(tc.result for tc in prev_result.tool_calls):
                                round_context += "Tool Results:\n"
                                for tc in prev_result.tool_calls:
                                    round_context += f"- {tc.name}: {tc.result}\n"

                    round_results = await asyncio.gather(*[
                        speculative_execute_round_task(
                            planner.worker_agent, subtask,
                            planner.get_orchestrator_context(all_results), round_context,
                        )
                        for subtask in current_tasks
                    ])
                    ex_results.extend(round_results)

                result = AgentResult(
                    agent_name=planner.worker_agent.agent_name,
                    task=task_str,
                    output="\n\n".join(res.output for res in ex_results),
                    tool_calls=[tc for res in ex_results for tc in res.tool_calls],
                )

            else:  # no planner
                await self.send_status_to_websocket("WorkerAgent", "Executing function calls.\n\n")

                rob = ReorderBuffer()
                loop = asyncio.get_running_loop()
                prediction_future = loop.create_future()
                cancel_event = threading.Event()

                shadow = self._launch_shadow(task_str, rob, loop, prediction_future, cancel_event)
                self._metrics["attempts"] += 1

                # Start call_llm immediately — predict() and call_llm I/O run on separate cores simultaneously
                worker_message, shadow_prediction = await asyncio.gather(
                    self.call_llm(
                        model=config.worker_model,
                        agent="WorkerAgent",
                        system_prompt=agent.system_prompt(),
                        messages=agent.messages(task),
                        temperature=config.temperature,
                        tool_choice="required",
                        tools=agent.tools,
                    ),
                    prediction_future,
                )

                result = await self._run_shadow_and_commit(
                    worker_message=worker_message,
                    rob=rob,
                    shadow=shadow,
                    shadow_prediction=shadow_prediction,
                    cancel_event=cancel_event,
                    agent=agent,
                    current_task=task.task,
                    agent_messages=agent_messages,
                )
                agent_messages.append(worker_message)

            if agent_evaluator:
                if not (should_retry := agent_evaluator.has_error(result)):
                    evaluation_message = await self.call_llm(
                        model=config.evaluator_model,
                        agent="AgentEvaluator",
                        system_prompt=agent_evaluator.system_prompt(),
                        messages=agent_evaluator.messages(task_str, result),
                        temperature=config.temperature,
                        response_format=agent_evaluator.schema,
                        status_message=f"Evaluating {task.agent_name}'s task completion",
                    )
                    agent_messages.append(evaluation_message)
                    should_retry = evaluation_message.formatted_output.reiterate

                if should_retry:
                    retry_task = f"""# Evaluation

The Evaluator of your task has indicated that there is crucial information missing to solve the task..

# Your Task:

{task_str}

# Your previous output:

{result.output}

# Your Previous tool calls:

{[tc.without_id() for tc in result.tool_calls]}

# YOUR GOAL:

Now, using the tools available to you and the previous results, continue with your original task and retrieve all the information necessary to complete and solve the task!"""

                    worker_message = await self.call_llm(
                        model=config.worker_model,
                        agent="WorkerAgent",
                        system_prompt=agent.system_prompt(),
                        messages=agent.messages(retry_task),
                        temperature=config.temperature,
                        tool_choice="required",
                        tools=agent.tools,
                        status_message="Retrying task",
                    )

                    _tool_start = time.time()
                    result = await self.invoke_tools(agent, task.task, worker_message)
                    agent_messages.append(AgentMessage(agent="WorkerAgent (tool invoke)", execution_time=time.time() - _tool_start))
                    agent_messages.append(worker_message)

            return result

        return await asyncio.gather(*[execute_single_task(task) for task in round_tasks])

    async def query(self, message: str, chat: Chat) -> QueryResponse:
        """Process query with speculative execution enabled."""
        self._metrics = {"attempts": 0, "hits": 0, "misses": 0, "predictions": []}

        # Switch predictor if config changed since last query
        config: SagePlusPlusConfig = self.get_config()
        predictor_type = config.predictor_type
        if predictor_type != self._current_predictor_type:
            candidate_tools: List[str] = []
            if predictor_type == "small-llm":
                try:
                    actions = await self.session.opaca_client.get_actions_simple()
                    candidate_tools = [
                        action["name"]
                        for actions_list in actions.values()
                        for action in actions_list
                    ]
                except Exception as e:
                    logger.warning(f"[SAGE++] Could not fetch tool names for SmallLLMPredictor: {e}")
            self.predictor = self._build_predictor(predictor_type, candidate_tools)
            self._current_predictor_type = predictor_type
            logger.info(f"[SAGE++] Switched predictor to '{predictor_type}'")

        response = await super().query(message, chat)
        response.speculative = self._metrics["hits"] > 0
        attempts = self._metrics["attempts"]
        hits = self._metrics["hits"]
        hit_rate = f"{hits / attempts:.1%}" if attempts > 0 else "n/a"
        logger.info(f"[SAGE++] predictor={predictor_type} metrics={self._metrics} hit_rate={hit_rate}")

        # Persist predictions for offline analysis
        self._save_predictions(predictor_type, message)

        return response

    def _save_predictions(self, predictor_type: str, query: str) -> None:
        """Append this query's prediction events to a per-predictor JSONL log file.

        Output directory: <repo>/src/benchmark/benchmark_results/predictions/
        File name:        predictions_<predictor_type>.jsonl

        Each line is a JSON record:
            {timestamp, predictor_type, query, summary: {attempts, hits, misses, hit_rate},
             predictions: [{subtask, predicted, actual, outcome}, ...]}
        """
        try:
            # SAGE_PREDICTIONS_DIR is set via Docker volume mount (see benchmark/docker-compose.yml).
            # Falls back to the local dev path when running the backend directly on the host.
            _default = str(
                Path(__file__).resolve().parent.parent.parent.parent
                / "benchmark" / "benchmark_results" / "predictions"
            )
            output_dir = Path(os.environ.get("SAGE_PREDICTIONS_DIR", _default))
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = output_dir / f"predictions_{predictor_type}.jsonl"

            attempts = self._metrics["attempts"]
            hits = self._metrics["hits"]
            record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "predictor_type": predictor_type,
                "query": query[:200],
                "summary": {
                    "attempts": attempts,
                    "hits": hits,
                    "misses": self._metrics["misses"],
                    "hit_rate": round(hits / attempts, 4) if attempts > 0 else 0.0,
                },
                "predictions": self._metrics["predictions"],
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            logger.info(f"[SAGE++] Prediction log appended → {log_file}")
        except Exception as e:
            logger.warning(f"[SAGE++] Could not save predictions: {e}")
