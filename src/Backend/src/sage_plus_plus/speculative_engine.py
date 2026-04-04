"""
Speculative execution engine managing shadow thread lifecycle.
"""

import asyncio
import logging
from typing import Any, Optional

import httpx

from .predictor.algorithms import BasePredictor
from .hazard_detection import HazardDetectionUnit
from .reorder_buffer import ReorderBuffer

logger = logging.getLogger(__name__)


class SpeculativeExecutionEngine:
    """Manages shadow thread for speculative tool execution.

    Two-phase design:
      Phase 1 (fast): predict tool name, signal via prediction_ready future
      Phase 2 (slow): invoke the tool, store result in ROB

    The main thread awaits prediction_ready (resolves almost instantly since
    predict() is synchronous), then starts call_llm(). By the time the LLM
    responds, Phase 2 has been running in parallel. The main thread then either:
      - MATCH: awaits the shadow task to finish and commits from ROB
      - MISS:  cancels the shadow task immediately, no wasted work
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def run_speculative(
        self,
        subtask: str,
        opaca_client: Any,
        hazard_unit: HazardDetectionUnit,
        predictor: BasePredictor,
        rob: ReorderBuffer,
        prediction_ready: asyncio.Future,
    ) -> Optional[str]:
        """
        Run speculative execution in background.

        Signals prediction_ready as soon as the tool name is known (before
        invocation starts), so the main thread can compare names without
        waiting for the network call.

        Returns:
            Predicted tool name on successful invocation, None otherwise.
        """
        try:
            # Phase 1: predict (fast, synchronous)
            predicted_tool = predictor.predict(subtask)

            if not hazard_unit.is_safe(predicted_tool):
                logger.debug(f"[HAZARD_BLOCK] Tool {predicted_tool} is unsafe, skipping speculation")
                prediction_ready.set_result(None)
                return None

            # Signal the main thread — name is known, invocation about to start
            prediction_ready.set_result(predicted_tool)

            # Phase 2: invoke (slow — runs concurrently with main thread's call_llm)
            if "--" in predicted_tool:
                agent_name, action_name = predicted_tool.split("--", 1)
            else:
                agent_name, action_name = None, predicted_tool

            result = await asyncio.wait_for(
                opaca_client.invoke_opaca_action(action_name, agent_name, {}),
                timeout=self.timeout,
            )
            await rob.store(predicted_tool, result)
            logger.debug(f"[SHADOW_HIT] Speculative invocation succeeded for {predicted_tool}")
            return predicted_tool

        except asyncio.CancelledError:
            # Main thread cancelled us (name mismatch) — stop immediately
            if not prediction_ready.done():
                prediction_ready.set_result(None)
            await rob.flush()
            raise

        except asyncio.TimeoutError:
            logger.debug(f"[SHADOW_TIMEOUT] Speculative invocation timed out for {subtask}")
            if not prediction_ready.done():
                prediction_ready.set_result(None)
            await rob.flush()
            return None

        except (httpx.HTTPStatusError, Exception) as e:
            logger.error(f"[SHADOW_ERROR] Speculative execution failed: {e}")
            if not prediction_ready.done():
                prediction_ready.set_result(None)
            await rob.flush()
            return None
