"""
Speculative execution engine — runs in a GIL-free OS thread.

Two-phase design:
  Phase 1 (fast): predict tool name, signal asyncio event loop via
                  loop.call_soon_threadsafe() so main coroutine can
                  start call_llm() immediately.
  Phase 2 (slow): invoke the tool synchronously via httpx, store
                  result in ROB.

Because this runs in a true OS thread under Python 3.14 free-threaded
mode (no GIL), Phase 1 and the main coroutine's call_llm() execute on
separate CPU cores simultaneously — zero blocking between them.
"""

import logging
import threading
import asyncio
from typing import Any, Optional

import httpx

from .predictor.algorithms import BasePredictor
from .hazard_detection import HazardDetectionUnit
from .reorder_buffer import ReorderBuffer

logger = logging.getLogger(__name__)


class SpeculativeExecutionEngine:
    """Manages shadow OS thread for speculative tool execution."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def run_speculative(
        self,
        subtask: str,
        opaca_client: Any,
        hazard_unit: HazardDetectionUnit,
        predictor: BasePredictor,
        rob: ReorderBuffer,
        loop: asyncio.AbstractEventLoop,
        prediction_future: asyncio.Future,
        cancel_event: threading.Event,
    ) -> None:
        """
        Run speculative execution in a GIL-free OS thread.

        Signals the asyncio event loop via loop.call_soon_threadsafe()
        as soon as the tool name is known (Phase 1), allowing call_llm()
        to start on the main coroutine immediately — true parallelism.

        Args:
            subtask:           Task description
            opaca_client:      OPACA client (for URL and auth headers)
            hazard_unit:       Hazard detection unit
            predictor:         Tool predictor (Rabia's model)
            rob:               Reorder buffer (threading.Lock-based)
            loop:              The asyncio event loop running on the main thread
            prediction_future: asyncio.Future to signal Phase 1 completion
            cancel_event:      threading.Event set by main on MISMATCH
        """
        try:
            # Phase 1: predict (CPU work — runs in parallel with call_llm on another core)
            predicted_tool = predictor.predict(subtask)

            if not hazard_unit.is_safe(predicted_tool):
                logger.debug(f"[HAZARD_BLOCK] Tool {predicted_tool} is unsafe, skipping speculation")
                loop.call_soon_threadsafe(prediction_future.set_result, None)
                return

            # Signal main coroutine — prediction name is ready, Phase 2 starting now
            loop.call_soon_threadsafe(prediction_future.set_result, predicted_tool)

            # Phase 2: invoke synchronously (I/O — runs in parallel with call_llm's I/O)
            if "--" in predicted_tool:
                agent_name, action_name = predicted_tool.split("--", 1)
            else:
                agent_name, action_name = None, predicted_tool

            agent_path = f"/{agent_name}" if agent_name else ""
            url = f"{opaca_client.url}/invoke/{action_name}{agent_path}"

            with httpx.Client() as client: # blocks the shadow thread
                response = client.post(
                    url,
                    json={},
                    headers=opaca_client._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

            # Check if main already cancelled us (MISMATCH) — discard if so
            if cancel_event.is_set():
                logger.debug(f"[SHADOW_CANCELLED] MISMATCH detected, discarding result for {predicted_tool}")
                return

            rob.store(predicted_tool, result)
            logger.debug(f"[SHADOW_HIT] Speculative invocation succeeded for {predicted_tool}")

        except httpx.TimeoutException:
            logger.debug(f"[SHADOW_TIMEOUT] Speculative invocation timed out for {subtask}")
            rob.flush()
            if not prediction_future.done():
                loop.call_soon_threadsafe(prediction_future.set_result, None)

        except httpx.HTTPStatusError as e:
            logger.error(f"[SHADOW_ERROR] OPACA returned error: {e}")
            rob.flush()
            if not prediction_future.done():
                loop.call_soon_threadsafe(prediction_future.set_result, None)

        except Exception as e:
            logger.error(f"[SHADOW_ERROR] Speculative execution failed: {e}")
            rob.flush()
            if not prediction_future.done():
                loop.call_soon_threadsafe(prediction_future.set_result, None)
