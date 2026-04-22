
import logging
import threading
import asyncio
from typing import Any, Optional

import httpx

from .predictor.algorithms import BasePredictor, ToolPrediction
from .hazard_detection import HazardDetectionUnit
from .reorder_buffer import ReorderBuffer

logger = logging.getLogger(__name__)


class SpeculativeExecutionEngine:

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
        try:
            if hasattr(predictor, "predict_call"):
                prediction = predictor.predict_call(subtask)
            else:
                prediction = ToolPrediction(predictor.predict(subtask), {})
            predicted_tool = prediction.name
            predicted_args = prediction.args or {}

            if not hazard_unit.is_safe(predicted_tool):
                logger.debug(f"[HAZARD_BLOCK] Tool {predicted_tool} is unsafe, skipping speculation")
                loop.call_soon_threadsafe(prediction_future.set_result, None)
                return
            loop.call_soon_threadsafe(prediction_future.set_result, prediction)

            if cancel_event.is_set():
                return
            if "--" in predicted_tool:
                agent_name, action_name = predicted_tool.split("--", 1)
            else:
                agent_name, action_name = None, predicted_tool

            agent_path = f"/{agent_name}" if agent_name else ""
            url = f"{opaca_client.url}/invoke/{action_name}{agent_path}"

            with httpx.Client() as client:
                response = client.post(
                    url,
                    json=predicted_args,
                    headers=opaca_client._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
            if cancel_event.is_set():
                logger.debug(f"[SHADOW_CANCELLED] MISMATCH detected, discarding result for {predicted_tool}")
                return

            rob.store(predicted_tool, result, predicted_args)
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
