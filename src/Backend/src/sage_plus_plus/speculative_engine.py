
import logging
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
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
            predicted_calls = list(prediction.calls)
            if not predicted_calls:
                loop.call_soon_threadsafe(prediction_future.set_result, None)
                return

            unsafe = [call.name for call in predicted_calls if not hazard_unit.is_safe(call.name)]
            if unsafe:
                logger.debug(f"[HAZARD_BLOCK] Tools {unsafe} are unsafe, skipping speculation")
                loop.call_soon_threadsafe(prediction_future.set_result, None)
                return
            loop.call_soon_threadsafe(prediction_future.set_result, prediction)

            if cancel_event.is_set():
                return

            def invoke_predicted_call(call):
                if cancel_event.is_set():
                    return None
                if "--" in call.name:
                    agent_name, action_name = call.name.split("--", 1)
                else:
                    agent_name, action_name = None, call.name

                agent_path = f"/{agent_name}" if agent_name else ""
                url = f"{opaca_client.url}/invoke/{action_name}{agent_path}"
                args = call.args or {}
                with httpx.Client() as client:
                    response = client.post(
                        url,
                        json=args,
                        headers=opaca_client._headers(),
                        timeout=self.timeout,
                    )
                response.raise_for_status()
                return (call.name, args, response.json())

            with ThreadPoolExecutor(max_workers=max(len(predicted_calls), 1)) as executor:
                stored_calls = list(executor.map(invoke_predicted_call, predicted_calls))
            if any(call is None for call in stored_calls):
                logger.debug("[SHADOW_CANCELLED] MISMATCH detected, discarding speculative sequence")
                return
            if cancel_event.is_set():
                logger.debug("[SHADOW_CANCELLED] MISMATCH detected, discarding speculative sequence")
                return

            rob.store_many(stored_calls)
            logger.debug("[SHADOW_HIT] Speculative invocation succeeded for %s", [call.name for call in predicted_calls])

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
